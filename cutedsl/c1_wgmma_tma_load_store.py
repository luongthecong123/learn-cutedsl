import torch
from typing import Tuple
import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
from cutlass.cute.testing import benchmark, JitArguments

"""
Hopper WGMMA GEMM with TMA load and TMA store, single-stage.

The simplest Hopper kernel using the two key SM90 hardware features:
  1. TMA (Tensor Memory Accelerator) for async GMEM ↔ SMEM copies
  2. WGMMA (Warp Group MMA) for async matrix multiply on tensor cores

ALGORITHM (no warp specialization, no pipeline, no TMA/WGMMA overlap):
  - TMA -> sync -> WGMMA -> sync -> Epilogue (RMEM -> (SMEM ->GMEM with TMA))

SMEM budget (per CTA, single stage):
  A: 128 x 128 x 2B = 32 KB
  B: 128 x 128 x 2B = 32 KB
  Total: 64 KB (1 stage)
"""


class Gemm_TC:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 128),
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.BK = self.tile_shape_mnk
        
        # With (64, 128, 64), atom_layout = (1,1,1) => 1 MMA warp group
        # With (128, 128, 64), atom_layout = (2,1,1) => 2 MMA warp groups        
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.BM > 64 and self.BN > 64
            else (1, 1, 1)
        )
        
        self.warp_size = cute.arch.WARP_SIZE
        self.num_threads_per_warp_group = 128
        
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group

        self.num_stages = 1
        assert self.num_stages == 1, "This script only supports single stage for simplicity"
        
        self.buffer_align_bytes = 1024
        
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor, 
        c: cute.Tensor,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        
        # Create swizzled layout: S<3,4,3> o 0 o ((8,16),(64,1),(1,1)):((64,512),(1,0),(0,0))
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype,
            num_stages=self.num_stages
        )
        print("self.a_smem_layout_staged: ", self.a_smem_layout_staged)

        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype,
            num_stages=self.num_stages
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.BM, self.BK)
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.BN, self.BK)
        )

        c_smem_layout = cute.make_layout(
            (self.BM, self.BN),
            stride=(self.BN, 1)
        )
        
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c,
            c_smem_layout,
            (self.BM, self.BN),
        )

        @cute.struct
        class SharedStorage:
            mbarrier_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # sC: cute.struct.Align[
            #     cute.struct.MemRange[
            #         self.c_dtype, self.BM * self.BN
            #     ],
            #     self.buffer_align_bytes,
            # ]

        self.shared_storage = SharedStorage

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.BN),
        )

        print("tiled_mma: ", self.tiled_mma)
        
        grid_dim = *cute.ceil_div(c.shape, (self.BM, self.BN)), 1
        self.kernel(
            tma_atom_a, 
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            ).launch(
            grid=grid_dim, 
            block=(self.threads_per_cta, 1, 1)
        )
    
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_tma_tensor: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_tma_tensor: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_tma_tensor: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        # ====== Thread, Block setup =======

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)  

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        tile_coord_mnk = (bidx, bidy, None)

        gA = cute.local_tile(
            input=mA_tma_tensor, 
            tiler=self.tile_shape_mnk,
            coord=tile_coord_mnk,
            proj=(1, None, 1))
        
        gB = cute.local_tile(
            input=mB_tma_tensor, 
            tiler=self.tile_shape_mnk,
            coord=tile_coord_mnk,
            proj=(None, 1, 1))
        
        gC = cute.local_tile(
            input=mC, 
            tiler=self.tile_shape_mnk,
            coord=tile_coord_mnk,
            proj=(1, 1, None))
        
        
        # ===== Smem allocation and copy setup ======
                        
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        
        c_smem_layout = cute.make_layout(
            (self.BM, self.BN),
            stride=(self.BN, 1)
        )
        
        # Reuse sA smem as sC
        sC = storage.sA.get_tensor(c_smem_layout)

        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA, 0, 2)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB, 0, 2)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            sB_for_tma_partition,
            gB_for_tma_partition,
        )

        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        mbar_ptr = storage.mbarrier_array_ptr.data_ptr()

        #===== mma thread partitioning memory spaces =====

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
        
        tCgC = thr_mma.partition_C(gC)        
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        #===== Main loop ======
        
        accumulators = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False) # acc = 0
    
        if warp_idx == 0 and tidx == 0:
            # Set expected arrival count for mbarrier, TMA is issued by only 1 thread, so the count is 1
            cute.arch.mbarrier_init(mbar_ptr, cnt=1)
            cute.arch.mbarrier_init_fence()
        # Make sure the mbarrier initialization is visible to all threads    
        cute.arch.sync_threads()
        
        # Signal that the phase/TMA is complete
        # Readers can check my previous commit where I didn't use phase
        phase = 0
        
        for kidx in range(mA_tma_tensor.shape[1] // self.BK):
            # I think cute launches using a single thread under the hood, so here if we use only one thread to call the copy, it will cause a deadlock. Hence we call it using the first warp instead.
            if warp_idx == 0:
                cute.copy(
                    tma_atom_a,
                    tAgA[None, kidx],
                    tAsA[None, 0],
                    tma_bar_ptr=mbar_ptr
                )
                
                cute.copy(
                    tma_atom_b,
                    tBgB[None, kidx],
                    tBsB[None, 0],
                    tma_bar_ptr=mbar_ptr
                )
                
                if tidx == 0:
                    # Track how bytes have been transferred with mbarrier
                    # When both arrival count above and expected bytes reach zero, the barrier is released and move on to the next phase
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_ptr,
                        tma_transaction_bytes,
                    )                

            # cute.arch.sync_threads()
            cute.arch.mbarrier_wait(mbar_ptr, phase)
            
            # Flip the phase using XOR
            phase ^= 1
            
            cute.nvgpu.warpgroup.fence()
            
            for k_block_idx in range(self.BK // self.tiled_mma.shape_mnk[2]):
                k_block_coord = (None, None, k_block_idx, 0)
                tCrA_k = tCrA[k_block_coord]
                tCrB_k = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_k,
                    tCrB_k,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

        #===== Store results ======

        # Option 1: Use StMatrixStore to store from register to smem
        
        # tCrC_out = cute.make_fragment_like(accumulators, dtype=cutlass.Float16)
        
        # for reg_idx in range(cute.size(tCrC_out)):
        #     tCrC_out[reg_idx] = cutlass.Float16(accumulators[reg_idx])

        # copy_atom_C = cute.make_copy_atom(
        #     cute.nvgpu.warp.StMatrix8x8x16bOp(
        #         transpose=False,
        #         num_matrices=4,
        #     ),
        #     self.c_dtype,
        # )

        # tiled_copy_r2s_C = cute.make_tiled_copy_C(copy_atom_C, tiled_mma)

        # thr_copy_stmatrix_C = tiled_copy_r2s_C.get_slice(tidx)
        # tCrC_copy_view = thr_copy_stmatrix_C.retile(tCrC_out)
        # tCsC_copy_view = thr_copy_stmatrix_C.partition_D(sC)
        
        # cute.copy(
        #     atom=tiled_copy_r2s_C,
        #     src=tCrC_copy_view,
        #     dst=tCsC_copy_view
        # )

        # Option 2: use tv_layout to store from register -> smem, then smem -> gmem with async copy
        
        tv_layout_C_tiled = tiled_mma.tv_layout_C_tiled
        
        for reg_idx in range(cute.size(accumulators)):
            coord = cute.idx2crd((tidx, reg_idx), tv_layout_C_tiled.shape)
            mn_tile_flat = cute.crd2idx(coord, tv_layout_C_tiled)
            m, n = cute.idx2crd(mn_tile_flat, c_smem_layout.shape)
            sC[m, n] = cutlass.Float16(accumulators[reg_idx])
        
        # TMA store from smem to gmem for both option 1 and 2
        cute.arch.sync_threads()
        cute.arch.fence_proxy("async.shared", space="cta")
        
        gC_tma = cute.local_tile(
            input=mC_tma_tensor,
            tiler=self.tile_shape_mnk,
            coord=tile_coord_mnk,
            proj=(1, 1, None)
        )
        
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_tma, 0, 2)
        
        tCsC, tCgC_store = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        
        if warp_idx == 0:
            cute.copy(tma_atom_c, tCsC, tCgC_store)

        # Option 3: directly store from register to gmem
        
        # Can't use tCgC because it was partitioned using warp group
        # thr_mma_store = tiled_mma.get_slice(tidx)
        # tCgC_store = thr_mma_store.partition_C(gC)

        # atom_universal = cute.make_copy_atom(
        #     cute.nvgpu.CopyUniversalOp(),
        #     mC.element_type 
        # )

        # tCrC_out = cute.make_fragment_like(accumulators, dtype=cutlass.Float16)
        
        # for reg_idx in range(cute.size(tCrC_out)):
        #     tCrC_out[reg_idx] = cutlass.Float16(accumulators[reg_idx])
            
        # cute.copy(
        #     atom=atom_universal,
        #     src=tCrC_out,
        #     dst=tCgC_store  # Use the correctly partitioned tensor
        # )      

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int]
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile
        )
        return tma_atom, tma_tensor

def main():
    M, N, K = 4096, 4096, 4096

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    gemm = Gemm_TC()
    compiled = cute.compile(gemm, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), f"CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()