import torch
from typing import Tuple
import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
from cutlass.cute.testing import benchmark, JitArguments

class Gemm_TC:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (64, 128, 64),
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.BK = self.tile_shape_mnk
        
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.BM > 64 and self.BN > 64
            else (1, 1, 1)
        )
        
        self.warp_size = cute.arch.WARP_SIZE
        self.num_threads_per_warp_group = 128
        
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        
        assert self.BM % 64 == 0, "bM must be divisible by 64 for WGMMA"
        assert self.BN % 64 == 0, "bN must be divisible by 64 for WGMMA"

        self.num_stages = 1
        
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)
        
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
        
        cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))        

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
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        self.c_smem_layout = cute.make_layout(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            stride=(self.tile_shape_mnk[1], 1)
        )

        self.c_smem_layout_swizzled = cute.make_composed_layout(
            inner=cute.make_swizzle(3, 4, 3),
            offset=0,
            outer=self.c_smem_layout
        )
        
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c,
            self.c_smem_layout_swizzled,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
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
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, self.tile_shape_mnk[0] * self.tile_shape_mnk[1]
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
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
            cta_layout_mnk,
            c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_swizzled,
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
        cta_layout_mnk: cute.Layout,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_swizzled: cute.ComposedLayout,
    ):

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)  

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

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
        
        sC = storage.sC.get_tensor(
            c_smem_layout_swizzled.outer, swizzle=c_smem_layout_swizzled.inner
        )

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
        
        tCgC = thr_mma.partition_C(gC)

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA, 0, 2)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB, 0, 2)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma_partition,
            gB_for_tma_partition,
        )
        
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        acc_shape = tCgC.shape
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        num_k_blocks = cute.size(tCrA, mode=[2])

        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        k_tile_cnt = mA_tma_tensor.shape[1] // self.BK
        
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        for kidx in range(k_tile_cnt):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, cnt=1)
                cute.arch.mbarrier_init_fence()
            
            cute.arch.sync_threads()
            
            if warp_idx == 0:
                if tidx == 0:
                    cute.arch.mbarrier_expect_tx(mbar_ptr, tma_transaction_bytes)
                    cute.arch.mbarrier_arrive(mbar_ptr)

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

            cute.arch.mbarrier_wait(mbar_ptr, 0)
            
            cute.nvgpu.warpgroup.fence()
            
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
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
        
        tv_layout_C_tiled = tiled_mma.tv_layout_C_tiled
        
        for reg_idx in range(cute.size(accumulators)):
            coord = cute.idx2crd((tidx, reg_idx), tv_layout_C_tiled.shape)
            mn_local_tile_flat = cute.crd2idx(coord, tv_layout_C_tiled)
            m_local, n_local = cute.idx2crd(mn_local_tile_flat, c_smem_layout_swizzled.outer.shape)
            sC[m_local, n_local] = cutlass.Float16(accumulators[reg_idx])
        
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

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

def main():
    M, N, K = 4096*2, 4096*2, 4096*2

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    gemm = Gemm_TC()
    compiled = cute.compile(gemm, A_, B_, C_)
    compiled(A_, B_, C_)

    # assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), f"CORRECTNESS FAILED"
    # print("CORRECTNESS PASS")
    # time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    # print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()