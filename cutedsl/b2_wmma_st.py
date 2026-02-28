import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from typing import Tuple

import torch

"""
Use HMMA tensor core (Ampere and above) to perform matrix multiplication
Each block has 16 warps (512 threads), arange to 4x4 tiles.
Each warp performs 16x16x16 GEMM, and calculate a tile of 16x16 in matC

Occupancy: RTX GPUs can launch 1536 threads/SM, each SM has 99 KB (minus 1 KB for system usage), therefore for a blocksize of 512 to reach 100% occupancy, each threadblock needs: each thread uses < 42 registers (65536/1536), each block uses < 33 KB of shared memory.

Current code:
smem/block = sA + sB + sC = 9 + 9 + 9 = 27 KB

Code flow:

    fragC = 0
    for tile_k in range(K/BK):
        1. Load tile from gmem -> smem sA, sB
        2. MMA: 
            2.1. Load sA, sB to register fragA, fragB
            2.2. MMA: fragC += fragA @ fragC

    // Store result
    option 1: Store result in fragC register -> sC -> gmem
    option 2: Store result in fragC register straight to gmem
"""


class Gemm_TC:
    def __init__(
        self,
        cta_tiler = (64, 64, 64),
        num_threads = 512
    ):
        self._cta_tiler = cta_tiler
        self._num_threads = num_threads
        self._bM, self._bN, self._bK = self._cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (4, 4, 1)
        self._smem_padding = 8
        self._num_vectorized = 4
        self.buffer_align_bytes = 1024
        assert self._bM % 16 == 0, "bM must be divisible by 16"
        assert self._bN % 16 == 0, "bN must be divisible by 16"        
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor, 
        mC: cute.Tensor,
    ):
        # print("[DEBUG GEMM TC] from host")
        
        #===============================================
        # MMA Layout
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0], # 4 * 16 = 64
            # if atom layout's N-mode is 1, to leverage the largest coalesced
            # shared memory -> register copy, set the tiled mma's N mode to 16
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2, # 4 * 8 * 2 = 64
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )     
        
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)
        #===============================================
        # SMEM layout
        padding = self._smem_padding
        self.sA_layout = cute.make_layout(
            shape=(self._bM, self._bK), 
            stride=(self._bK + padding, 1))
        self.sB_layout = cute.make_layout(
            shape=(self._bN, self._bK), 
            stride=(self._bK + padding, 1))
        self.sC_layout = cute.make_layout(
            shape=(self._bM, self._bN),
            stride=(self._bN + padding, 1)
        )
        
        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    mA.element_type, cute.cosize(self.sB_layout)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    mB.element_type, cute.cosize(self.sB_layout)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    mC.element_type, cute.cosize(self.sC_layout)
                ],
                self.buffer_align_bytes,
            ]            

        self.shared_storage = SharedStorage        
        
        #===============================================
        # COPY layout
        num_vectorized = self._num_vectorized
        atom_copy_A = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vectorized
        )
        atom_copy_B = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width * num_vectorized
        )
        # K-major
        major_mode_size = self._bK // num_vectorized
        tA = cute.make_layout(
            shape=(self._num_threads // major_mode_size, major_mode_size),
            stride=(major_mode_size, 1)
        )
        vA = cute.make_layout(shape=(1, num_vectorized), stride=(0, 1))
        tiler_mn, tvA = cute.make_layout_tv(tA, vA)

        tiled_copy_A = cute.make_tiled_copy_tv(atom_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_copy_B, tA, vA)
        
        # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, 1)
        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        self.kernel(
            mA, mB, mC,
            self.sA_layout, self.sB_layout, self.sC_layout,
            tiled_copy_A, tiled_copy_B,
            tiled_mma, permutation_mnk
            ).launch(
            grid=grid_dim, 
            block=(self._num_threads,1,1)
        )
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        permutation_mnk: Tuple
    ):  
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(
            layout = sA_layout
        )
        sB = storage.sB.get_tensor(
            layout = sB_layout
        )
        sC = storage.sC.get_tensor(
            layout = sC_layout
        )
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()    
        
        gA = cute.local_tile(
            input=mA, 
            tiler=self._cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, None, 1))
        
        gB = cute.local_tile(
            input=mB, 
            tiler=self._cta_tiler,
            coord=(bidx, bidy, None),
            proj=(None, 1, 1))
        
        gC = cute.local_tile(
            input=mC, 
            tiler=self._cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, 1, None))
        
        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)

        tAgA = thr_copyA.partition_S(gA)
        tAsA = thr_copyA.partition_D(sA)
        tBgB = thr_copyB.partition_S(gB)
        tBsB = thr_copyB.partition_D(sB)
        

        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCsC = thr_mma.partition_C(sC)
        
        # ====================== Registers allocation for  MmaF16BF16Op        
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        # Clear the accumulator
        tCrC.fill(0.0)
        
        # Creates the tiled copy so that it matches the thread-value layout
        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False,num_matrices=4),
            mA.element_type,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False,num_matrices=4),
            mB.element_type,
        )

        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tid)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tid)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)   
        
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
        
        
        for kidx in range(mA.shape[1] // self._bK):
            # Load sA, sB: gmem -> smem
            
            # Option 1: use cute.copy and tiled copy
            
            # cute.copy(
            #     atom=tiled_copy_A,
            #     src=tAgA[None, None, None, kidx],
            #     dst=tAsA[None, None, None]
            # )
            
            # cute.copy(
            #     atom=tiled_copy_B,
            #     src=tBgB[None, None, None, kidx],
            #     dst=tBsB[None, None, None]
            # )
            
            # Option 2: use autovec_copy since this is trivial copy
            
            cute.autovec_copy(
                src=tAgA[None, None, None, kidx],
                dst=tAsA[None, None, None]                
            )
            
            cute.autovec_copy(
                src=tBgB[None, None, None, kidx],
                dst=tBsB[None, None, None]                
            )
            
            cute.arch.sync_threads()
            
            # Load sA -> register A
            cute.copy(
                atom=tiled_copy_s2r_A,
                src=tCsA_copy_view,
                dst=tCrA_copy_view
            )
            
            cute.copy(
                atom=tiled_copy_s2r_B,
                src=tCsB_copy_view,
                dst=tCrB_copy_view
            )
            
            # GEMM on register fragments
            cute.gemm(
                atom=tiled_mma,
                d=tCrC,
                a=tCrA,
                b=tCrB,
                c=tCrC
            )
            
            cute.arch.sync_threads()
        
        # Option 1: fragC register -> smem -> gmem        
        # Cast FP32 -> FP16
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for reg_idx in range(cute.size(tCrC_out)):
            tCrC_out[reg_idx] = cutlass.Float16(tCrC[reg_idx])       
        
        # Copy fragC register results back to smem
        cute.autovec_copy(
            src=tCrC_out,
            dst=tCsC
        )
        
        cute.arch.sync_threads()
        
        # Copy from smem to gmem
        atom_copy_s2g_C = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type
        )
        
        tiled_copy_s2g_C = cute.make_tiled_copy_C(atom_copy_s2g_C, tiled_mma)
        
        cute.copy(
            atom=tiled_copy_s2g_C,
            src=tCsC,
            dst=tCgC
        )
        
        
        # Option 2: Copy straight to GMEM
        
        # atom_universal = cute.make_copy_atom(
        #     cute.nvgpu.CopyUniversalOp(),
        #     mC.element_type 
        # )
        
        # tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        
        # tv_layout_C_tiled = tiled_mma.tv_layout_C_tiled
        # fragC_layout = cute.make_layout(
        #     (permutation_mnk[0], permutation_mnk[1]),
        #     stride=(permutation_mnk[1], 1)
        # )
        
        # # Float32 -> Float16
        # # for reg_idx in range(cute.size(tCrC_out)):
        # #     tCrC_out[reg_idx] = cutlass.Float16(tCrC[reg_idx])
        
        # # Equivalent to above
        # tCrC_out.store((tCrC.load()).to(cutlass.Float16))   
         
        # cute.copy(
        #     atom=atom_universal,
        #     src=tCrC_out,
        #     dst=tCgC
        # )
        
        

def main():
    M, N, K = 4096, 4096, 4096
    M, N, K = 512, 1024, 512

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    gemm = Gemm_TC(cta_tiler=(64, 64, 64))
    compiled = cute.compile(gemm, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")


if __name__ == "__main__":
    main()