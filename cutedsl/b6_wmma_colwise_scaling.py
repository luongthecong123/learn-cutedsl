import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from typing import Tuple

class Gemm_TC:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (64, 64, 64),
        num_threads: int = 512
    ):
        self._cta_tiler = cta_tiler
        self._num_threads = num_threads
        self._bM, self._bN, self._bK = self._cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (4, 4, 1)
        self._smem_padding = 8
        self._num_vectorized = 4
        assert self._bM % 16 == 0, "bM must be divisible by 16"
        assert self._bN % 16 == 0, "bN must be divisible by 16"        
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor, 
        mC: cute.Tensor,
        mR: cute.Tensor,
    ):
        # ====== MMA Layout ======
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        
        print(f"[DEBUG GEMM TC] mma_op: {mma_op}")

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
        
        print(f"[DEBUG GEMM TC] tiled_mma: {tiled_mma}")
        
        # ====== SMEM layout ======
        padding = self._smem_padding
        sA_layout = cute.make_layout(shape=(self._bM, self._bK), stride=(self._bK + padding, 1))
        sB_layout = cute.make_layout(shape=(self._bN, self._bK), stride=(self._bK + padding, 1))
        sR_layout = cute.make_layout(shape=(1, self._bN), stride=(0, 1))
        
        # ====== COPY layout ======
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
        atom_copy_R = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mR.element_type,
            num_bits_per_copy=mR.element_type.width * num_vectorized
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
        tiled_copy_R = cute.make_tiled_copy_tv(atom_copy_R, tA, vA)
        
        # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, 1)
        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        self.kernel(
            mA, mB, mC, mR,
            sA_layout, sB_layout, sR_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_R,
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
        mR: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        sR_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_R: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        permutation_mnk: Tuple
    ):
        # ====== Thread, Block setup =======
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        # ===== Smem allocation and copy setup ======
        allocator = cutlass.utils.SmemAllocator()
        sA = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
        sB = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)
        sR = allocator.allocate_tensor(cutlass.Float16, sR_layout, 16, None)
        
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
        
        gR = cute.local_tile(
            input=mR,
            tiler=self._cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, 1, None)            
        )

        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)
        thr_copyR = tiled_copy_R.get_slice(tid)

        tAgA = thr_copyA.partition_S(gA)
        tAsA = thr_copyA.partition_D(sA)
        tBgB = thr_copyB.partition_S(gB)
        tBsB = thr_copyB.partition_D(sB)
        tRgR = thr_copyR.partition_S(gR)
        tRsR = thr_copyR.partition_D(sR)
        
        # Load column-wise scaling factor
        if tid < self._bN:
            sR[tid] = mR[bidy * self._bN + tid]
        
        # ===== mma thread partitioning memory spaces =====
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        
        # ====== Shared memory to register copy ======
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
        
        # ====== Main loop ======
        tCrC.fill(0.0)
        
        for kidx in range(mA.shape[1] // self._bK):
            # Load sA, sB
            cute.copy(
                atom=tiled_copy_A,
                src=tAgA[None, None, None, kidx],
                dst=tAsA[None, None, None]
            )
            
            cute.copy(
                atom=tiled_copy_B,
                src=tBgB[None, None, None, kidx],
                dst=tBsB[None, None, None]
            )
            
            cute.arch.sync_threads()
            
            # mma:
            # Can also remove mmak for loop.
            for mmak in range(self._bK // self.mma_inst_shape[2]):
                # Load smem to register
                cute.copy(
                    atom=tiled_copy_s2r_A,
                    src=tCsA_copy_view[None, None, mmak],
                    dst=tCrA_copy_view[None, None, mmak]
                )
                
                cute.copy(
                    atom=tiled_copy_s2r_B,
                    src=tCsB_copy_view[None, None, mmak],
                    dst=tCrB_copy_view[None, None, mmak]
                )
            
            # GEMM
            cute.gemm(
                atom=tiled_mma,
                d=tCrC,
                a=tCrA,
                b=tCrB,
                c=tCrC
            )
            
            cute.arch.sync_threads()
        
        # ====== Store results ======
        atom_universal = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type 
        )
        
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        
        tv_layout_C_tiled = tiled_mma.tv_layout_C_tiled

        print(f"[DEBUG] tv_layout_C_tiled: {tv_layout_C_tiled}")
        
        for reg_idx in range(cute.size(tCrC_out)):
            coord = cute.idx2crd((tid, reg_idx), tv_layout_C_tiled.shape)
            mn_flat = cute.crd2idx(coord, tv_layout_C_tiled)
            m, n = cute.idx2crd(mn_flat, (permutation_mnk[0], permutation_mnk[1]))
            tCrC_out[reg_idx] = cutlass.Float16(tCrC[reg_idx] * sR[n])
            
        cute.copy(
            atom=atom_universal,
            src=tCrC_out,
            dst=tCgC
        )

def main():
    torch.manual_seed(42)

    M, N, K = 4096, 4096, 4096
    
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    r = torch.randn(1, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    r_ = from_dlpack(r, assumed_align=16)

    gemm = Gemm_TC()
    compiled = cute.compile(gemm, a_, b_, c_, r_)
    compiled(a_, b_, c_, r_)

    c_test_colwise = (a @ b.T) * r  # column-wise scale by r (shape 1 x N)
    
    assert torch.allclose(c, c_test_colwise, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(a_, b_, c_, r_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")


if __name__ == "__main__":
    main()