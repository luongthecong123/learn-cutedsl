import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.testing import benchmark, JitArguments

@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(
        T.i32(), [],
        "tcgen05.fence::after_thread_sync;\n\tmov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


class Gemm_TC:
    def __init__(
        self,
        mma_tiler_mnk: Tuple[int, int, int] = (128, 256, 64),
        mma_inst_shape_mnk: Tuple[int, int, int] = (128, 256, 16),
    ):
        self.mma_tiler_mnk = mma_tiler_mnk
        self.BM, self.BN, self.BK = mma_tiler_mnk
        self.mma_inst_shape_mnk = mma_inst_shape_mnk

        self.threads_per_cta = 128
        self.num_stages = 1

        self.io_dtype = cutlass.Float16
        self.acc_dtype = cutlass.Float32

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
    ):
        # Construct tcgen05 tiled MMA
        op = tcgen05.MmaF16BF16Op(
            self.io_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        print("tiled_mma: ", self.tiled_mma)

        # Construct SMEM layouts (sm100 style — takes tiled_mma as first arg)
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler_mnk, a.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler_mnk, b.element_type, self.num_stages,
        )
        print("a_smem_layout: ", self.a_smem_layout)
        print("b_smem_layout: ", self.b_smem_layout)

        # Single-stage view for TMA atom creation
        a_smem_layout_one_stage = cute.select(self.a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])

        # Construct TMA load atoms (MMA-aware partitioning)
        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, a, a_smem_layout_one_stage, self.mma_tiler_mnk, self.tiled_mma,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, b, b_smem_layout_one_stage, self.mma_tiler_mnk, self.tiled_mma,
        )

        # Shared storage: barriers + TMEM allocation tracking
        # (SMEM tensors for A/B are allocated separately via SmemAllocator)
        @cute.struct
        class SharedStorage:
            tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        # Launch kernel
        grid_dim = *cute.ceil_div(c.shape, (self.BM, self.BN)), 1
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=grid_dim,
            block=(self.threads_per_cta, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_tma_tensor: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_tma_tensor: cute.Tensor,
        mC: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        # ====== Thread & Block setup ======
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()
        mma_coord_mnk = (bidx, bidy, None)

        # ====== SMEM allocation ======
        # Allocated independently from SharedStorage (sm100 style)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ====== TMEM allocation ======
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        # ====== Partition tensors for MMA ======
        gA = cute.local_tile(mA_tma_tensor, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, self.mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

        # tcgen05: single-thread partitioning (not warp-group based like WGMMA)
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)   # (MMA, MMA_M, MMA_K, RestK)
        tCgB = thr_mma.partition_B(gB)   # (MMA, MMA_N, MMA_K, RestK)
        tCgC = thr_mma.partition_C(gC)   # (MMA, MMA_M, MMA_N)

        # Fragments: A/B read directly from SMEM (tcgen05 MMA reads SMEM, not registers)
        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K)
        tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K)

        # Accumulator lives in TMEM (not registers like WGMMA)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)  # (MMA, MMA_M, MMA_N)

        # ====== TMA partition (through MMA partitioning) ======
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # ====== Setup TMEM pointer ======
        # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
        # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        # Swap the pointer in tCtAcc
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # ====== Epilogue setup: TMEM → RMEM → GMEM ======
        # Sub-tile the accumulator for better instruction-level parallelism
        subtile_cnt = 4
        epi_tiler = (
            (cute.size(tCtAcc, mode=[0, 0]),
             cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
        )
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)  # (EpiTile, NumTiles)
        gC_epi = cute.zipped_divide(tCgC, epi_tiler)         # (EpiTile, NumTiles)

        # Every thread loads 64 x fp32 from TMEM
        tmem_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
            cutlass.Float32,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)  # (TmemCpy, NumTmemCpy, NumTiles)
        tDgC = tmem_thr_copy.partition_D(gC_epi)       # (TmemCpy, NumTmemCpy, NumTiles)

        tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, self.acc_dtype)
        tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, self.io_dtype)

        # ====== Barrier setup ======
        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        tma_transaction_bytes = cute.size_in_bytes(
            self.io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(
            self.io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
        )

        # ====== Main loop (only warp 0 does TMA + MMA) ======
        num_k_tiles = cute.size(gA, mode=[2])

        
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)# acc = 0
    
        if warp_idx == 0 and tidx == 0:
            # Set expected arrival count for mbarrier, TMA is issued by only 1 thread, so the count is 1
            cute.arch.mbarrier_init(tma_mbar, cnt=1)
            cute.arch.mbarrier_init(mma_mbar, cnt=1)
            cute.arch.mbarrier_init_fence()
        # Make sure the mbarrier initialization is visible to all threads    
        cute.arch.sync_threads()
        
        # Signal that the phase/TMA/mma is complete
        tma_phase = 0
        mma_phase = 0
        
        # cute launches tma and umma using a single thread under the hood, so here if we use only one thread to call the copy, it will cause a deadlock. Hence we call it using the first warp instead.
        if warp_idx == 0:   
            for kidx in range(mA_tma_tensor.shape[1] // self.BK):
                cute.copy(
                    tma_atom_a,
                    tAgA[None, kidx],
                    tAsA[None, 0],
                    tma_bar_ptr=tma_mbar
                )
                
                cute.copy(
                    tma_atom_b,
                    tBgB[None, kidx],
                    tBsB[None, 0],
                    tma_bar_ptr=tma_mbar
                )
                
                if tidx == 0:
                    # Track how bytes have been transferred with mbarrier
                    # When both arrival count above and expected bytes reach zero, the barrier is released and move on to the next phase
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        tma_mbar,
                        tma_transaction_bytes,
                    )                

                # cute.arch.sync_threads()
                cute.arch.mbarrier_wait(tma_mbar, tma_phase)
                
                # Flip the phase using XOR
                tma_phase ^= 1
                
                _ = tcgen05_fence()

                num_k_blocks = cute.size(tCrA, mode=[2])

                for k_block_idx in range(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, 0)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    
                if tidx == 0:    
                    tcgen05.commit(mma_mbar)
                    
                cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                mma_phase ^= 1


        # ====== Epilogue: TMEM → RMEM → GMEM ======

        # Release TMEM allocation lock
        tmem.relinquish_alloc_permit()

        cute.arch.sync_threads()

        # Sub-tiled TMEM → RMEM → GMEM copy (all threads participate)
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
            tCrC.store(tCrAcc.load().to(self.io_dtype))
            cute.autovec_copy(tCrC, tDgC[None, None, i])

        # Deallocate TMEM
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)


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

    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()