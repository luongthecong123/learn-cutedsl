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
        mma_tiler_mnk: Tuple[int, int, int] = (128, 256, 256), # or faster with (128, 512, 128)
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

        # cta_tile_shape_mnk for epilogue copy atom selection (no 2cta, so M is full)
        self.cta_tile_shape_mnk = self.mma_tiler_mnk

        # c layout enum for get_tmem_load_op
        self.c_layout = utils.LayoutEnum.from_tensor(c)

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

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

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

        # ====== TMEM ======
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )

        # ====== Partition tensors for MMA ======
        gA = cute.local_tile(mA_tma_tensor, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, self.mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

        # tcgen05: single-thread partitioning (not warp-group based like WGMMA)
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        # Fragments: A/B read directly from SMEM
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # Accumulator lives in TMEM
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc)
        print("num_tmem_cols: ", num_tmem_cols)
        tmem.allocate(num_tmem_cols)

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
        # Delegate copy atom selection to sm100_utils — picks Ld32x32bOp repetition
        # based on tile shape, c_layout, dtypes; subtile_cnt derived from partition shape
        epi_tile = self.cta_tile_shape_mnk[:2]
        print("epi_tile: ", epi_tile)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.io_dtype,
            self.acc_dtype,
            epi_tile,
            False,  # use_2cta_instrs
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc_epi)
        # tCgC is (MMA, MMA_M, MMA_N) — rank 3, mma_coord already applied via local_tile
        tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_gC = tmem_thr_copy.partition_D(tCgC_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, self.acc_dtype)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.io_dtype)

        # Group trailing dims; subtile_cnt falls out of partition shape (mode 3)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_dtype)

        # ====== Barrier setup ======
        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        tma_transaction_bytes = cute.size_in_bytes(
            self.io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(
            self.io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
        )

        # ====== Main loop (only warp 0 does TMA + MMA) ======
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)# acc = 0

        if warp_idx == 0 and tidx == 0:
            # Set expected arrival count for mbarrier, TMA is issued by only 1 thread, so the count is 1, same for UMMA
            cute.arch.mbarrier_init(tma_mbar, cnt=1)
            cute.arch.mbarrier_init(mma_mbar, cnt=1)
            cute.arch.mbarrier_init_fence()
        # Make sure the mbarrier initialization is visible to all threads
        cute.arch.sync_threads()

        # Signal that the phase/TMA/mma is complete
        tma_phase = 0
        mma_phase = 0

        for kidx in range(mA_tma_tensor.shape[1] // self.BK):
            # cute launches tma and umma using a single thread under the hood, so here if we use only one thread to call the op, it will cause a deadlock. Hence we call it using the first warp instead.
            if warp_idx == 0:
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

            cute.arch.mbarrier_wait(tma_mbar, tma_phase)

            # Flip the phase using XOR
            tma_phase ^= 1

            _ = tcgen05_fence()

            num_k_blocks = cute.size(tCrA, mode=[2])

            if warp_idx == 0:
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

        # subtile_cnt derived from partition shape (mode 3)
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            cute.copy(tmem_tiled_copy, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc)
            tTR_rC.store(tTR_rAcc.load().to(self.io_dtype))
            cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

        # Deallocate TMEM
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