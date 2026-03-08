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

"""
Blackwell tcgen05 GEMM with warp-specialized pipeline.

Builds on d1 (single-stage, single-warp TMA+MMA) by introducing:
  1. Multi-stage software pipeline (PipelineTmaUmma) to overlap TMA loads with UMMA compute
  2. Warp specialization: separate warps for TMA, MMA, and epilogue
  3. Manual mbarrier for UMMA completion (tcgen05.commit → mbarrier_wait)

WARP SPECIALIZATION (6 warps, 192 threads):
  - Warp 5 (threads 160-191): TMA load producer
  - Warp 4 (threads 128-159): MMA compute (tcgen05 UMMA, single-thread issue)
  - Warps 0-3 (threads 0-127): Epilogue (TMEM → RMEM → GMEM)

ALGORITHM (warp specialization, pipeline smem staging, TMA/UMMA overlap):
  - TMA/UMMA overlapping -> sync -> Epilogue (TMEM → RMEM → GMEM)

SMEM budget (per CTA, 4 stages):
  A: 128 x 64 x 2B = 16 KB/stage
  B: 256 x 64 x 2B = 32 KB/stage
  Total: 48 KB/stage x 4 = 192 KB
"""

class Gemm_TC:
    def __init__(
        self,
        cta_tile_shape_mnk: Tuple[int, int, int] = (128, 256, 64),
    ):
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.BM, self.BN, self.BK = cta_tile_shape_mnk
        self.mma_inst_shape_mnk = (self.BM, self.BN, 16)

        # Warp specialization
        self.epi_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (*self.epi_warp_ids, self.mma_warp_id, self.tma_warp_id)
        )

        self.num_stages = 4

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

        op = tcgen05.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        print("tiled_mma: ", self.tiled_mma)

        # Construct SMEM layouts with pipeline staging (num_stages instead of 1)
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.cta_tile_shape_mnk, a.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.cta_tile_shape_mnk, b.element_type, self.num_stages,
        )
        print("a_smem_layout: ", self.a_smem_layout)
        print("b_smem_layout: ", self.b_smem_layout)

        # Single-stage view for TMA atom creation
        a_smem_layout_one_stage = cute.select(self.a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])

        # Construct TMA load atoms (MMA-aware partitioning)
        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, a, a_smem_layout_one_stage, self.cta_tile_shape_mnk, self.tiled_mma,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, b, b_smem_layout_one_stage, self.cta_tile_shape_mnk, self.tiled_mma,
        )

        # c layout enum for get_tmem_load_op
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Shared storage: pipeline barriers + TMEM allocation tracking
        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]  # manual mbarrier for UMMA completion
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

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        # ====== SMEM allocation ======
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ====== TMEM allocation ======
        tmem_barrier_id = 1
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=tmem_barrier_id,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )

        # ====== Partition tensors for MMA ======
        # (bM, bK, num_k_tiles)
        gA = cute.local_tile(mA_tma_tensor, self.cta_tile_shape_mnk, mma_coord_mnk, proj=(1, None, 1))
        # (bN, bK, num_k_tiles)
        gB = cute.local_tile(mB_tma_tensor, self.cta_tile_shape_mnk, mma_coord_mnk, proj=(None, 1, 1))
        # (bM, bN)
        gC = cute.local_tile(mC, self.cta_tile_shape_mnk, mma_coord_mnk, proj=(1, 1, None))

        # tcgen05: single-thread partitioning (not warp-group based like WGMMA)
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        # Fragments: A/B read directly from SMEM (now multi-stage)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # Accumulator lives in TMEM
        acc_shape = tiled_mma.partition_shape_C(self.cta_tile_shape_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc)
        print("num_tmem_cols: ", num_tmem_cols)
        tmem.allocate(num_tmem_cols)

        # ====== TMA partition (through MMA partitioning) ======
        # group_modes(sA, 0, 3) groups spatial dims, leaving the stage dim
        # Result: tAsA is (TMA_part, stages), tAgA is (TMA_part, num_k_tiles)
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
        # CTA-wide sync before retrieving TMEM start address
        # (only warp 0 does allocation, so all warps sync here)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # ====== Epilogue setup: TMEM → RMEM → GMEM ======
        epi_tile = self.cta_tile_shape_mnk[:2]
        print("epi_tile: ", epi_tile)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs=False,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc_epi)
        tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_gC = tmem_thr_copy.partition_D(tCgC_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, self.acc_dtype)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

        # Group trailing dims; subtile_cnt falls out of partition shape (mode 3)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)

        # ====== Pipeline setup ======
        # Trivial CTA layout for 1CTA (no cluster, no 2CTA)
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))

        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(
            self.b_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
        )

        # AB pipeline: TMA producer (warp 5) → UMMA consumer (warp 4)
        # PipelineTmaUmma synchronizes TMA loads with tcgen05 MMA consumption
        # - producer_group: single thread (TMA is issued by 1 thread)
        # - consumer_group: 1 thread (UMMA is single-thread on tcgen05)
        producer, consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=tma_transaction_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        # Manual mbarrier for UMMA async completion (replaces PipelineUmmaAsync)
        # tcgen05.commit(mma_mbar) signals when all preceding TMEM writes finish.
        mma_mbar = storage.mma_mbar_ptr.data_ptr()
        if warp_idx == 0 and tidx == 0:
            cute.arch.mbarrier_init(mma_mbar, cnt=1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # ====== Main loop ======
        num_k_tiles = mA_tma_tensor.shape[1] // self.BK

        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)  # acc = 0

        # ------ TMA Load Producer ------
        if warp_idx == self.tma_warp_id:
            for kidx in cutlass.range(num_k_tiles):
                ab_empty = producer.acquire_and_advance()

                cute.copy(
                    tma_atom_a,
                    tAgA[None, ab_empty.count],
                    tAsA[None, ab_empty.index],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[None, ab_empty.count],
                    tBsB[None, ab_empty.index],
                    tma_bar_ptr=ab_empty.barrier,
                )

        # ------ MMA Compute (UMMA Consumer) ------
        if warp_idx == self.mma_warp_id:
            for kidx in cutlass.range(num_k_tiles):
                ab_full = consumer.wait_and_advance()

                num_k_blocks = cute.size(tCrA, mode=[2])
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, ab_full.index)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                ab_full.release()

            # Signal UMMA completion: hardware arrives at mma_mbar
            # when all preceding TMEM writes finish
            if tidx == self.mma_warp_id * self.threads_per_warp:
                tcgen05.commit(mma_mbar)

        # ====== Epilogue: TMEM → RMEM → GMEM ======

        # Release TMEM allocation lock
        tmem.relinquish_alloc_permit()

        # All threads wait for UMMA to finish writing TMEM
        cute.arch.mbarrier_wait(mma_mbar, 0)

        if warp_idx <= self.epi_warp_ids[-1]:
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                cute.copy(tmem_tiled_copy, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc)
                tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
                cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

        # Producer tail
        if warp_idx == self.tma_warp_id:
            producer.tail()

        # Sync all threads before TMEM deallocation
        pipeline.sync(barrier_id=tmem_barrier_id)
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
