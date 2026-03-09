import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.testing import benchmark, JitArguments

"""
Blackwell tcgen05 GEMM with 2CTA MMA and warp-specialized pipeline.

Builds on d2 (1CTA warp-specialized pipeline) by adding:
  1. 2CTA MMA instructions (CtaGroup.TWO) — doubles the M dimension to 256
  2. 2x1 cluster with TMA multicast
  3. Halved per-CTA B SMEM → allows 7 pipeline stages (vs 4 in d2)

WARP SPECIALIZATION (same as d2):
  - Warp 5: TMA load producer (both CTAs independently load their SMEM tiles)
  - Warp 4: MMA compute (leader CTA only issues cooperative UMMA instructions)
  - Warps 0-3: Epilogue (TMEM → RMEM → GMEM)

2CTA MMA:
  The (256, 256, 16) instruction is cooperative — two CTAs share the work.
  Each CTA holds 128 rows of A in SMEM and half of B.
  The leader CTA (V=0) issues the MMA instruction; hardware coordinates
  reading from both CTAs' SMEM and writing to both CTAs' TMEM.

SMEM budget (per CTA, 7 stages):
  A: 128 x 64 x 2B = 16 KB/stage   (same as d2)
  B: 128 x 64 x 2B = 16 KB/stage   (halved vs d2's 256 x 64 = 32 KB)
  Total: 32 KB/stage x 7 = 224 KB   (vs 48 KB/stage x 4 = 192 KB in d2)
"""


class Gemm_TC:
    def __init__(
        self,
        mma_tiler_mnk: Tuple[int, int, int] = (256, 256, 64),
        mma_inst_shape_mnk: Tuple[int, int, int] = (256, 256, 16),
        cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1),
    ):
        self.mma_tiler_mnk = mma_tiler_mnk
        self.BM, self.BN, self.BK = mma_tiler_mnk
        self.mma_inst_shape_mnk = mma_inst_shape_mnk
        self.cluster_shape_mnk = cluster_shape_mnk
        # Per-CTA tile shape: M is halved for 2CTA
        self.cta_tile_shape_mnk = (
            self.BM // self.cluster_shape_mnk[0],
            self.BN,
            self.BK,
        )

        # Warp specialization (same as d2)
        self.epi_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (*self.epi_warp_ids, self.mma_warp_id, self.tma_warp_id)
        )

        self.num_stages = 7

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

        # Construct tcgen05 tiled MMA (2CTA cooperative)
        op = tcgen05.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        print("tiled_mma: ", self.tiled_mma)

        # Construct SMEM layouts
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler_mnk, a.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler_mnk, b.element_type, self.num_stages,
        )
        print("a_smem_layout: ", self.a_smem_layout)
        print("b_smem_layout: ", self.b_smem_layout)

        a_smem_layout_one_stage = cute.select(self.a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])

        # VMNK layout for 2CTA cluster
        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (self.tiled_mma.thr_id,))

        # TMA atoms with multicast
        op_g2s = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, a, a_smem_layout_one_stage, self.mma_tiler_mnk, self.tiled_mma,
            self.cta_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, b, b_smem_layout_one_stage, self.mma_tiler_mnk, self.tiled_mma,
            self.cta_layout_vmnk.shape,
        )

        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Shared storage
        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]  # manual mbarrier for UMMA completion
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        # Grid: M tiles = M / (BM/2), rounded up to cluster shape
        grid_shape = cute.round_up(
            cute.ceil_div(
                (*c.shape, 1), (self.BM // 2, self.BN, self.BK)
            ),
            self.cluster_shape_mnk,
        )

        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,
            self.a_smem_layout,
            self.b_smem_layout,
            self.cta_layout_vmnk,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
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
        cta_layout_vmnk: cute.Layout,
    ):
        # ====== Thread & Block setup ======
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()

        # 2CTA cluster coordinates
        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        mma_coord_vmnk = (
            bidx % cute.size(cta_layout_vmnk, mode=[0]),   # V coord (0 or 1)
            bidx // cute.size(cta_layout_vmnk, mode=[0]),   # M tile coord
            bidy,                                            # N tile coord
            None,                                            # K (iterated)
        )
        mma_coord_mnk = mma_coord_vmnk[1:]
        is_leader_cta = mma_coord_vmnk[0] == 0

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
            is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # ====== Partition tensors for MMA ======
        gA = cute.local_tile(mA_tma_tensor, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, self.mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

        # Partition by V coord
        thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0])
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc)
        print("num_tmem_cols: ", num_tmem_cols)
        tmem.allocate(num_tmem_cols)

        # ====== TMA partition (cluster-aware) ======
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            cta_in_cluster_coord_vmnk[2],  # N coord for A multicast
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            cta_in_cluster_coord_vmnk[1],  # M coord for B multicast
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # Multicast masks for TMA
        tma_mcast_mask_a = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2,
        )
        tma_mcast_mask_b = cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1,
        )

        # ====== Setup TMEM pointer ======
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
            True,  # use_2cta_instrs
        )
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        tTR_tAcc = tmem_thr_copy.partition_S(tAcc_epi)
        tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
        tTR_gC = tmem_thr_copy.partition_D(tCgC_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, self.acc_dtype)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))

        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)

        # ====== Pipeline setup ======
        # Transaction bytes: x2 since both CTAs contribute TMA loads
        tma_transaction_bytes = (
            cute.size_in_bytes(self.a_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
            + cute.size_in_bytes(self.b_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
        ) * cute.size(cta_layout_vmnk, mode=[0])

        num_mcast_ctas_a = cute.size(cta_layout_vmnk.shape[2])
        num_mcast_ctas_b = cute.size(cta_layout_vmnk.shape[1])
        num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1

        # AB pipeline: TMA producer (warp 5, both CTAs) → UMMA consumer (warp 4, leader only)
        producer, consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_producer,
            ),
            tx_count=tma_transaction_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        # Set expected arrival count for mbarrier, TMA is issued by only 1 thread, so the count is 1
        # tcgen05.commit(mma_mbar) signals when all preceding TMEM writes finish.
        mma_mbar = storage.mma_mbar_ptr.data_ptr()
        if warp_idx == 0 and tidx == 0:
            cute.arch.mbarrier_init(mma_mbar, cnt=1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # ====== Main loop ======
        num_k_tiles = cute.size(gA, mode=[2])

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
                    mcast_mask=tma_mcast_mask_a,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[None, ab_empty.count],
                    tBsB[None, ab_empty.index],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=tma_mcast_mask_b,
                )

        # ------ MMA Compute (leader CTA only) ------
        if warp_idx == self.mma_warp_id:
            if is_leader_cta:
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

                # Signal UMMA completion: cta_mask trick from CUTLASS
                if tidx == self.mma_warp_id * self.threads_per_warp:
                    cta_mask = (1 << cute.size(cta_layout_vmnk, mode=[0])) - 1
                    tcgen05.commit(mma_mbar, mask=cta_mask, cta_group=tcgen05.CtaGroup.TWO)

        # ====== Epilogue: TMEM → RMEM → GMEM ======

        tmem.relinquish_alloc_permit()

        cute.arch.mbarrier_wait(mma_mbar, 0)

        if warp_idx <= self.epi_warp_ids[-1]:
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            print("subtile_cnt: ", subtile_cnt)
            for subtile_idx in cutlass.range(subtile_cnt):
                cute.copy(tmem_tiled_copy, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc)
                tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
                cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

        # Producer tail
        # Both CTAs must complete their TMA pipeline to avoid invalid dsmem access.
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
