"""draftv4_hist_pdl.py — draftv4_hist + PDL (programmatic dependent launch).

Same histogram topk algorithm as draftv4_hist.py, but launches the score
kernel and topk kernel with use_pdl=True and inserts:
  * cute.arch.griddepcontrol_launch_dependents()  in the score kernel
    (after SMEM alloc), so the topk kernel can begin allocating SMs /
    loading state while the score kernel is still running.
  * cute.arch.griddepcontrol_wait()  at the top of the topk kernel,
    before the first score_output read (setup / phase 1).

Goal: hide topk kernel launch overhead (~1–3 µs) and the topk prologue
by overlapping them with the tail of the score kernel.

Original draftv4 notes:

Changes vs draftv3.py:
  1. THREADS_PER_CTA: 128 → 512 (4× more cooperative cp.async threads).
  2. cp.async A : i32-view tiled_copy_tv over all 512 threads (replaces
     128-thread autovec_copy). 4× fewer instructions, fully coalesced 16B
     transactions per warp.
  3. cp.async B : same treatment for q (B operand) — was previously also
     autovec_copy with 128 threads.
  4. sA / sB : SW128 swizzle in i32 view used as the cp.async destination
     layout (matches flat-T), preserves the fp8 swizzle for tcgen05 MMA.
  5. Compute path : MMA still warp-0 only; epilogue still BM=128 first
     compute threads. Other 384 threads are spectators outside cp.async.
  6. TopK : SMEM-cached radix bits (LIMIT_TOPK_SEQ_LEN=16384). One float→radix
     pass at setup; phase 1 reads precomputed bits from smem.

Score formula and data layout: identical to draftv3.
"""

import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.cutlass_dsl import dsl_user_op, T

TOP_K          = 2048
LIMIT_REQUEST  = 32
LIMIT_SEQ_LEN  = 320000
DIM_SPLIT      = 128
PAGE_SIZE      = 64
NUM_HEADS      = 64
HEAD_DIM       = 128

ROW_STRIDE     = HEAD_DIM + 4
PAGES_PER_TILE = DIM_SPLIT // PAGE_SIZE
BM             = DIM_SPLIT
BN             = NUM_HEADS
PAGE_BYTES     = PAGE_SIZE * ROW_STRIDE
FP8_REGION     = PAGE_SIZE * HEAD_DIM

# i32 view constants for cp.async loads (matches flat-T)
HEAD_DIM_I32   = HEAD_DIM // 4          # 32
PAGE_BYTES_I32 = PAGE_BYTES // 4        # 2112
BK_I32         = HEAD_DIM_I32           # 32
Q_T_STRIDE_I32 = BN * BK_I32            # 2048

MMA_INST_MNK    = (128, BN, 32)
CTA_TILE_MNK    = (BM, BN, HEAD_DIM)
THREADS_PER_CTA = 512                   # ← was 128 in draftv3
COMPUTE_THREADS = 128                   # MMA accumulator rows / epilogue threads

# A-load tiled_copy_tv params
N_PER_THREAD_A_I32 = (BM * HEAD_DIM_I32) // THREADS_PER_CTA  # 8
A_THR_ROWS         = THREADS_PER_CTA // HEAD_DIM_I32         # 16
# B-load tiled_copy_tv params
N_PER_THREAD_B_I32 = (BN * BK_I32) // THREADS_PER_CTA        # 4
B_THR_ROWS         = THREADS_PER_CTA // BK_I32               # 16

NUM_VEC = 4
K_ITERS = HEAD_DIM // NUM_VEC

# Histogram radix-select params.
HIST_BINS  = 256
NUM_PASSES = 4

# When True, cache LIMIT_TOPK_SEQ_LEN int32 radix bits in SMEM at setup time
# and always read from SMEM in phase-1 / phase-2.  Removes the dynamic
# `use_smem = sl <= smem_cap` branch that pessimised codegen (each smem-vs-gmem
# load became a runtime predicate).  Assumes max sl <= LIMIT_TOPK_SEQ_LEN.
USE_LIMIT_TOPK_SEQ_LEN: cutlass.Constexpr = True
# TopK SMEM-cached radix bits cap (set ≥ max seq_len of targeted workloads).
LIMIT_TOPK_SEQ_LEN = 16384

# ── Helpers ───────────────────────────────────────────────────────────────────
@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def float_to_radix(v: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(
        T.i32(), [v.ir_value()],
        "{"
        ".reg .u32 x; .reg .u32 mask; .reg .pred pneg; .reg .pred pnan;"
        "mov.b32 x, $1;"
        "setp.lt.f32 pneg, $1, 0f00000000;"
        "setp.neu.f32 pnan, $1, $1;"
        "selp.u32 mask, 0xFFFFFFFF, 0x80000000, pneg;"
        "xor.b32 x, x, mask;"
        "selp.u32 $0, 0xFFFFFFFF, x, pnan;"
        "}",
        "=r,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return cutlass.Uint32(r)


@cute.jit
def warp_sum_i32(val: cutlass.Int32) -> cutlass.Int32:
    for i in cutlass.range_constexpr(5):
        peer = cute.arch.shuffle_sync_bfly(val, 1 << i)
        val = val + peer
    return val


@cute.jit
def count_element(bits, desired, desired_mask, digit_pos_u, c0, c1, c2, c3):
    if (bits & desired_mask) == (desired & desired_mask):
        digit = (bits >> digit_pos_u) & cutlass.Uint32(3)
        if digit == cutlass.Uint32(0):
            c0 = c0 + cutlass.Int32(1)
        if digit == cutlass.Uint32(1):
            c1 = c1 + cutlass.Int32(1)
        if digit == cutlass.Uint32(2):
            c2 = c2 + cutlass.Int32(1)
        if digit == cutlass.Uint32(3):
            c3 = c3 + cutlass.Int32(1)
    return c0, c1, c2, c3


# ── Main class ────────────────────────────────────────────────────────────────

class Indexer_kvsplit_v4_hist_pdl:
    def __init__(self):
        self.top_k         = TOP_K
        self.dim_split     = DIM_SPLIT
        self.page_size     = PAGE_SIZE

        self.indexer_threads      = THREADS_PER_CTA
        self.pass_through_threads = 1024
        self.topk_threads         = 1024

        self.wsize = cute.arch.WARP_SIZE

        self.limit_request = LIMIT_REQUEST
        self.limit_seq_len = LIMIT_SEQ_LEN

        self.num_stages  = 1
        self.tmem_ld_rep = BN

        self.ws_score_output = torch.empty(
            LIMIT_REQUEST, LIMIT_SEQ_LEN, dtype=torch.float32, device="cuda"
        )

    @cute.jit
    def __call__(
        self,
        q_index_fp8, k_index_cache_fp8, weights,
        seq_lens, block_table, score_output, top_k_indices, stream,
    ):
        self.ab_dtype  = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32

        op = tcgen05.MmaFP8Op(
            self.ab_dtype, self.acc_dtype, MMA_INST_MNK,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)

        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, self.ab_dtype, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, self.ab_dtype, self.num_stages,
        )

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
            weights_smem:     cute.struct.MemRange[cutlass.Float32, BN]
        self.shared_storage = SharedStorage

        T_, max_num_pages = block_table.shape
        pages_per_split   = self.dim_split // self.page_size
        num_splits        = (max_num_pages + pages_per_split - 1) // pages_per_split

        if max_num_pages <= 32:
            self.pass_through_kernel(seq_lens, block_table, top_k_indices).launch(
                grid=[T_, 1, 1], block=[1024, 1, 1], stream=stream
            )
        else:
            self.indexer_ksplit_kernel(
                self.tiled_mma, self.a_smem_layout, self.b_smem_layout,
                q_index_fp8, k_index_cache_fp8, weights,
                seq_lens, block_table, num_splits, score_output, top_k_indices,
            ).launch(
                grid=[T_ + num_splits, 1, 1],
                block=[self.indexer_threads, 1, 1], stream=stream,
                use_pdl=True,
            )
            self.topk_kernel(
                seq_lens, block_table, num_splits, score_output, top_k_indices
            ).launch(
                grid=[T_, 1, 1], block=[self.topk_threads, 1, 1], stream=stream,
                use_pdl=True,
            )

    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(
            dtype, cute.make_layout(shape, stride=stride), align, None
        )

    @cute.kernel
    def pass_through_kernel(self, seq_lens, block_table, topk_indices):
        top_k_len: cutlass.Constexpr = self.top_k
        T_, max_num_pages = block_table.shape
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

        max_seq_len = seq_lens[bidx]

        alloc      = cutlass.utils.SmemAllocator()
        smem_sparse = self._smem(alloc, cutlass.Int32, (top_k_len,),       (1,), 4)
        smem_page   = self._smem(alloc, cutlass.Int32, (top_k_len // 64,), (1,), 4)

        for i in range(tidx, top_k_len, self.pass_through_threads):
            smem_sparse[i] = -1
        for j in range(tidx, top_k_len // 64, self.pass_through_threads):
            smem_page[j] = block_table[bidx, j]
        cute.arch.sync_threads()

        if warp_idx < max_num_pages:
            page_idx   = smem_page[warp_idx]
            page_start = warp_idx * cutlass.Int32(PAGE_SIZE)
            page_end   = page_start + cutlass.Int32(PAGE_SIZE)
            if page_end > max_seq_len:
                page_end = max_seq_len
            for i in range(lane_idx, page_end - page_start, self.wsize):
                token_idx = page_start + i
                if token_idx < max_seq_len:
                    smem_sparse[token_idx] = page_idx * cutlass.Int32(PAGE_SIZE) + i
        cute.arch.sync_threads()

        for i in range(tidx, top_k_len, self.pass_through_threads):
            topk_indices[bidx, i] = smem_sparse[i]

    @cute.kernel
    def indexer_ksplit_kernel(
        self, tiled_mma, a_smem_layout, b_smem_layout,
        q_index_fp8, k_index_cache_fp8, weights,
        seq_lens, block_table, num_splits, score_output, topk_indices,
    ):
        cute.arch.griddepcontrol_launch_dependents()
        
        top_k_len:      cutlass.Constexpr = self.top_k
        limit_request:  cutlass.Constexpr = self.limit_request
        tmem_ld_rep:    cutlass.Constexpr = self.tmem_ld_rep
        ab_dtype:       cutlass.Constexpr = self.ab_dtype
        acc_dtype:      cutlass.Constexpr = self.acc_dtype

        T_, max_num_pages = block_table.shape

        tidx, _, _  = cute.arch.thread_idx()
        bidx, _, _  = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        # ── SMEM allocation ─────────────────────────────────────────────
        # Order matches flat-T: swizzled buffers first for alignment.
        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(
            element_type=ab_dtype, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        sB = alloc.allocate_tensor(
            element_type=ab_dtype, layout=b_smem_layout.outer,
            byte_alignment=1024, swizzle=b_smem_layout.inner,
        )
        storage = alloc.allocate(self.shared_storage)
        smem_indexer_T_idx = self._smem(alloc, cutlass.Int32, (limit_request,),    (1,), 4)
        smem_num_idxer     = self._smem(alloc, cutlass.Int32, (1,),                (1,), 4)
        smem_sparse        = self._smem(alloc, cutlass.Int32, (top_k_len,),        (1,), 4)
        smem_page          = self._smem(alloc, cutlass.Int32, (top_k_len // 64,),  (1,), 4)

        mma_mbar = storage.mma_mbar_ptr.data_ptr()
        sWeights_ptr = cute.make_ptr(
            cutlass.Float32, storage.weights_smem.data_ptr().toint(),
            mem_space=cute.AddressSpace.smem, assumed_align=4,
        )
        sWeights = cute.make_tensor(sWeights_ptr, cute.make_layout((BN,), stride=(1,)))

        # i32 views of sA / sB for cp.async destination
        sA_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BM, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1)),
        )
        sA_i32_ptr = cute.recast_ptr(sA.iterator, dtype=cutlass.Int32)
        sA_load    = cute.make_tensor(sA_i32_ptr, sA_load_layout)

        sB_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BN, BK_I32), stride=(BK_I32, 1)),
        )
        sB_i32_ptr = cute.recast_ptr(sB.iterator, dtype=cutlass.Int32)
        sB_load    = cute.make_tensor(sB_i32_ptr, sB_load_layout)

        # cp.async tiled copies (static — built once)
        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
        )
        thr_layout_a = cute.make_layout((A_THR_ROWS, HEAD_DIM_I32),
                                        stride=(HEAD_DIM_I32, 1))
        val_layout_a = cute.make_layout((N_PER_THREAD_A_I32, 1), stride=(1, 1))
        tiled_copy_a = cute.make_tiled_copy_tv(atom_cpa, thr_layout_a, val_layout_a)
        thr_copy_a   = tiled_copy_a.get_slice(tidx)

        thr_layout_b = cute.make_layout((B_THR_ROWS, BK_I32),
                                        stride=(BK_I32, 1))
        val_layout_b = cute.make_layout((N_PER_THREAD_B_I32, 1), stride=(1, 1))
        tiled_copy_b = cute.make_tiled_copy_tv(atom_cpa, thr_layout_b, val_layout_b)
        thr_copy_b   = tiled_copy_b.get_slice(tidx)

        tBsB = thr_copy_b.partition_D(sB_load)
        tAsA = thr_copy_a.partition_D(sA_load)

        q_i32_base_full = cute.recast_ptr(q_index_fp8.iterator, dtype=cutlass.Int32)
        k_i32_base_full = cute.recast_ptr(k_index_cache_fp8.iterator, dtype=cutlass.Int32)
        # ── PDL: all independent setup (SMEM alloc, copy atoms, base ptrs)
        #         is done.  Signal now so dependent topk CTAs can schedule on
        #         SMs and run their independent prologue (SMEM alloc + τ-init)
        #         in parallel with this kernel's main MMA work.
        
        # ── Pass-through path ──────────────────────────────────────────
        if bidx >= num_splits:
            bidx_pass   = bidx - num_splits
            max_seq_len = seq_lens[bidx_pass]

            if max_seq_len <= cutlass.Int32(2048):
                for i in range(tidx, top_k_len, self.indexer_threads):
                    smem_sparse[i] = -1
                for j in range(tidx, top_k_len // 64, self.indexer_threads):
                    smem_page[j] = block_table[bidx_pass, j]
                cute.arch.sync_threads()

                for token_idx in range(tidx, max_seq_len, self.indexer_threads):
                    page_local  = token_idx // cutlass.Int32(PAGE_SIZE)
                    tok_off     = token_idx - page_local * cutlass.Int32(PAGE_SIZE)
                    global_page = smem_page[page_local]
                    smem_sparse[token_idx] = global_page * cutlass.Int32(PAGE_SIZE) + tok_off
                cute.arch.sync_threads()

                for i in range(tidx, top_k_len, self.indexer_threads):
                    topk_indices[bidx_pass, i] = smem_sparse[i]

        # ── Indexer (tcgen05 MMA scoring) ─────────────────────────────
        else:
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
            tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
            num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
            tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

            if warp_idx == 0:
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)

            tmem_barrier_id = 1
            cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                acc_dtype, alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)

            M_acc          = cute.size(tCtAcc, mode=[0, 0])
            ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(tmem_ld_rep))
            epi_tiler      = ((M_acc, tmem_ld_rep),)
            tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)
            copy_atom_t2r  = cute.make_copy_atom(ld_op, acc_dtype)
            tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
            tmem_thr_copy  = tmem_tiled_copy.get_slice(tidx)
            tTR_tAcc       = tmem_thr_copy.partition_S(tCtAcc_epi)
            tTR_rAcc       = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, acc_dtype)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # ── Compaction: warp 0 picks T_idx whose seq_len > 2048 ──
            if warp_idx == 0:
                base = cutlass.Int32(0)
                for chunk_start in cutlass.range_constexpr(0, limit_request, 32):
                    i      = cutlass.Int32(chunk_start) + lane_idx
                    is_idx = cutlass.Int32(0)
                    if i < T_:
                        if seq_lens[i] > cutlass.Int32(2048):
                            is_idx = cutlass.Int32(1)
                    scan = is_idx
                    for s in cutlass.range_constexpr(5):
                        peer = cute.arch.shuffle_sync_up(scan, 1 << s, mask_and_clamp=0)
                        if lane_idx >= cutlass.Int32(1 << s):
                            scan = scan + peer
                    excl = scan - is_idx
                    if is_idx != cutlass.Int32(0):
                        smem_indexer_T_idx[base + excl] = i
                    base = base + cute.arch.shuffle_sync(scan, 31)
                if lane_idx == cutlass.Int32(0):
                    smem_num_idxer[0] = base
            cute.arch.sync_threads()
            num_idxer_requests = smem_num_idxer[0]

            # ── mbar init once ─────────────────────────────────────────
            if warp_idx == 0:
                if tidx == 0:
                    cute.arch.mbarrier_init(mma_mbar, cnt=1)
                    cute.arch.mbarrier_init_fence()
            cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

            mma_phase = cutlass.Int32(0)

            # ── Persistent loop over indexer requests ─────────────────
            for indexer_request in range(num_idxer_requests):
                T_idx       = smem_indexer_T_idx[indexer_request]
                req_seq_len = seq_lens[T_idx]
                request_num_tiles = (
                    (req_seq_len + cutlass.Int32(BM - 1)) // cutlass.Int32(BM)
                )

                if bidx < request_num_tiles:
                    page0_id = cutlass.Int32(block_table[T_idx, bidx * PAGES_PER_TILE + 0])
                    page1_id = cutlass.Int32(block_table[T_idx, bidx * PAGES_PER_TILE + 1])

                    if tidx < cutlass.Int32(BN):
                        sWeights[tidx] = weights[T_idx, tidx]

                    # ── cp.async B (q for T_idx) — 512-thread i32 view ──
                    q_off_i32 = T_idx * cutlass.Int32(Q_T_STRIDE_I32)
                    qB_base = cute.make_ptr(
                        cutlass.Int32,
                        (q_i32_base_full + q_off_i32).toint(),
                        mem_space=cute.AddressSpace.gmem, assumed_align=16,
                    )
                    gB_i32 = cute.make_tensor(
                        qB_base,
                        cute.make_layout((BN, BK_I32), stride=(BK_I32, 1)),
                    )
                    tBgB = thr_copy_b.partition_S(gB_i32)
                    cute.copy(atom_cpa, tBgB, tBsB)

                    # ── cp.async A (paged K, 2 pages) — 512-thread i32 view ──
                    page0_off_i32 = page0_id * cutlass.Int32(PAGE_BYTES_I32)
                    jump_i32      = (page1_id - page0_id) * cutlass.Int32(PAGE_BYTES_I32)
                    i32_base = cute.make_ptr(
                        cutlass.Int32,
                        (k_i32_base_full + page0_off_i32).toint(),
                        mem_space=cute.AddressSpace.gmem, assumed_align=4,
                    )
                    gA_i32 = cute.make_tensor(
                        i32_base,
                        cute.make_layout(
                            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM_I32),
                            stride=((HEAD_DIM_I32, jump_i32), 1),
                        ),
                    )
                    tAgA = thr_copy_a.partition_S(gA_i32)
                    cute.copy(atom_cpa, tAgA, tAsA)

                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    cute.arch.fence_view_async_shared()

                    # ── tcgen05 FP8 MMA (warp 0 only) ────────────────
                    tcgen05_fence()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    if warp_idx == 0:
                        for k_block_idx in range(num_k_blocks):
                            k_block_coord = (None, None, k_block_idx, 0)
                            cute.gemm(
                                tiled_mma, tCtAcc,
                                tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        if tidx == 0:
                            tcgen05.commit(mma_mbar)

                    cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                    mma_phase = mma_phase ^ cutlass.Int32(1)

                    # ── Epilogue: only first BM=128 threads ──────────
                    if tidx < cutlass.Int32(BM):
                        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

                        page_sel      = tidx // cutlass.Int32(PAGE_SIZE)
                        token_in_page = tidx - page_sel * cutlass.Int32(PAGE_SIZE)
                        page_id_t     = cutlass.Int32(
                            block_table[T_idx, bidx * PAGES_PER_TILE + page_sel]
                        )
                        scale_f32_off = (
                            page_id_t * cutlass.Int32(PAGE_BYTES // 4)
                            + cutlass.Int32(FP8_REGION // 4)
                            + token_in_page
                        )
                        fp32_base = cute.recast_ptr(
                            k_index_cache_fp8.iterator, dtype=cutlass.Float32
                        )
                        scale_ptr = cute.make_ptr(
                            cutlass.Float32,
                            (fp32_base + scale_f32_off).toint(),
                            mem_space=cute.AddressSpace.gmem, assumed_align=1,
                        )
                        scale = cute.make_tensor(
                            scale_ptr, cute.make_layout((1,), stride=(1,))
                        )[0]

                        m_out = bidx * cutlass.Int32(BM) + tidx
                        out_val = cutlass.Float32(0.0)
                        for n_idx in cutlass.range_constexpr(BN):
                            val = tTR_rAcc[n_idx] * scale
                            out_val = (
                                out_val
                                + max(val, cutlass.Float32(0.0)) * sWeights[n_idx]
                            )
                        score_output[T_idx, m_out] = out_val

                    cute.arch.sync_threads()

            if warp_idx == 0:
                cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.barrier(barrier_id=tmem_barrier_id)
            if warp_idx == 0:
                cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

    @cute.kernel
    def topk_kernel(
        self, seq_lens, block_table, num_splits, score_output, topk_indices,
    ):
        cute.arch.griddepcontrol_wait()
        
        top_k_len:    cutlass.Constexpr = self.top_k
        topk_threads: cutlass.Constexpr = self.topk_threads
        num_warps:    cutlass.Constexpr = self.topk_threads // 32
        smem_cap:     cutlass.Constexpr = LIMIT_TOPK_SEQ_LEN

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

        # ── SMEM allocation hoisted out of dynamic `seq_lens[bidx] > 2048`
        #    branch so the allocator runs unconditionally (no runtime branch
        #    on SMEM bookkeeping).  Only the `smem_bits` cache is gated by
        #    the compile-time USE_LIMIT_TOPK_SEQ_LEN switch.
        allocator       = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(USE_LIMIT_TOPK_SEQ_LEN):
            smem_bits   = self._smem(allocator, cutlass.Int32, (smem_cap,),                (1,), 4)
        smem_warp_hist   = self._smem(allocator, cutlass.Int32, (num_warps * HIST_BINS,), (1,), 4)
        smem_hist        = self._smem(allocator, cutlass.Int32, (HIST_BINS,),              (1,), 4)
        smem_tau         = self._smem(allocator, cutlass.Int32, (5,),                      (1,), 4)
        smem_warp_above  = self._smem(allocator, cutlass.Int32, (num_warps,),              (1,), 4)
        smem_warp_tie    = self._smem(allocator, cutlass.Int32, (num_warps,),              (1,), 4)
        smem_above_round = self._smem(allocator, cutlass.Int32, (1,),                      (1,), 4)
        smem_tie_round   = self._smem(allocator, cutlass.Int32, (1,),                      (1,), 4)

        warp_hist_ptr = smem_warp_hist.iterator
        warp_base     = warp_idx * cutlass.Int32(HIST_BINS)

        if seq_lens[bidx] > cutlass.Int32(2048):
            sl      = seq_lens[bidx]
            max_col = score_output.shape[1]

            if tidx == cutlass.Int32(0):
                smem_tau[0] = cutlass.Int32(0)          # desired
                smem_tau[1] = cutlass.Int32(0)          # desired_mask
                smem_tau[2] = cutlass.Int32(0)          # above_total
                smem_tau[3] = cutlass.Int32(top_k_len)  # k_to_find
                smem_tau[4] = cutlass.Int32(0)          # early_exit

            # ── PDL: consumer prologue done (SMEM alloc + τ-state init).
            #         Wait for producer to finish writing score_output before
            #         the setup/phase-1 loops that read it.  sync_threads
            #         after wait ensures all threads see consistent memory.
            

            # ── Setup: cache float→radix bits in SMEM (when sl fits) ──
            if cutlass.const_expr(USE_LIMIT_TOPK_SEQ_LEN):
                setup_base = tidx * cutlass.Int32(NUM_VEC)
                while setup_base + cutlass.Int32(NUM_VEC - 1) < sl:
                    b0 = float_to_radix(score_output[bidx, setup_base])
                    b1 = float_to_radix(score_output[bidx, setup_base + cutlass.Int32(1)])
                    b2 = float_to_radix(score_output[bidx, setup_base + cutlass.Int32(2)])
                    b3 = float_to_radix(score_output[bidx, setup_base + cutlass.Int32(3)])
                    smem_bits[setup_base + cutlass.Int32(0)] = cutlass.Int32(b0)
                    smem_bits[setup_base + cutlass.Int32(1)] = cutlass.Int32(b1)
                    smem_bits[setup_base + cutlass.Int32(2)] = cutlass.Int32(b2)
                    smem_bits[setup_base + cutlass.Int32(3)] = cutlass.Int32(b3)
                    setup_base = setup_base + cutlass.Int32(topk_threads * NUM_VEC)
                while setup_base < sl:
                    b = float_to_radix(score_output[bidx, setup_base])
                    smem_bits[setup_base] = cutlass.Int32(b)
                    setup_base = setup_base + cutlass.Int32(1)
                cute.arch.sync_threads()

            # ── Phase 1: 4-pass 8-bit histogram radix select ─────────
            for pass_c in cutlass.range_constexpr(NUM_PASSES):
                digit_pos   = 24 - pass_c * 8
                digit_pos_u = cutlass.Uint32(digit_pos)

                early = smem_tau[4]
                if early == cutlass.Int32(0):
                    desired_s      = cutlass.Uint32(smem_tau[0])
                    desired_mask_s = cutlass.Uint32(smem_tau[1])
                    desired_pin_s  = desired_s & desired_mask_s

                    # Clear warp sub-hists.
                    ci = tidx
                    while ci < cutlass.Int32(num_warps * HIST_BINS):
                        smem_warp_hist[ci] = cutlass.Int32(0)
                        ci = ci + cutlass.Int32(topk_threads)
                    cute.arch.sync_threads()

                    # Bin elements matching desired_mask.  USE_LIMIT_TOPK_SEQ_LEN
                    # is a compile-time switch: True ⇒ SMEM-only, False ⇒ GMEM-only.
                    if cutlass.const_expr(USE_LIMIT_TOPK_SEQ_LEN):
                        pos = tidx
                        while pos < sl:
                            bits = cutlass.Uint32(smem_bits[pos])
                            if (bits & desired_mask_s) == desired_pin_s:
                                bin = cutlass.Int32((bits >> digit_pos_u) & cutlass.Uint32(0xFF))
                                cute.arch.atomic_add(warp_hist_ptr + (warp_base + bin),
                                                     cutlass.Int32(1),
                                                     sem="relaxed", scope="cta")
                            pos = pos + cutlass.Int32(topk_threads)
                    else:
                        pos = tidx
                        while pos < sl:
                            bits = float_to_radix(score_output[bidx, pos])
                            if (bits & desired_mask_s) == desired_pin_s:
                                bin = cutlass.Int32((bits >> digit_pos_u) & cutlass.Uint32(0xFF))
                                cute.arch.atomic_add(warp_hist_ptr + (warp_base + bin),
                                                     cutlass.Int32(1),
                                                     sem="relaxed", scope="cta")
                            pos = pos + cutlass.Int32(topk_threads)
                    cute.arch.sync_threads()

                    # Merge 32 sub-hists → smem_hist.
                    if tidx < cutlass.Int32(HIST_BINS):
                        s = cutlass.Int32(0)
                        for w in cutlass.range_constexpr(32):  # num_warps = 32
                            s = s + smem_warp_hist[w * HIST_BINS + tidx]
                        smem_hist[tidx] = s
                    cute.arch.sync_threads()

                    # τ-find on tid 0, fully unrolled over 256 bins.
                    if tidx == cutlass.Int32(0):
                        k_need = smem_tau[3]
                        acc    = cutlass.Int32(0)
                        tau_b  = cutlass.Int32(0)
                        done   = cutlass.Int32(0)
                        for i_c in cutlass.range_constexpr(HIST_BINS):
                            bi_c = HIST_BINS - 1 - i_c
                            if done == cutlass.Int32(0):
                                cnt_b = smem_hist[bi_c]
                                if acc + cnt_b >= k_need:
                                    tau_b = cutlass.Int32(bi_c)
                                    done  = cutlass.Int32(1)
                                else:
                                    acc = acc + cnt_b
                        new_desired_mask = desired_mask_s | (cutlass.Uint32(0xFF) << digit_pos_u)
                        new_desired      = desired_s | ((cutlass.Uint32(tau_b) & cutlass.Uint32(0xFF)) << digit_pos_u)
                        chosen_cnt       = smem_hist[tau_b]
                        new_above_total  = smem_tau[2] + acc
                        new_k_to_find    = k_need - acc

                        smem_tau[0] = cutlass.Int32(new_desired)
                        smem_tau[1] = cutlass.Int32(new_desired_mask)
                        smem_tau[2] = new_above_total
                        smem_tau[3] = new_k_to_find
                        if chosen_cnt == new_k_to_find:
                            smem_tau[4] = cutlass.Int32(1)
                    cute.arch.sync_threads()

            desired      = cutlass.Uint32(smem_tau[0])
            desired_mask = cutlass.Uint32(smem_tau[1])
            above_total  = smem_tau[2]
            need_ties    = smem_tau[3]
            desired_pin = desired & desired_mask

            # ── Phase 2: fused scatter pass (reads from smem_bits or gmem) ──
            above_cursor = cutlass.Int32(0)
            tie_cursor   = cutlass.Int32(0)

            col = cutlass.Int32(0)
            while col < sl:
                cur_col  = col + tidx
                is_valid = cur_col < sl

                bits = cutlass.Uint32(0)
                if is_valid:
                    if cutlass.const_expr(USE_LIMIT_TOPK_SEQ_LEN):
                        bits = cutlass.Uint32(smem_bits[cur_col])
                    else:
                        bits = float_to_radix(score_output[bidx, cur_col])

                is_b = cutlass.Int32(0)
                is_t = cutlass.Int32(0)
                if is_valid:
                    masked = bits & desired_mask
                    if masked > desired_pin:
                        is_b = cutlass.Int32(1)
                    if masked == desired_pin:
                        is_t = cutlass.Int32(1)

                scan_b = is_b
                for s in cutlass.range_constexpr(5):
                    peer = cute.arch.shuffle_sync_up(scan_b, 1 << s, mask_and_clamp=0)
                    if lane_idx >= cutlass.Int32(1 << s):
                        scan_b = scan_b + peer
                my_b_excl  = scan_b - is_b
                warp_b_tot = cute.arch.shuffle_sync(scan_b, 31)

                scan_t = is_t
                for s in cutlass.range_constexpr(5):
                    peer2 = cute.arch.shuffle_sync_up(scan_t, 1 << s, mask_and_clamp=0)
                    if lane_idx >= cutlass.Int32(1 << s):
                        scan_t = scan_t + peer2
                my_t_excl  = scan_t - is_t
                warp_t_tot = cute.arch.shuffle_sync(scan_t, 31)

                if lane_idx == cutlass.Int32(31):
                    smem_warp_above[warp_idx] = warp_b_tot
                    smem_warp_tie[warp_idx]   = warp_t_tot
                cute.arch.sync_threads()

                if warp_idx == cutlass.Int32(0):
                    wta      = smem_warp_above[lane_idx]
                    orig_wta = wta
                    for s in cutlass.range_constexpr(5):
                        p = cute.arch.shuffle_sync_up(wta, 1 << s, mask_and_clamp=0)
                        if lane_idx >= cutlass.Int32(1 << s):
                            wta = wta + p
                    smem_warp_above[lane_idx] = wta - orig_wta
                    above_round_tot = warp_sum_i32(orig_wta)
                    if lane_idx == cutlass.Int32(0):
                        smem_above_round[0] = above_round_tot

                    wtt      = smem_warp_tie[lane_idx]
                    orig_wtt = wtt
                    for s in cutlass.range_constexpr(5):
                        p2 = cute.arch.shuffle_sync_up(wtt, 1 << s, mask_and_clamp=0)
                        if lane_idx >= cutlass.Int32(1 << s):
                            wtt = wtt + p2
                    smem_warp_tie[lane_idx] = wtt - orig_wtt
                    tie_round_tot = warp_sum_i32(orig_wtt)
                    if lane_idx == cutlass.Int32(0):
                        smem_tie_round[0] = tie_round_tot
                cute.arch.sync_threads()

                warp_b_off = smem_warp_above[warp_idx]
                warp_t_off = smem_warp_tie[warp_idx]

                if is_b > cutlass.Int32(0):
                    goff = above_cursor + warp_b_off + my_b_excl
                    if goff < above_total:
                        page_local_b  = cur_col // cutlass.Int32(PAGE_SIZE)
                        tok_offset_b  = cur_col - page_local_b * cutlass.Int32(PAGE_SIZE)
                        global_page_b = cutlass.Int32(block_table[bidx, page_local_b])
                        topk_indices[bidx, goff] = (
                            global_page_b * cutlass.Int32(PAGE_SIZE) + tok_offset_b
                        )

                if is_t > cutlass.Int32(0):
                    toff    = tie_cursor + warp_t_off + my_t_excl
                    wrt_pos = above_total + toff
                    if toff < need_ties:
                        if wrt_pos < cutlass.Int32(top_k_len):
                            page_local_t  = cur_col // cutlass.Int32(PAGE_SIZE)
                            tok_offset_t  = cur_col - page_local_t * cutlass.Int32(PAGE_SIZE)
                            global_page_t = cutlass.Int32(block_table[bidx, page_local_t])
                            topk_indices[bidx, wrt_pos] = (
                                global_page_t * cutlass.Int32(PAGE_SIZE) + tok_offset_t
                            )

                above_round = smem_above_round[0]
                tie_round   = smem_tie_round[0]
                cute.arch.sync_threads()

                above_cursor = above_cursor + above_round
                tie_cursor   = tie_cursor   + tie_round
                col          = col + cutlass.Int32(topk_threads)


# ── Compilation helper ────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_hybrid():
    T_            = cute.sym_int()
    max_num_pages = cute.sym_int()
    num_pages     = cute.sym_int()

    q_index_fp8       = _fake(cute.Float8E4M3FN, (T_, NUM_HEADS, HEAD_DIM),               (2, 1, 0),    16)
    k_index_cache_fp8 = _fake(cute.Int8,         (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4), (3, 2, 1, 0), 16)
    weights           = _fake(cute.Float32,      (T_, NUM_HEADS),                          (1, 0),       4)
    seq_lens          = _fake(cute.Int32,        (T_,),                                    (0,),         4)
    block_table       = _fake(cute.Int32,        (T_, max_num_pages),                      (1, 0),       4)
    score_output      = _fake(cute.Float32,      (LIMIT_REQUEST, LIMIT_SEQ_LEN),           (1, 0),       16)
    top_k_indices     = _fake(cute.Int32,        (T_, TOP_K),                              (1, 0),       4)
    stream            = make_fake_stream(use_tvm_ffi_env_stream=True)

    indexer  = Indexer_kvsplit_v4_hist_pdl()
    compiled = cute.compile(
        indexer,
        q_index_fp8, k_index_cache_fp8, weights,
        seq_lens, block_table, score_output, top_k_indices, stream,
        options="--enable-tvm-ffi",
    )
    return indexer, compiled


_indexer, _compiled = compile_hybrid()


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    _compiled(
        q_index_fp8, k_index_cache_fp8, weights,
        seq_lens, block_table, _indexer.ws_score_output, topk_indices,
    )
