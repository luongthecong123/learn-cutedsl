"""kv_split_umma_v3_stages_pe_prologue.py

Faithful, fully-functional port of `kv_split_umma_v3_1024.py` with two
structural changes on the UMMA path:

  1. **PE prologue.** PE work (q_pe @ kpe^T) is hoisted out of the per-token
     UMMA loop into a one-shot prologue. PE for token i is packed into
     panel `i` of the existing sA/sB byte budget (no extra smem). The 9th
     "KPE" panel from v3 is dropped — sA/sB now have 8 K-panels of 64
     bf16 each (HEAD_DIM_CKV = 512). PE UMMAs are fired *unconditionally*
     for all `LIMIT_REQUEST` slots (cheap, garbage in unused slots is
     never read by the consumer).

  2. **Producer / consumer split inside the UMMA path.** The 512 UMMA
     threads (16 warps) are now specialized:
        * warps 0..7   = PRODUCER (256 thr, 32 reg/thread).
                         Pure cp.async + UMMA fires + index calculation.
                         Walk every full split (`num_valid == DIM_SPLIT`)
                         and emit q_nope/CKV in 4 sequential chunks per
                         token; commit `score_mbars[T_idx]` when done.
        * warps 8..15  = CONSUMER (256 thr, default reg).
                         Wait `score_mbars[T_idx]`, do tmem→smem score,
                         softmax (writes partial_lse), 4-stage output GEMV
                         (writes partial_out).
     Per-token tmem slots (`LIMIT_REQUEST × MMA_N` cols) keep the
     producer/consumer pipeline non-overlapping (serialized for now —
     this kernel sets up the layout for later inter-token pipelining).

The SGEMM path (warps 16..31, 512 thr) and the reduce kernel are byte-for-byte
identical to v3_1024.

Work-assignment prologue still runs on warps 8..15 (consumer half), since
the producer warps must stay light (32 reg) and never touch heavy index work.
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
import math
import torch

# ── Input constants (matched to v3) ───────────────────────────────────────────
NUM_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
TOP_K = 2048
NUM_PAGES = 8462
PAGE_SIZE = 64
FLAT_CACHE = NUM_PAGES * PAGE_SIZE
LN2 = 0.6931471805599453
SM_SCALE: cutlass.Constexpr = 0.1352337788608801
LIMIT_REQUEST = 8
assert LIMIT_REQUEST <= 8
DIM_CHUNK = 8
NUM_SPLITS = 16
DIM_SPLIT = (TOP_K + NUM_SPLITS - 1) // NUM_SPLITS  # 128
HEADS_PER_SPLIT = 2

# ── Stages-kernel constants ───────────────────────────────────────────────────
# UMMA inst shape
_MMA_M, _MMA_N, _MMA_K = DIM_SPLIT, 8, 16
_MMA_K_PACK   = 4
_MMA_K_PACKED = _MMA_K * _MMA_K_PACK              # 64 bf16 per panel
_MMA_K_TILES  = HEAD_DIM_CKV // _MMA_K_PACKED     # 8 CKV panels
# 8 panels — PE for token i packs into panel i during the prologue, then those
# panels are reused for CKV in the main loop.
_MMA_K_TILES_FULL = _MMA_K_TILES                  # 8

# CKV chunking (4 stages × 2 panels each)
PANELS_PER_CHUNK: cutlass.Constexpr = 2
NUM_CKV_CHUNKS:   cutlass.Constexpr = _MMA_K_TILES // PANELS_PER_CHUNK   # 4
CHUNK_PACKED:     cutlass.Constexpr = _MMA_K_PACKED * PANELS_PER_CHUNK   # 128
CKV_KBLOCKS_PER_CHUNK: cutlass.Constexpr = _MMA_K_PACK * PANELS_PER_CHUNK  # 8

TMEM_COLS_PER_TOKEN = _MMA_N                      # 8

# Bar IDs
PROLOGUE_BAR_ID    = 1
PROD_BAR_ID        = 2
CONS_BAR_ID        = 3
SGEMM_BAR_ID       = 4
UMMA_TOKEN_BAR_ID  = 5  # producer/consumer rendezvous between tokens (uses sA)


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.jit
def _panel_copy_layout(num_rows: int, k_packed: int, k_tiles: int):
    return cute.make_layout((num_rows, (k_packed, k_tiles)),
                            stride=(k_packed, (1, num_rows * k_packed)))


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Dsa():
    def __init__(self):
        self.wsize = cute.arch.WARP_SIZE

        # Prologue / output
        self.swz_rot_shift = 7
        self.sp_vec_size_i32 = 4
        self.out_stages = 4
        self.out_vec = HEAD_DIM_CKV // (self.out_stages * self.wsize)  # 4

        # ── UMMA workers (full splits) — split into PRODUCER + CONSUMER ──
        self.umma_threads      = 512
        self.num_umma_warps    = self.umma_threads // self.wsize    # 16
        self.num_prod_warps    = 8
        self.num_cons_warps    = 8
        self.prod_threads      = self.num_prod_warps * self.wsize   # 256
        self.cons_threads      = self.num_cons_warps * self.wsize   # 256
        self.umma_inst         = (DIM_SPLIT, 8, 16)
        self.tmem_ld_rep       = HEADS_PER_SPLIT
        self.ab_dtype          = cutlass.BFloat16
        self.acc_dtype         = cutlass.Float32

        # ── SGEMM workers (partial splits) — unchanged from v3_1024 ──
        self.sgemm_threads     = 512
        self.num_sgemm_warps   = self.sgemm_threads // self.wsize   # 16
        self.sgemm_ckv_vec     = 4
        self.sgemm_kpe_vec     = 2

        # ── Reduce kernel ──
        self.reduce_threads = 256
        self.reduce_warps   = self.reduce_threads // self.wsize
        self.vec_reduce     = 2

        # Persistent partial workspace
        self.partial_out = torch.empty(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.float32, device="cuda")
        self.partial_lse = torch.empty(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2,            dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(
        self,
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_cache:      cute.Tensor,
        kpe_cache:      cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor,
        stream,
    ):
        T, _, _ = q_nope.shape
        ckv_flat = cute.make_tensor(ckv_cache.iterator,
            cute.make_layout((FLAT_CACHE, HEAD_DIM_CKV), stride=(HEAD_DIM_CKV, 1)))
        kpe_flat = cute.make_tensor(kpe_cache.iterator,
            cute.make_layout((FLAT_CACHE, HEAD_DIM_KPE), stride=(HEAD_DIM_KPE, 1)))

        op = tcgen05.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.umma_inst,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        @cute.struct
        class SharedStorage:
            score_mbars:      cute.struct.MemRange[cutlass.Int64, LIMIT_REQUEST]
            prologue_mbars:   cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.compute_kernel(
            tiled_mma,
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
            partial_out, partial_lse, output, lse,
        ).launch(grid=[NUM_HEADS // HEADS_PER_SPLIT, NUM_SPLITS, 1],
                 block=[self.umma_threads + self.sgemm_threads, 1, 1], stream=stream)

        self.reduce_kernel(
            sparse_indices, partial_out, partial_lse, output, lse,
        ).launch(grid=[T, NUM_HEADS, 1],
                 block=[self.reduce_threads, 1, 1], stream=stream)

    @staticmethod
    def _smem(allocator, dtype, shape, stride, byte_alignment=16, swizzle=None):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), byte_alignment, swizzle)

    @cute.kernel
    def compute_kernel(
        self,
        tiled_mma,
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

        # ========= SMEM setup =========
        alloc = cutlass.utils.SmemAllocator()

        smem_sp_indices = self._smem(alloc, cutlass.Int32,   (DIM_CHUNK, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_assign     = self._smem(alloc, cutlass.Int32,   (DIM_CHUNK, 2),         (2, 1))
        # Per-path score/logits buffers (UMMA path uses the consumer-only ones)
        smem_score        = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_score_sgemm  = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_logits_flat       = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))
        smem_logits_flat_sgemm = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))

        # UMMA partial out (8 consumer warps × 2 heads × 128 = 8 KB)
        smem_partial_umma = self._smem(alloc, cutlass.Float32,
            (self.num_cons_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
            (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))

        # SGEMM partial out (16 warps × 2 heads × 128 = 16 KB)
        smem_partial_sgemm = self._smem(alloc, cutlass.Float32,
            (self.num_sgemm_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
            (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))

        # ── UMMA smem: 8-panel layout (panels 0..7 hold PE in prologue, then CKV) ──
        swizzle = cute.make_swizzle(3, 4, 3)
        a_outer = cute.make_layout(
            ((_MMA_M, _MMA_K), 1, (_MMA_K_PACK, _MMA_K_TILES_FULL)),
            stride=((_MMA_K_PACKED, 1), 0, (_MMA_K, _MMA_M * _MMA_K_PACKED)))
        b_outer = cute.make_layout(
            ((_MMA_N, _MMA_K), 1, (_MMA_K_PACK, _MMA_K_TILES_FULL)),
            stride=((_MMA_K_PACKED, 1), 0, (_MMA_K, _MMA_N * _MMA_K_PACKED)))
        sA = alloc.allocate_tensor(cutlass.BFloat16, a_outer, byte_alignment=16, swizzle=swizzle)
        sB = alloc.allocate_tensor(cutlass.BFloat16, b_outer, byte_alignment=16, swizzle=swizzle)
        sA_ckv_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(_MMA_M, _MMA_K_PACKED, _MMA_K_TILES))
        sB_ckv_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(_MMA_N, _MMA_K_PACKED, _MMA_K_TILES))

        panel_stride_A: cutlass.Constexpr = _MMA_M * _MMA_K_PACKED
        panel_stride_B: cutlass.Constexpr = _MMA_N * _MMA_K_PACKED
        chunk_stride_A: cutlass.Constexpr = panel_stride_A * PANELS_PER_CHUNK
        chunk_stride_B: cutlass.Constexpr = panel_stride_B * PANELS_PER_CHUNK

        k_split_shape_chunk = cute.make_layout(((_MMA_K_PACKED, PANELS_PER_CHUNK),))
        k_split_shape_pe    = cute.make_layout(((_MMA_K_PACKED, 1),))

        # ── cp.async tiled copies ─────────────────────────────────────────────
        # Chunked CKV copy (64-bit, 4 elems/lane) — legacy
        atom_cpa_chunk   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=64)
        thr_layout       = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout_chunk = cute.make_layout(((4, 1),), stride=((1, 0),))
        tiled_copy_chunk = cute.make_tiled_copy_tv(atom_cpa_chunk, thr_layout, val_layout_chunk)
        lane_copy_chunk  = tiled_copy_chunk.get_slice(lane_idx)

        # PE copy (32-bit, 2 elems/lane) — legacy
        atom_cpa_pe   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        val_layout_pe = cute.make_layout(((2, 1),), stride=((1, 0),))
        tiled_copy_pe = cute.make_tiled_copy_tv(atom_cpa_pe, thr_layout, val_layout_pe)
        lane_copy_pe  = tiled_copy_pe.get_slice(lane_idx)

        # ---- 128-bit cp.async (16B per transaction) ----
        # CKV chunk row = 128 bf16 = 256 B → 16 threads/row, 8 vals each.
        atom_cpa_chunk128   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout_chunk128 = cute.make_layout(((16, 1),), stride=((1, 16),))
        val_layout_chunk128 = cute.make_layout(((8, 1),),  stride=((1, 0),))
        tiled_copy_chunk128 = cute.make_tiled_copy_tv(atom_cpa_chunk128, thr_layout_chunk128, val_layout_chunk128)
        lane_copy_chunk128  = tiled_copy_chunk128.get_slice(lane_idx % 16)

        # PE row (KPE = QPE = 64 bf16 = 128 B) → 8 threads/row, 8 vals each.
        atom_cpa_pe128   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout_pe128 = cute.make_layout(((8, 1),), stride=((1, 8),))
        val_layout_pe128 = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy_pe128 = cute.make_tiled_copy_tv(atom_cpa_pe128, thr_layout_pe128, val_layout_pe128)
        lane_copy_pe128  = tiled_copy_pe128.get_slice(lane_idx % 8)

        # Full-width 128-bit copy for output GEMV's sA reads (kept for parity with v3)
        sA_ckv_out = cute.zipped_divide(sA_ckv_copy, (1, self.out_vec))

        storage             = alloc.allocate(self.shared_storage)
        score_mbar_base     = storage.score_mbars.data_ptr()
        prologue_mbar_base  = storage.prologue_mbars.data_ptr()

        # ========= Work-assignment prologue (warps 8..15 = consumer half) =========
        head_base_idx, split_idx_old, _ = cute.arch.block_idx()
        T, _, _ = q_nope.shape

        sparse_indices_  = cute.zipped_divide(sparse_indices, (1, self.sp_vec_size_i32))
        smem_sp_indices_ = cute.zipped_divide(smem_sp_indices, (1, self.sp_vec_size_i32))
        if DIM_CHUNK <= warp_idx < DIM_CHUNK + T:
            warp_idx_assign = warp_idx - DIM_CHUNK
            split_idx_new = (split_idx_old + warp_idx_assign * self.swz_rot_shift) % cutlass.Int32(NUM_SPLITS)
            split_vec_stride = DIM_SPLIT // self.sp_vec_size_i32
            si_vec = sparse_indices_[(0, None), (warp_idx_assign, split_idx_new * split_vec_stride + lane_idx)].load()
            num_valid_partial = 0
            for v in range(self.sp_vec_size_i32):
                val = si_vec[v]
                if 0 <= val < FLAT_CACHE:
                    num_valid_partial += 1
                else:
                    val = 0
                smem_sp_indices_[(0, v), (warp_idx_assign, lane_idx)] = val
            num_valid = warp_reduce(num_valid_partial, lambda a, b: a + b, width=self.wsize)
            if lane_idx == 0:
                smem_assign[warp_idx_assign, 0] = split_idx_new
                smem_assign[warp_idx_assign, 1] = num_valid

        # ── tmem / mbarrier setup ──
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C((_MMA_M, _MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        per_token_cols  = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(per_token_cols * LIMIT_REQUEST)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                for i in range(LIMIT_REQUEST):
                    cute.arch.mbarrier_init(score_mbar_base + i, cnt=1)
                # 8 warp_groups commit → cnt=8 arrivals.
                cute.arch.mbarrier_init(prologue_mbar_base, cnt=8)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)

        tCtAcc_base     = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        M_acc           = cute.size(tCtAcc_base, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler       = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi_base = cute.zipped_divide(tCtAcc_base, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, cutlass.Float32)
        cons_tidx       = tidx - self.prod_threads
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi_base[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(cons_tidx)
        tTR_tAcc_base   = tmem_thr_copy.partition_S(tCtAcc_epi_base)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc_base[None, None, 0].shape, cutlass.Float32)

        # Hoisted views
        smem_score_              = cute.zipped_divide(smem_score,              (1, DIM_SPLIT // self.wsize))
        smem_score_sgemm_        = cute.zipped_divide(smem_score_sgemm,        (1, DIM_SPLIT // self.wsize))
        smem_logits_flat_        = cute.zipped_divide(smem_logits_flat,        (HEADS_PER_SPLIT,))
        smem_logits_flat_sgemm_  = cute.zipped_divide(smem_logits_flat_sgemm,  (HEADS_PER_SPLIT,))
        smem_partial_umma_  = cute.zipped_divide(smem_partial_umma,  (1, 1, self.out_vec))
        smem_partial_sgemm_ = cute.zipped_divide(smem_partial_sgemm, (1, 1, self.out_vec))
        ckv_flat_out        = cute.zipped_divide(ckv_flat,           (1, self.out_vec))
        # SGEMM score views (q from gmem, k from gmem)
        q_nope_z   = cute.zipped_divide(q_nope,   (1, 1, self.sgemm_ckv_vec))
        q_pe_z     = cute.zipped_divide(q_pe,     (1, 1, self.sgemm_kpe_vec))
        ckv_flat_z = cute.zipped_divide(ckv_flat, (1, self.sgemm_ckv_vec))
        kpe_flat_z = cute.zipped_divide(kpe_flat, (1, self.sgemm_kpe_vec))

        # ============================================================
        # ROLE SPLIT
        # ============================================================
        is_producer = warp_idx < self.num_prod_warps                       # 0..7
        is_consumer = (warp_idx >= self.num_prod_warps) & (warp_idx < self.num_umma_warps)  # 8..15

        # ============================================================
        # PE PROLOGUE — fully cooperative across all 32 warps (1024 thr).
        # 8 warp_groups × 4 warps each. warp_group g owns token g.
        # 128b cp.async: 8 thr/row. Within a warp_group (128 thr): 16 row groups
        # cover 128 KPE rows in kpe_rows_per_group=8 rounds of 16.
        # ============================================================
        warp_group_idx = warp_idx // 4         # 0..7
        lane_wg        = warp_idx %  4         # 0..3
        kpe_row_group_in_wg = lane_wg * 4 + (lane_idx // 8)   # 0..15
        kpe_rows_per_group: cutlass.Constexpr = _MMA_M // 16  # 8

        for i in cutlass.range_constexpr(LIMIT_REQUEST):
            if i < T:
                num_valid_pe = smem_assign[i, 1]
                if num_valid_pe == DIM_SPLIT:
                    sA_pe_i = cute.make_tensor(
                        sA.iterator + i * panel_stride_A,
                        _panel_copy_layout(_MMA_M, _MMA_K_PACKED, 1))
                    sB_pe_i = cute.make_tensor(
                        sB.iterator + i * panel_stride_B,
                        _panel_copy_layout(_MMA_N, _MMA_K_PACKED, 1))

                    if warp_group_idx == i:
                        # q_pe: lane_wg==0, lanes 0..15 own qpe_row ∈ {0,1}.
                        if lane_wg == 0:
                            qpe_row = lane_idx // 8
                            if qpe_row < HEADS_PER_SPLIT:
                                head_h = head_base_idx * HEADS_PER_SPLIT + qpe_row
                                cute.copy(atom_cpa_pe128,
                                          lane_copy_pe128.partition_S(cute.composition(q_pe[i, head_h, None], k_split_shape_pe)),
                                          lane_copy_pe128.partition_D(sB_pe_i[qpe_row, None]))

                        # kpe rows: 16 row groups × 8 rows/group = 128 rows.
                        for r in range(kpe_rows_per_group):
                            row_idx  = r * 16 + kpe_row_group_in_wg
                            flat_row = smem_sp_indices[i, row_idx]
                            cute.copy(atom_cpa_pe128,
                                      lane_copy_pe128.partition_S(cute.composition(kpe_flat[flat_row, None], k_split_shape_pe)),
                                      lane_copy_pe128.partition_D(sA_pe_i[row_idx, None]))

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()   # 1024-thread fence

        tcgen05_fence()
        # PE MMA: warp_group g lead warp (lane_wg==0) fires for token g.
        if lane_wg == 0:
            for g in cutlass.range_constexpr(LIMIT_REQUEST):
                if warp_group_idx == g and g < T:
                    num_valid_g = smem_assign[g, 1]
                    if num_valid_g == DIM_SPLIT:
                        tCtAcc_g = cute.make_tensor(
                            tmem_ptr + cutlass.Int32(g * TMEM_COLS_PER_TOKEN),
                            tCtAcc_tmpl.layout)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in range(_MMA_K_PACK):
                            k_flat = g * _MMA_K_PACK + kb
                            coord  = (None, None, k_flat)
                            cute.gemm(tiled_mma, tCtAcc_g,
                                      tCrA[coord], tCrB[coord], tCtAcc_g)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            # Each warp_group lead lane commits unconditionally → cnt=8.
            if lane_idx == 0:
                tcgen05.commit(prologue_mbar_base)

        # ============================================================
        # PRODUCER (warps 0..7, 256 thr)
        # Per-token CKV-chunked UMMA fires.
        # ============================================================
        if is_producer:
            # All producer warps wait for PE UMMAs to retire before reusing sA/sB.
            cute.arch.mbarrier_wait(prologue_mbar_base, cutlass.Int32(0))

            # ─── Main producer loop ───────────────────────────────────────
            num_rounds = DIM_SPLIT // 16                                # 8
            prod_row_group = warp_idx * 2 + (lane_idx // 16)            # 0..15
            PROD_ROW_GROUPS: cutlass.Constexpr = 16

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    num_valid = smem_assign[T_idx, 1]
                    if num_valid == DIM_SPLIT:
                        tCtAcc_i = cute.make_tensor(
                            tmem_ptr + cutlass.Int32(T_idx * TMEM_COLS_PER_TOKEN),
                            tCtAcc_tmpl.layout)

                        for c in cutlass.range_constexpr(NUM_CKV_CHUNKS):
                            sA_chunk = cute.make_tensor(
                                sA.iterator + c * chunk_stride_A,
                                _panel_copy_layout(_MMA_M, _MMA_K_PACKED, PANELS_PER_CHUNK))
                            sB_chunk = cute.make_tensor(
                                sB.iterator + c * chunk_stride_B,
                                _panel_copy_layout(_MMA_N, _MMA_K_PACKED, PANELS_PER_CHUNK))

                            # q_nope chunk → sB (row groups 0..1 own one head each)
                            if prod_row_group < HEADS_PER_SPLIT:
                                head_h = head_base_idx * HEADS_PER_SPLIT + prod_row_group
                                q_nope_chunk = cute.make_tensor(
                                    q_nope[T_idx, head_h, None].iterator + c * CHUNK_PACKED,
                                    cute.make_layout((CHUNK_PACKED,), stride=(1,)))
                                cute.copy(atom_cpa_chunk128,
                                          lane_copy_chunk128.partition_S(cute.composition(q_nope_chunk, k_split_shape_chunk)),
                                          lane_copy_chunk128.partition_D(sB_chunk[prod_row_group, None]))

                            # CKV chunk → sA (8 rows per row-group)
                            for round_idx in range(num_rounds):
                                row_idx  = round_idx * PROD_ROW_GROUPS + prod_row_group
                                flat_row = smem_sp_indices[T_idx, row_idx]
                                ckv_chunk = cute.make_tensor(
                                    ckv_flat[flat_row, None].iterator + c * CHUNK_PACKED,
                                    cute.make_layout((CHUNK_PACKED,), stride=(1,)))
                                cute.copy(atom_cpa_chunk128,
                                          lane_copy_chunk128.partition_S(cute.composition(ckv_chunk, k_split_shape_chunk)),
                                          lane_copy_chunk128.partition_D(sA_chunk[row_idx, None]))

                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.fence_view_async_shared()
                            cute.arch.barrier(barrier_id=PROD_BAR_ID, number_of_threads=self.prod_threads)

                            tcgen05_fence()
                            if warp_idx == 0:
                                # ACC=True throughout — PE prologue seeded the slot.
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                for kb in range(CKV_KBLOCKS_PER_CHUNK):
                                    k_flat = c * CKV_KBLOCKS_PER_CHUNK + kb
                                    coord  = (None, None, k_flat)
                                    cute.gemm(tiled_mma, tCtAcc_i,
                                              tCrA[coord], tCrB[coord], tCtAcc_i)

                        if warp_idx == 0 and lane_idx == 0:
                            tcgen05.commit(score_mbar_base + T_idx)

                        # Producer waits for its own MMAs to retire before
                        # the next iteration overwrites sA with new CKV.
                        # (Mirrors the working kernel's
                        #  `mbarrier_wait(score_mbar_base + (i-1), 0)` at top of iter.)
                        cute.arch.mbarrier_wait(score_mbar_base + T_idx,
                                                cutlass.Int32(0))

                # Rendezvous with consumer at every T_idx (T_idx < T) so the
                # consumer's output GEMV (which reads sA) finishes before the
                # producer overwrites sA with the next token's CKV.
                cute.arch.barrier(barrier_id=UMMA_TOKEN_BAR_ID,
                                  number_of_threads=self.umma_threads)

        # ============================================================
        # CONSUMER (warps 8..15, 256 thr)
        # tmem ld → softmax → 4-stage output GEMV
        # ============================================================
        if is_consumer:
            cons_warp_idx = warp_idx - self.num_prod_warps                # 0..7

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]

                    if num_valid == DIM_SPLIT:
                        cute.arch.mbarrier_wait(score_mbar_base + T_idx, cutlass.Int32(0))

                        tTR_tAcc_i = cute.make_tensor(
                            tTR_tAcc_base.iterator + cutlass.Int32(T_idx * TMEM_COLS_PER_TOKEN),
                            tTR_tAcc_base.layout)

                        if cons_tidx < DIM_SPLIT:
                            cute.copy(tmem_tiled_copy, tTR_tAcc_i[None, None, 0], tTR_rAcc)
                            smem_score[0, cons_tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                            smem_score[1, cons_tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

                        cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)

                        # Softmax (2 warps, 1 head each)
                        if cons_warp_idx < HEADS_PER_SPLIT:
                            num_elems: cutlass.Constexpr = DIM_SPLIT // self.wsize
                            head_idx_global = head_base_idx * HEADS_PER_SPLIT + cons_warp_idx
                            vec = smem_score_[(0, None), (cons_warp_idx, lane_idx)].load()
                            vec_masked = cute.make_rmem_tensor(
                                cute.make_layout((num_elems,), stride=(1,)), cutlass.Float32)
                            for v_idx in range(num_elems):
                                vec_masked[v_idx] = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                if col_idx < num_valid:
                                    vec_masked[v_idx] = vec[v_idx]
                            row_max = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                row_max = cute.arch.fmax(row_max, vec_masked[v_idx])
                            row_max = warp_reduce(row_max, cute.arch.fmax)
                            row_sum = cutlass.Float32(0)
                            for v_idx in range(num_elems):
                                e = cute.math.exp(vec_masked[v_idx] - row_max)
                                vec_masked[v_idx] = e
                                row_sum += e
                            row_sum = warp_reduce(row_sum, lambda a, b: a + b)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                smem_logits_flat[col_idx * HEADS_PER_SPLIT + cons_warp_idx] = vec_masked[v_idx]
                            if lane_idx == 0:
                                partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                                partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum

                        cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)

                        # Output GEMV — 4 stages × 8 consumer warps (16 rounds each)
                        num_rounds_out: cutlass.Constexpr = DIM_SPLIT // self.num_cons_warps
                        out0 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        for stage_idx in range(self.out_stages):
                            out0.fill(cutlass.Float32(0))
                            out1.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out):
                                k = round_idx * self.num_cons_warps + cons_warp_idx
                                if k < num_valid:
                                    gmem_ckv_vec = sA_ckv_out[(0, None), (k, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0[v_idx], out1[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0[v_idx], out1[v_idx]))
                            smem_partial_umma_[(0, 0, None), (cons_warp_idx, 0, lane_idx)].store(out0.load())
                            smem_partial_umma_[(0, 0, None), (cons_warp_idx, 1, lane_idx)].store(out1.load())
                            cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)
                            thr_group_idx  = cons_tidx // DIM_SPLIT
                            thr_group_lane = cons_tidx %  DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_cons_warps):
                                    final_sum += smem_partial_umma[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum
                            cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)

                # Rendezvous with producer at every T_idx (T_idx < T).
                cute.arch.barrier(barrier_id=UMMA_TOKEN_BAR_ID,
                                  number_of_threads=self.umma_threads)

        if warp_idx >= self.num_umma_warps:
            sgemm_warp_idx = warp_idx - self.num_umma_warps  # 0..15
            sgemm_tidx     = tidx - self.umma_threads        # 0..511

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]

                    if 0 < num_valid < DIM_SPLIT:
                        head_idx0 = head_base_idx * HEADS_PER_SPLIT
                        head_idx1 = head_base_idx * HEADS_PER_SPLIT + 1

                        # ── Phase 1: Score ──
                        num_rounds_score = (num_valid + self.num_sgemm_warps - 1) // self.num_sgemm_warps
                        for round_idx in range(num_rounds_score):
                            col_idx = round_idx * self.num_sgemm_warps + sgemm_warp_idx
                            if col_idx < num_valid:
                                flat_cache_idx = smem_sp_indices[T_idx, col_idx]
                                acc0 = cutlass.Float32(0)
                                acc1 = cutlass.Float32(0)

                                for i in range(HEAD_DIM_CKV // (self.sgemm_ckv_vec * self.wsize)):
                                    row_idx = i * self.wsize + lane_idx
                                    qn0_frag = q_nope_z[(0, 0, None), (T_idx, head_idx0, row_idx)].load().to(cutlass.Float32)
                                    qn1_frag = q_nope_z[(0, 0, None), (T_idx, head_idx1, row_idx)].load().to(cutlass.Float32)
                                    ckv_frag = ckv_flat_z[(0, None), (flat_cache_idx, row_idx)].load().to(cutlass.Float32)
                                    for v in range(self.sgemm_ckv_vec):
                                        acc0, acc1 = cute.arch.fma_packed_f32x2(
                                            (qn0_frag[v], qn1_frag[v]),
                                            (ckv_frag[v], ckv_frag[v]),
                                            (acc0, acc1))

                                for i in range(HEAD_DIM_KPE // (self.sgemm_kpe_vec * self.wsize)):
                                    row_idx = i * self.wsize + lane_idx
                                    qp0_frag = q_pe_z[(0, 0, None), (T_idx, head_idx0, row_idx)].load().to(cutlass.Float32)
                                    qp1_frag = q_pe_z[(0, 0, None), (T_idx, head_idx1, row_idx)].load().to(cutlass.Float32)
                                    kpe_frag = kpe_flat_z[(0, None), (flat_cache_idx, row_idx)].load().to(cutlass.Float32)
                                    for v in range(self.sgemm_kpe_vec):
                                        acc0, acc1 = cute.arch.fma_packed_f32x2(
                                            (qp0_frag[v], qp1_frag[v]),
                                            (kpe_frag[v], kpe_frag[v]),
                                            (acc0, acc1))

                                acc0 = warp_reduce(acc0, lambda a, b: a + b)
                                acc1 = warp_reduce(acc1, lambda a, b: a + b)
                                if lane_idx == 0:
                                    smem_score_sgemm[0, col_idx] = acc0 * cutlass.Float32(sm_scale)
                                    smem_score_sgemm[1, col_idx] = acc1 * cutlass.Float32(sm_scale)

                        cute.arch.barrier(barrier_id=SGEMM_BAR_ID, number_of_threads=self.sgemm_threads)

                        # ── Phase 2: Softmax ──
                        if sgemm_warp_idx < HEADS_PER_SPLIT:
                            num_elems: cutlass.Constexpr = DIM_SPLIT // self.wsize
                            head_idx_global = head_base_idx * HEADS_PER_SPLIT + sgemm_warp_idx
                            vec = smem_score_sgemm_[(0, None), (sgemm_warp_idx, lane_idx)].load()
                            vec_masked = cute.make_rmem_tensor(
                                cute.make_layout((num_elems,), stride=(1,)), cutlass.Float32)
                            for v_idx in range(num_elems):
                                vec_masked[v_idx] = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                if col_idx < num_valid:
                                    vec_masked[v_idx] = vec[v_idx]
                            row_max = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                row_max = cute.arch.fmax(row_max, vec_masked[v_idx])
                            row_max = warp_reduce(row_max, cute.arch.fmax)
                            row_sum = cutlass.Float32(0)
                            for v_idx in range(num_elems):
                                e = cute.math.exp(vec_masked[v_idx] - row_max)
                                vec_masked[v_idx] = e
                                row_sum += e
                            row_sum = warp_reduce(row_sum, lambda a, b: a + b)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                smem_logits_flat_sgemm[col_idx * HEADS_PER_SPLIT + sgemm_warp_idx] = vec_masked[v_idx]
                            if lane_idx == 0:
                                partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                                partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum

                        cute.arch.barrier(barrier_id=SGEMM_BAR_ID, number_of_threads=self.sgemm_threads)

                        # ── Phase 3: Output GEMV ──
                        num_rounds_out_s = (num_valid + self.num_sgemm_warps - 1) // self.num_sgemm_warps
                        out0_s = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1_s = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)

                        for stage_idx in range(self.out_stages):
                            out0_s.fill(cutlass.Float32(0))
                            out1_s.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out_s):
                                k = round_idx * self.num_sgemm_warps + sgemm_warp_idx
                                if k < num_valid:
                                    flat_cache_idx = smem_sp_indices[T_idx, k]
                                    gmem_ckv_vec = ckv_flat_out[(0, None), (flat_cache_idx, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_sgemm_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0_s[v_idx], out1_s[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0_s[v_idx], out1_s[v_idx]))

                            smem_partial_sgemm_[(0, 0, None), (sgemm_warp_idx, 0, lane_idx)].store(out0_s.load())
                            smem_partial_sgemm_[(0, 0, None), (sgemm_warp_idx, 1, lane_idx)].store(out1_s.load())

                            cute.arch.barrier(barrier_id=SGEMM_BAR_ID, number_of_threads=self.sgemm_threads)

                            thr_group_idx  = sgemm_tidx // DIM_SPLIT
                            thr_group_lane = sgemm_tidx %  DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_sgemm_warps):
                                    final_sum += smem_partial_sgemm[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum

                            cute.arch.barrier(barrier_id=SGEMM_BAR_ID, number_of_threads=self.sgemm_threads)

        # Epilogue
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

    @cute.kernel
    def reduce_kernel(
        self,
        sparse_indices: cute.Tensor,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        T_idx, head_idx, _ = cute.arch.block_idx()

        alloc = cutlass.utils.SmemAllocator()
        smem_red_i32 = self._smem(alloc, cutlass.Int32,   (32,),          (1,))
        smem_max_sum = self._smem(alloc, cutlass.Float32, (NUM_SPLITS, 2), (2, 1))

        partial_cnt = cutlass.Int32(0)
        for i in range(tidx, TOP_K, self.reduce_threads):
            idx = sparse_indices[T_idx, i]
            if idx >= cutlass.Int32(0):
                partial_cnt += cutlass.Int32(1)

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt_sum
        cute.arch.sync_threads()

        if warp_idx == 0:
            val = cutlass.Int32(0)
            if lane_idx < self.reduce_warps:
                val = smem_red_i32[lane_idx]
            val = warp_reduce(val, lambda a, b: a + b, width=self.reduce_warps)
            if lane_idx == 0:
                smem_red_i32[0] = val
        cute.arch.sync_threads()

        num_valid = smem_red_i32[0]
        num_active_splits = (num_valid + DIM_SPLIT - 1) // DIM_SPLIT

        if tidx < num_active_splits:
            smem_max_sum[tidx, 0] = partial_lse[T_idx, tidx, head_idx, 0]
            smem_max_sum[tidx, 1] = partial_lse[T_idx, tidx, head_idx, 1]
        cute.arch.sync_threads()

        partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, self.vec_reduce))
        output_v      = cute.zipped_divide(output,      (1, 1, self.vec_reduce))

        g_max = -cutlass.Float32(math.inf)
        for s in range(num_active_splits):
            local_max = smem_max_sum[s, 0]
            if local_max > g_max:
                g_max = local_max

        g_lse_sum = cutlass.Float32(0)
        acc_rmem = cute.make_rmem_tensor(cute.make_layout((self.vec_reduce,), stride=(1,)), cutlass.Float32)
        acc_rmem[0] = cutlass.Float32(0)
        acc_rmem[1] = cutlass.Float32(0)
        acc = acc_rmem.load()

        for s in range(num_active_splits):
            l_max = smem_max_sum[s, 0]
            l_sum = smem_max_sum[s, 1]
            scale = cute.math.exp(l_max - g_max)
            g_lse_sum += l_sum * scale
            a = partial_out_v[(0, 0, 0, None), (T_idx, s, head_idx, tidx)].load()
            acc = acc + scale * a

        if tidx == 0:
            lse[T_idx, head_idx] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)

        output_v[(0, 0, None), (T_idx, head_idx, tidx)].store((acc / g_lse_sum).to(cutlass.BFloat16))


def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_hybrid():
    T = cute.sym_int()
    q_nope         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K), (1, 0), 4)
    sm_scale       = SM_SCALE
    partial_out    = _fake(cute.Float32,  (LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, NUM_HEADS), (1, 0), 4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    hybrid = Dsa()
    compiled = cute.compile(
        hybrid,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )
    return hybrid, compiled


_hybrid, _compiled = compile_hybrid()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
              _hybrid.partial_out, _hybrid.partial_lse, output, lse)
