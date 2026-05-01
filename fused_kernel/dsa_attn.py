"""
kernel5cl_1024: kernel5cl extended for arbitrary T (up to 1024).

T < 3: fused_kernel — grid [T, 16, 1], 1024 threads (unchanged)
T ≥ 3: compute_kernel (XOR-persistent + cp.async + FastGEMV score + PDL)
         — grid [NUM_HEADS, NUM_SPLITS, 1], 1024 threads
       + reduce_kernel (vectorized tensorSSA + PDL)
         — grid [T_MAX, NUM_HEADS, 1], 256 threads
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
TOP_K = 2048
NUM_PAGES = 8462
PAGE_SIZE = 64
T_MAX = 8
MAX_REQ_CONCURR = 1024  # max concurrent requests; partial buffers pre-allocated
NUM_SPLITS = 8
DIM_SPLIT = (TOP_K + NUM_SPLITS - 1) // NUM_SPLITS  # 256
LN2 = 0.6931471805599453
SM_SCALE: cutlass.Constexpr = 0.1352337788608801


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class HybridDSA():
    def __init__(self):
        self.num_heads = NUM_HEADS
        self.head_dim_ckv = HEAD_DIM_CKV
        self.head_dim_kpe = HEAD_DIM_KPE
        self.top_k = TOP_K
        self.num_pages = NUM_PAGES
        self.page_size = PAGE_SIZE
        self.t_max = T_MAX
        self.max_req_concurr = MAX_REQ_CONCURR
        self.num_splits = NUM_SPLITS
        self.dim_split = DIM_SPLIT

        # ── Fused kernel (A) constants ────────────────────────────────────────
        self.fused_threads = 1024
        self.fused_warps = self.fused_threads // 32   # 32
        self.dims_per_lane = self.head_dim_ckv // 32  # 16
        self.fused_num_vec = 8
        self.fused_iters   = self.dims_per_lane // self.fused_num_vec  # 2

        # ── Compute kernel (B) constants ──────────────────────────────────────
        self.compute_threads = 1024
        self.compute_warps = self.compute_threads // 32  # 32
        self.vec_size_ckv = 8
        self.vec_size_kpe = 2
        self.vec_size_out = 16
        self.iters_per_lane_ckv = self.head_dim_ckv // (32 * self.vec_size_ckv)  # 2
        self.sparse_thr_per_T = 128
        self.num_warps_per_T = self.sparse_thr_per_T // 32  # 4
        self.vec_sparse = 4
        self.vec_q = 8
        self.top_k_chunks = self.top_k // self.vec_sparse   # 512
        self.q_nope_chunks = self.head_dim_ckv // self.vec_q # 64
        self.q_pe_chunks = self.head_dim_kpe // self.vec_q   # 8

        # ── FastGEMV score constants ──────────────────────────────────────────
        self.rows_per_warp = 4
        self.rows_per_round_score = self.compute_warps * self.rows_per_warp  # 128

        # ── Reduce kernel (C) constants ───────────────────────────────────────
        self.reduce_threads = 256
        self.reduce_warps = self.reduce_threads // 32  # 8
        self.vec_reduce = 2

        # ── Workspace: allocated once at MAX_REQ_CONCURR size ─────────────────
        self.partial_out = torch.empty(
            MAX_REQ_CONCURR, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV,
            dtype=torch.float32, device="cuda")
        self.partial_lse = torch.empty(
            MAX_REQ_CONCURR, NUM_HEADS, NUM_SPLITS, 2,
            dtype=torch.float32, device="cuda")

    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)

    # ══════════════════════════════════════════════════════════════════════════
    # Dispatcher (__call__) — branches on T
    # ══════════════════════════════════════════════════════════════════════════

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
        stream):

        T, num_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[2]

        N: cutlass.Constexpr = self.num_pages * self.page_size
        ckv_flat = cute.make_tensor(
            ckv_cache.iterator,
            cute.make_layout((N, self.head_dim_ckv), stride=(self.head_dim_ckv, 1)))
        kpe_flat = cute.make_tensor(
            kpe_cache.iterator,
            cute.make_layout((N, self.head_dim_kpe), stride=(self.head_dim_kpe, 1)))

        if T < 3:
            self.fused_kernel(
                q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale, output, lse
            ).launch(grid=[T, num_heads, 1], block=[self.fused_threads, 1, 1], stream=stream)
        else:
            self.compute_kernel(
                q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
                partial_out, partial_lse, output, lse,
            ).launch(grid=[self.num_heads, self.num_splits, 1],
                     block=[self.compute_threads, 1, 1], stream=stream, use_pdl=True)

            # Fixed grid [T_MAX, num_heads, 1] — reduce kernel loops over groups
            self.reduce_kernel(
                sparse_indices, partial_out, partial_lse, output, lse,
            ).launch(grid=[self.t_max, self.num_heads, 1],
                     block=[self.reduce_threads, 1, 1], stream=stream, use_pdl=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Kernel A: single-block fused (T < 3) — unchanged from kernel5cl
    # ══════════════════════════════════════════════════════════════════════════

    @cute.kernel
    def fused_kernel(
        self,
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_cache:      cute.Tensor,
        kpe_cache:      cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        output:         cute.Tensor,
        lse:            cute.Tensor):

        T, num_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = kpe_cache.shape[1]
        top_k_len    = 2048
        dims_per_lane: cutlass.Constexpr = self.dims_per_lane
        num_vec: cutlass.Constexpr = self.fused_num_vec
        iters_per_lane: cutlass.Constexpr = self.fused_iters

        bidx, bidy, _ = cute.arch.block_idx()
        num_threads: cutlass.Constexpr = self.fused_threads
        num_warps:   cutlass.Constexpr = self.fused_warps
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        wsize    = cute.arch.WARP_SIZE

        allocator = cutlass.utils.SmemAllocator()
        smem_logits  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((top_k_len,),    stride=(1,)), 16, None)
        smem_sparse  = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((top_k_len,),    stride=(1,)),  4, None)
        smem_red_i32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),           stride=(1,)),  4, None)
        smem_red_f32 = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),           stride=(1,)), 16, None)
        smem_q_nope  = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
        smem_q_pe    = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
        smem_partial = allocator.allocate_tensor(cutlass.Float32,
            cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)

        # ── Load phase ────────────────────────────────────────────────────────
        partial_cnt_valid = 0
        for i in range(tidx, top_k_len, num_threads):
            idx = sparse_indices[bidx, i]
            smem_sparse[i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt_valid += 1

        for i in range(tidx, head_dim_ckv, num_threads):
            smem_q_nope[i] = q_nope[bidx, bidy, i]
        for i in range(tidx, head_dim_kpe, num_threads):
            smem_q_pe[i] = q_pe[bidx, bidy, i]

        sum_valid = warp_reduce(partial_cnt_valid, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = sum_valid
        cute.arch.sync_threads()

        if warp_idx == 0:
            val = smem_red_i32[lane_idx]
            sum_valid = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_i32[0] = sum_valid
        cute.arch.sync_threads()

        valid_count = smem_red_i32[0]
        num_rounds  = (valid_count + num_warps - 1) // num_warps

        # ── Score phase: LDG.128 loads + fp32 scalar multiply ─────────────────
        q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec,))

        for round_idx in range(num_rounds):
            sparse_idx = round_idx * num_warps + warp_idx
            if sparse_idx < valid_count:
                cur_idx = smem_sparse[sparse_idx]

                ckv_row = ckv_cache[cur_idx, None]
                ckv_z   = cute.zipped_divide(ckv_row, (num_vec,))

                sum_partial = cutlass.Float32(0)
                for it in range(iters_per_lane):
                    group  = it * wsize + lane_idx
                    q_frag = q_nope_z[(None, (group,))].load()
                    K_frag = ckv_z[(None, (group,))].load()
                    for v in range(num_vec):
                        sum_partial += cutlass.Float32(q_frag[v]) * cutlass.Float32(K_frag[v])

                for k_idx in range(head_dim_kpe // wsize):
                    q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                    kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                    sum_partial += q_p * kv

                s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_logits[sparse_idx] = s * sm_scale

        cute.arch.sync_threads()

        # ── Softmax pass 1: block-wide max ────────────────────────────────────
        partial_max = -cutlass.Float32(math.inf)
        for idx in range(tidx, valid_count, num_threads):
            v = smem_logits[idx]
            if v > partial_max:
                partial_max = v

        max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
        if lane_idx == 0:
            smem_red_f32[warp_idx] = max_val
        cute.arch.sync_threads()
        if warp_idx == 0:
            val = smem_red_f32[lane_idx]
            max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
            smem_red_f32[0] = max_val
        cute.arch.sync_threads()

        row_max = smem_red_f32[0]

        # ── Softmax pass 2: exp + sum + WRITE BACK ────────────────────────────
        partial_sum = cutlass.Float32(0)
        for idx in range(tidx, valid_count, num_threads):
            e = cute.math.exp(smem_logits[idx] - row_max)
            smem_logits[idx] = e
            partial_sum += e

        sum_val = warp_reduce(partial_sum, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_f32[warp_idx] = sum_val
        cute.arch.sync_threads()
        if warp_idx == 0:
            val = smem_red_f32[lane_idx]
            sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_f32[0] = sum_val
        cute.arch.sync_threads()

        row_sum = smem_red_f32[0]

        if tidx == 0:
            lse[bidx, bidy] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)

        # ── Output phase: vectorized LDG.128 reads ───────────────────────────
        out_regs = cute.make_rmem_tensor(
            cute.make_layout((dims_per_lane,), stride=(1,)),
            cutlass.Float32,
        )
        for k in range(dims_per_lane):
            out_regs[k] = cutlass.Float32(0)

        for round_idx in range(num_rounds):
            j = round_idx * num_warps + warp_idx
            if j < valid_count:
                kv_idx = smem_sparse[j]
                weight = smem_logits[j] / row_sum

                V_row = ckv_cache[kv_idx, None]
                V_z   = cute.zipped_divide(V_row, (num_vec,))

                for it in range(iters_per_lane):
                    group = it * wsize + lane_idx
                    frag  = V_z[(None, (group,))].load()
                    for v in range(num_vec):
                        out_regs[it * num_vec + v] += weight * cutlass.Float32(frag[v])

        for it in range(iters_per_lane):
            for v in range(num_vec):
                smem_partial[warp_idx, (it * wsize + lane_idx) * num_vec + v] = out_regs[it * num_vec + v]

        cute.arch.sync_threads()

        for i in range(tidx, head_dim_ckv, num_threads):
            acc = cutlass.Float32(0)
            for w in range(num_warps):
                acc += smem_partial[w, i]
            output[bidx, bidy, i] = cutlass.BFloat16(acc)

    # ══════════════════════════════════════════════════════════════════════════
    # Kernel B: XOR-persistent compute (T ≥ 3) — FastGEMV 4-row score
    #   Grid: [16, 8, 1] × 1024 threads
    #   Outer group loop (groups of T_MAX=8) for arbitrary T
    # ══════════════════════════════════════════════════════════════════════════

    @cute.kernel
    def compute_kernel(
        self,
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor):

        T, _, _ = q_nope.shape
        head_dim_ckv:   cutlass.Constexpr = self.head_dim_ckv
        head_dim_kpe:   cutlass.Constexpr = self.head_dim_kpe
        top_k_len:      cutlass.Constexpr = self.top_k
        dim_split:      cutlass.Constexpr = self.dim_split
        num_splits:     cutlass.Constexpr = self.num_splits
        num_threads:    cutlass.Constexpr = self.compute_threads
        num_warps:      cutlass.Constexpr = self.compute_warps
        vec_size_ckv:   cutlass.Constexpr = self.vec_size_ckv
        vec_size_kpe:   cutlass.Constexpr = self.vec_size_kpe
        vec_size_out:   cutlass.Constexpr = self.vec_size_out
        iters_per_lane_ckv: cutlass.Constexpr = self.iters_per_lane_ckv
        sparse_thr_per_T:   cutlass.Constexpr = self.sparse_thr_per_T
        num_warps_per_T:    cutlass.Constexpr = self.num_warps_per_T
        t_max:          cutlass.Constexpr = self.t_max
        vec_sparse:     cutlass.Constexpr = self.vec_sparse
        vec_q:          cutlass.Constexpr = self.vec_q
        top_k_chunks:   cutlass.Constexpr = self.top_k_chunks
        q_nope_chunks:  cutlass.Constexpr = self.q_nope_chunks
        q_pe_chunks:    cutlass.Constexpr = self.q_pe_chunks
        rows_per_warp:  cutlass.Constexpr = self.rows_per_warp
        rows_per_round_score: cutlass.Constexpr = self.rows_per_round_score

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        wsize = cute.arch.WARP_SIZE

        head_idx = bidx
        split_idx_old = bidy

        # ── SMEM allocation ──────────────────────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()
        smem_sparse      = self._smem(alloc, cutlass.Int32,    (t_max, top_k_len),        (top_k_len, 1),     4)
        smem_num_valid   = self._smem(alloc, cutlass.Int32,    (t_max,),                   (1,),               4)
        smem_logits      = self._smem(alloc, cutlass.Float32,  (dim_split,),               (1,),              16)
        smem_red_i32     = self._smem(alloc, cutlass.Int32,    (t_max, 32),                (32, 1),            4)
        smem_max_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                      (1,),              16)
        smem_sum_red_f32 = self._smem(alloc, cutlass.Float32,  (32,),                      (1,),              16)
        smem_q_nope      = self._smem(alloc, cutlass.BFloat16, (t_max, head_dim_ckv),      (head_dim_ckv, 1), 16)
        smem_q_pe        = self._smem(alloc, cutlass.BFloat16, (t_max, head_dim_kpe),      (head_dim_kpe, 1), 16)
        smem_partial     = self._smem(alloc, cutlass.Float32,  (num_warps, head_dim_ckv),  (head_dim_ckv, 1), 16)
        smem_out         = self._smem(alloc, cutlass.Float32,  (head_dim_ckv,),            (1,),              16)

        # ── Thread-group indices ─────────────────────────────────────────────
        wg_per_T_idx   = tidx // sparse_thr_per_T
        thr_idx_per_T  = tidx % sparse_thr_per_T
        lane_idx_per_T = thr_idx_per_T % wsize
        warp_per_T_idx = warp_idx % num_warps_per_T

        # ── cp.async copy atom ───────────────────────────────────────────────
        copy_atom_q = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)

        q_nope_vec      = cute.zipped_divide(q_nope,      (1, 1, vec_q))
        smem_q_nope_vec = cute.zipped_divide(smem_q_nope, (1, vec_q))
        q_pe_vec        = cute.zipped_divide(q_pe,        (1, 1, vec_q))
        smem_q_pe_vec   = cute.zipped_divide(smem_q_pe,   (1, vec_q))
        si_vec          = cute.zipped_divide(sparse_indices, (1, vec_sparse))

        # ── Vectorized views (independent of T, computed once) ───────────────
        smem_q_nope_ = cute.zipped_divide(smem_q_nope, (1, vec_size_ckv))
        ckv_flat_    = cute.zipped_divide(ckv_flat,     (1, vec_size_ckv))
        kpe_flat_    = cute.zipped_divide(kpe_flat,     (1, vec_size_kpe))
        smem_q_pe_   = cute.zipped_divide(smem_q_pe,   (1, vec_size_kpe))

        # ── Outer group loop: process up to T_MAX tokens per group ───────────
        num_groups = (T + t_max - 1) // t_max
        for group_idx in range(num_groups):
            t_group_start = group_idx * t_max
            group_size = T - t_group_start
            if group_size > t_max:
                group_size = t_max
            T_global_wg = t_group_start + wg_per_T_idx

            # ══════════════════════════════════════════════════════════════
            # Phase 1: cp.async fire q_nope + q_pe (non-blocking)
            # ══════════════════════════════════════════════════════════════
            if T_global_wg < T:
                for chunk in range(thr_idx_per_T, q_nope_chunks, sparse_thr_per_T):
                    cute.copy(copy_atom_q,
                        q_nope_vec[(0, 0, None), (T_global_wg, head_idx, chunk)],
                        smem_q_nope_vec[(0, None), (wg_per_T_idx, chunk)])
                for chunk in range(thr_idx_per_T, q_pe_chunks, sparse_thr_per_T):
                    cute.copy(copy_atom_q,
                        q_pe_vec[(0, 0, None), (T_global_wg, head_idx, chunk)],
                        smem_q_pe_vec[(0, None), (wg_per_T_idx, chunk)])

            cute.arch.cp_async_commit_group()

            # ══════════════════════════════════════════════════════════════
            # Phase 2: sparse_load — vec4 + early-exit + valid count
            # ══════════════════════════════════════════════════════════════
            partial_cnt = 0
            if T_global_wg < T:
                chunk = cutlass.Int32(thr_idx_per_T)
                while chunk < cutlass.Int32(top_k_chunks):
                    vec = si_vec[(0, None), (T_global_wg, chunk)].load()
                    v0 = vec[0]
                    for v in range(vec_sparse):
                        smem_sparse[wg_per_T_idx, chunk * vec_sparse + v] = vec[v]
                        if vec[v] >= cutlass.Int32(0):
                            partial_cnt += 1
                    if v0 < cutlass.Int32(0):
                        chunk = cutlass.Int32(top_k_chunks)
                    else:
                        chunk = chunk + cutlass.Int32(sparse_thr_per_T)

                cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
                if lane_idx_per_T == 0:
                    smem_red_i32[wg_per_T_idx, warp_per_T_idx] = cnt_sum

                cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                                  number_of_threads=sparse_thr_per_T)

                if warp_per_T_idx == 0:
                    val     = smem_red_i32[wg_per_T_idx, lane_idx_per_T]
                    cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps_per_T)
                    smem_red_i32[wg_per_T_idx, 0] = cnt_sum

                cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                                  number_of_threads=sparse_thr_per_T)

                smem_num_valid[wg_per_T_idx] = smem_red_i32[wg_per_T_idx, 0]

            # ══════════════════════════════════════════════════════════════
            # Phase 3: cp_async_wait + sync
            # ══════════════════════════════════════════════════════════════
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ── Clamp negative sparse indices → 0 for safe OOB-free FastGEMV
            for t_fix in range(t_max):
                if t_fix < group_size:
                    for i in range(tidx, top_k_len, num_threads):
                        if smem_sparse[t_fix, i] < cutlass.Int32(0):
                            smem_sparse[t_fix, i] = cutlass.Int32(0)
            cute.arch.sync_threads()

            # ── PDL: signal dependents after group 0 prologue ────────────
            if group_idx == 0:
                cute.arch.griddepcontrol_launch_dependents()

            # ── Inner T-loop with XOR swizzle (within current group) ─────
            for T_local in range(group_size):
                T_global = t_group_start + T_local
                split_idx_new = T_local ^ split_idx_old

                num_valid_T = smem_num_valid[T_local]
                split_start = split_idx_new * dim_split
                is_OOB = split_start >= num_valid_T

                if not is_OOB:
                    local_valid = min(num_valid_T - split_start, dim_split)
                    num_rounds = (local_valid + num_warps - 1) // num_warps

                    # ── Score (FastGEMV: 4-row interleaved per warp) ─────
                    num_rounds_score = (local_valid + rows_per_round_score - 1) // rows_per_round_score

                    for round_idx in range(num_rounds_score):
                        base_sparse = round_idx * rows_per_round_score + warp_idx * rows_per_warp

                        cur_idx0 = smem_sparse[T_local, split_start + base_sparse + 0]
                        cur_idx1 = smem_sparse[T_local, split_start + base_sparse + 1]
                        cur_idx2 = smem_sparse[T_local, split_start + base_sparse + 2]
                        cur_idx3 = smem_sparse[T_local, split_start + base_sparse + 3]

                        ckv_row0 = ckv_flat_[(0, None), (cur_idx0, None)]
                        ckv_row1 = ckv_flat_[(0, None), (cur_idx1, None)]
                        ckv_row2 = ckv_flat_[(0, None), (cur_idx2, None)]
                        ckv_row3 = ckv_flat_[(0, None), (cur_idx3, None)]

                        kpe_row0 = kpe_flat_[(0, None), (cur_idx0, None)]
                        kpe_row1 = kpe_flat_[(0, None), (cur_idx1, None)]
                        kpe_row2 = kpe_flat_[(0, None), (cur_idx2, None)]
                        kpe_row3 = kpe_flat_[(0, None), (cur_idx3, None)]

                        sums = cute.make_rmem_tensor(
                            cute.make_layout((rows_per_warp,), stride=(1,)),
                            cutlass.Float32,
                        )
                        for r in range(rows_per_warp):
                            sums[r] = cutlass.Float32(0)

                        # CKV dot products
                        for it in range(iters_per_lane_ckv):
                            rest_idx = it * wsize + lane_idx
                            qn_frag = smem_q_nope_[(0, None), (T_local, rest_idx)].load()

                            ckv_f0 = ckv_row0[None, rest_idx].load()
                            ckv_f1 = ckv_row1[None, rest_idx].load()
                            ckv_f2 = ckv_row2[None, rest_idx].load()
                            ckv_f3 = ckv_row3[None, rest_idx].load()

                            for v in range(vec_size_ckv):
                                qv = cutlass.Float32(qn_frag[v])
                                sums[0] = sums[0] + qv * cutlass.Float32(ckv_f0[v])
                                sums[1] = sums[1] + qv * cutlass.Float32(ckv_f1[v])
                                sums[2] = sums[2] + qv * cutlass.Float32(ckv_f2[v])
                                sums[3] = sums[3] + qv * cutlass.Float32(ckv_f3[v])

                        # KPE dot products
                        qp_frag = smem_q_pe_[(0, None), (T_local, lane_idx)].load()
                        kpe_f0 = kpe_row0[None, lane_idx].load()
                        kpe_f1 = kpe_row1[None, lane_idx].load()
                        kpe_f2 = kpe_row2[None, lane_idx].load()
                        kpe_f3 = kpe_row3[None, lane_idx].load()
                        for v in range(vec_size_kpe):
                            qv = cutlass.Float32(qp_frag[v])
                            sums[0] = sums[0] + qv * cutlass.Float32(kpe_f0[v])
                            sums[1] = sums[1] + qv * cutlass.Float32(kpe_f1[v])
                            sums[2] = sums[2] + qv * cutlass.Float32(kpe_f2[v])
                            sums[3] = sums[3] + qv * cutlass.Float32(kpe_f3[v])

                        # Batched warp reduction
                        for r in range(rows_per_warp):
                            sums[r] = warp_reduce(sums[r], lambda a, b: a + b, width=32)
                        if lane_idx == 0:
                            for r in range(rows_per_warp):
                                smem_logits[base_sparse + r] = sums[r] * sm_scale

                    cute.arch.sync_threads()

                    # ── Softmax: max ─────────────────────────────────────
                    partial_max = -cutlass.Float32(math.inf)
                    for idx in range(tidx, local_valid, num_threads):
                        v = smem_logits[idx]
                        if v > partial_max:
                            partial_max = v

                    max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
                    if lane_idx == 0:
                        smem_max_red_f32[warp_idx] = max_val
                    cute.arch.sync_threads()
                    if warp_idx == 0:
                        val = smem_max_red_f32[lane_idx]
                        max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
                        smem_max_red_f32[0] = max_val
                    cute.arch.sync_threads()

                    row_max = smem_max_red_f32[0]

                    # ── Softmax: exp + sum ───────────────────────────────
                    local_sum = cutlass.Float32(0)
                    for idx in range(tidx, local_valid, num_threads):
                        e = cute.math.exp(smem_logits[idx] - row_max)
                        smem_logits[idx] = e
                        local_sum += e

                    sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
                    if lane_idx == 0:
                        smem_sum_red_f32[warp_idx] = sum_val
                    cute.arch.sync_threads()
                    if warp_idx == 0:
                        val = smem_sum_red_f32[lane_idx]
                        sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
                        smem_sum_red_f32[0] = sum_val
                    cute.arch.sync_threads()

                    row_sum = smem_sum_red_f32[0]

                    # ── Output ───────────────────────────────────────────
                    out_regs = cute.make_rmem_tensor(cute.make_layout((vec_size_out,), stride=(1,)), cutlass.Float32)
                    for i in range(vec_size_out):
                        out_regs[i] = cutlass.Float32(0)

                    for round_idx in range(num_rounds):
                        sparse_idx = round_idx * num_warps + warp_idx
                        if sparse_idx < local_valid:
                            cur_idx = smem_sparse[T_local, split_start + sparse_idx]
                            ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)]
                            e = smem_logits[sparse_idx]

                            for it in range(iters_per_lane_ckv):
                                rest_idx = it * wsize + lane_idx
                                ckv_vec = ckv_row_[None, rest_idx].load()
                                for i in range(vec_size_ckv):
                                    out_regs[it * vec_size_ckv + i] += e * cutlass.Float32(ckv_vec[i])

                    if warp_idx < local_valid:
                        for it in range(iters_per_lane_ckv):
                            for v in range(vec_size_ckv):
                                smem_partial[warp_idx, (it * wsize + lane_idx) * vec_size_ckv + v] = out_regs[it * vec_size_ckv + v]

                    cute.arch.sync_threads()

                    num_active_warps = local_valid if local_valid < num_warps else num_warps
                    for i in range(tidx, head_dim_ckv, num_threads):
                        acc = cutlass.Float32(0)
                        for w in range(num_active_warps):
                            acc += smem_partial[w, i]
                        smem_out[i] = acc
                    cute.arch.sync_threads()

                    is_single_split_request = num_valid_T < dim_split

                    if is_single_split_request and split_idx_new == 0:
                        for i in range(tidx, head_dim_ckv, num_threads):
                            output[T_global, head_idx, i] = cutlass.BFloat16(smem_out[i] / row_sum)
                        if tidx == 0:
                            lse[T_global, head_idx] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
                    else:
                        for i in range(tidx, head_dim_ckv, num_threads):
                            partial_out[T_global, head_idx, split_idx_new, i] = smem_out[i]
                        if tidx == 0:
                            partial_lse[T_global, head_idx, split_idx_new, 0] = row_max
                            partial_lse[T_global, head_idx, split_idx_new, 1] = row_sum


    # ══════════════════════════════════════════════════════════════════════════
    # Kernel C: Reduce — vectorized tensorSSA + PDL
    #   Grid: [T_MAX, 16, 1] × 256 threads — loops over groups of T_MAX
    #   No sentinel — uses num_valid < dim_split to skip single-split rows
    # ══════════════════════════════════════════════════════════════════════════

    @cute.kernel
    def reduce_kernel(
        self,
        sparse_indices: cute.Tensor,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor):

        T, _ = sparse_indices.shape
        head_dim_ckv:   cutlass.Constexpr = self.head_dim_ckv
        top_k_len:      cutlass.Constexpr = self.top_k
        dim_split:      cutlass.Constexpr = self.dim_split
        num_splits:     cutlass.Constexpr = self.num_splits
        num_threads:    cutlass.Constexpr = self.reduce_threads
        num_warps:      cutlass.Constexpr = self.reduce_warps
        vec_reduce:     cutlass.Constexpr = self.vec_reduce
        t_max:          cutlass.Constexpr = self.t_max

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        block_t_idx = bidx   # 0..T_MAX-1
        head_idx    = bidy

        alloc = cutlass.utils.SmemAllocator()
        smem_red_i32 = self._smem(alloc, cutlass.Int32,   (32,),            (1,),   4)
        smem_max_sum = self._smem(alloc, cutlass.Float32,  (num_splits, 2),  (2, 1), 4)

        # zipped_divide views for vectorized access (computed once)
        partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, vec_reduce))
        output_v      = cute.zipped_divide(output, (1, 1, vec_reduce))

        # ── Outer loop over groups of T_MAX ──────────────────────────────
        num_groups = (T + t_max - 1) // t_max
        for group_idx in range(num_groups):
            T_idx = group_idx * t_max + block_t_idx
            if T_idx < T:
                # ── Count valid for this T_idx ───────────────────────────
                partial_cnt = 0
                for i in range(tidx, top_k_len, num_threads):
                    idx = sparse_indices[T_idx, i]
                    if idx >= cutlass.Int32(0):
                        partial_cnt += 1

                cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_red_i32[warp_idx] = cnt_sum
                cute.arch.sync_threads()

                if warp_idx == 0:
                    val = smem_red_i32[lane_idx]
                    cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps)
                    smem_red_i32[0] = cnt_sum
                cute.arch.sync_threads()

                num_valid = smem_red_i32[0]

                # ── PDL: wait for compute after valid counting ───────────
                cute.arch.griddepcontrol_wait()

                # ── Reduce this (T_idx, head_idx) ────────────────────────
                is_single_split = num_valid < dim_split

                if not is_single_split:
                    num_active_splits = (num_valid + dim_split - 1) // dim_split

                    if tidx < num_active_splits:
                        smem_max_sum[tidx, 0] = partial_lse[T_idx, head_idx, tidx, 0]
                        smem_max_sum[tidx, 1] = partial_lse[T_idx, head_idx, tidx, 1]

                    cute.arch.sync_threads()

                    g_max = -cutlass.Float32(math.inf)
                    for s in range(num_active_splits):
                        local_max = smem_max_sum[s, 0]
                        if local_max > g_max:
                            g_max = local_max

                    g_lse_sum = cutlass.Float32(0)
                    acc_rmem = cute.make_rmem_tensor(cute.make_layout((vec_reduce,), stride=(1,)), cutlass.Float32)
                    acc_rmem[0] = cutlass.Float32(0)
                    acc_rmem[1] = cutlass.Float32(0)
                    acc = acc_rmem.load()

                    for s in range(num_active_splits):
                        l_max = smem_max_sum[s, 0]
                        l_sum = smem_max_sum[s, 1]
                        scale = cute.math.exp(l_max - g_max)
                        g_lse_sum += l_sum * scale

                        a = partial_out_v[(0, 0, 0, None), (T_idx, head_idx, s, tidx)].load()
                        acc = acc + scale * a

                    if tidx == 0:
                        lse[T_idx, head_idx] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)

                    output_v[(0, 0, None), (T_idx, head_idx, tidx)].store((acc / g_lse_sum).to(cutlass.BFloat16))

                cute.arch.sync_threads()


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

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
    partial_out    = _fake(cute.Float32,  (MAX_REQ_CONCURR, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (MAX_REQ_CONCURR, NUM_HEADS, NUM_SPLITS, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, NUM_HEADS), (1, 0), 4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    hybrid = HybridDSA()

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
