import torch
from typing import Tuple
import math
import json

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from cutlass.cute.testing import benchmark, JitArguments

# ── Profiler constants ──────────────────────────────────────────────
PROBE_HEADER = 1
PROBE_ENTRY  = 4

NUM_PROBE_ROLES = 2
TMA_ROLE = 0
MMA_ROLE = 1
ROLE_NAMES = {TMA_ROLE: "TMA", MMA_ROLE: "MMA"}

TAGS = {
    "tma_load":     0,
    "wgmma":         2,
    "tma_store":     4,
    "frag_to_smem": 6,
}
TAG_NAMES = {v: k for k, v in TAGS.items()}
TAG_LIST = ["tma_load", "", "wgmma", "", "tma_store", "", "frag_to_smem", ""]


# ── Inline PTX helpers ──────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(
        T.i64(), [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(
        T.i32(), [],
        "mov.u32 $0, %smid;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


# @dsl_user_op
# def cp_async_bulk_wait_group_read_0(*, loc=None, ip=None) -> cutlass.Int32:
#     t = llvm.inline_asm(
#         T.i32(), [],
#         "cp.async.bulk.wait_group.read 0;\n\tmov.u32 $0, 0;",
#         "=r",
#         has_side_effects=True,
#         is_align_stack=False,
#         asm_dialect=llvm.AsmDialect.AD_ATT,
#         loc=loc, ip=ip,
#     )
#     return cutlass.Int32(t)


# ── Profiler helpers (used inside kernel) ───────────────────────────

def range_start(probe, row, cnt, sm_val, tag_val):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()


def range_stop(probe, row, cnt):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64() - probe[row, off + 2]
    return cnt + cutlass.Int32(1)


def range_finalize(probe, row, cnt):
    probe[row, 0] = cutlass.Int64(cnt)


# ── Host-side profiler analysis and trace output ────────────────────

def dump_probe(probe: torch.Tensor, num_blocks: int,
                  out_path: str = "cute_warp_spec_trace.json"):
    probe_cpu = probe.cpu().contiguous().tolist()
    total_rows = num_blocks * NUM_PROBE_ROLES

    # ── Per-block detail ──
    for bid in range(min(num_blocks, 4)):
        for role in range(NUM_PROBE_ROLES):
            row_idx = bid * NUM_PROBE_ROLES + role
            data = probe_cpu[row_idx]
            cnt = int(data[0])
            print(f"\n--- Block {bid}, {ROLE_NAMES[role]} warp: {cnt} entries ---")
            for i in range(cnt):
                off = PROBE_HEADER + i * PROBE_ENTRY
                sm_id, tag = int(data[off]), int(data[off + 1])
                start, dur = int(data[off + 2]), int(data[off + 3])
                print(f"  sm={sm_id} {TAG_NAMES.get(tag, f'tag_{tag}'):20s} "
                      f"start={start} dur={dur} ns")

    # ── Chrome trace JSON ──
    events, global_base, sm_seen = [], None, set()
    for row_idx in range(total_rows):
        for i in range(int(probe_cpu[row_idx][0])):
            s = int(probe_cpu[row_idx][PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (global_base is None or s < global_base):
                global_base = s
    global_base = global_base or 0

    for row_idx in range(total_rows):
        data = probe_cpu[row_idx]
        cnt = int(data[0])
        if cnt == 0:
            continue
        sm_id = int(data[PROBE_HEADER])
        role = row_idx % NUM_PROBE_ROLES
        if (sm_id, role) in sm_seen:
            continue
        sm_seen.add((sm_id, role))
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag, start, dur = int(data[off+1]), int(data[off+2]), int(data[off+3])
            if start == 0 and dur == 0:
                continue
            events.append(dict(
                name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - global_base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id, tid=role))

    with open(out_path, "w") as f:
        json.dump({"traceEvents": events}, f)
    num_sms = len({e["pid"] for e in events})
    print(f"Trace: {len(events)} events from {num_sms} SMs → {out_path}")
    print("Open with chrome://tracing or https://ui.perfetto.dev")


# ── Kernel class ────────────────────────────────────────────────────

class Gemm_TC:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 64),
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.BK = self.tile_shape_mnk

        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.BM > 64 and self.BN > 64
            else (1, 1, 1)
        )

        self.warp_size = cute.arch.WARP_SIZE
        self.warp_group_size = 128

        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_cta = (
            self.mma_warp_groups * self.warp_group_size + self.warp_size
        )

        assert self.BM % 64 == 0
        assert self.BN % 64 == 0

        self.num_stages = 1
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)

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
            op, tensor, smem_layout, smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        probe: cute.Tensor,
        use_measure: cutlass.Constexpr,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype,
            num_stages=self.num_stages,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype,
            num_stages=self.num_stages,
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged,
            (self.BM, self.BK), 1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b, self.b_smem_layout_staged,
            (self.BN, self.BK), 1,
        )

        c_smem_layout = cute.make_layout(
            (self.BM, self.BN), stride=(self.BN, 1),
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c, c_smem_layout,
            (self.BM, self.BN),
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
                    self.c_dtype, self.BM * self.BN
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
            tiler_mn=(64, self.BN),
        )

        grid_dim = *cute.ceil_div(c.shape, (self.BM, self.BN)), 1
        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            self.tiled_mma,
            cta_layout_mnk,
            c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            probe,
            use_measure,
        ).launch(
            grid=grid_dim,
            block=(self.threads_per_cta, 1, 1),
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
        probe: cute.Tensor,
        use_measure: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        is_mma_warp = tidx < self.mma_warp_groups * self.warp_group_size
        is_tma_warp = warp_idx // 4 == self.mma_warp_groups
        tma_leader = self.mma_warp_groups * self.warp_group_size

        # ── Profiler row setup ──
        grid_x = cute.arch.grid_dim()[0]
        bid = bidy * grid_x + bidx
        sm = smid_u32()

        tma_row = bid * NUM_PROBE_ROLES + TMA_ROLE
        mma_row = bid * NUM_PROBE_ROLES + MMA_ROLE
        tma_cnt = cutlass.Int32(0)
        mma_cnt = cutlass.Int32(0)

        # ── TMA descriptor prefetch ──
        if is_tma_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ── Shared memory ──
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem_alloc = cutlass.utils.SmemAllocator()
        storage = smem_alloc.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        c_smem_layout = cute.make_layout(
            (self.BM, self.BN), stride=(self.BN, 1)
        )
        sC = storage.sC.get_tensor(c_smem_layout)

        # ── Tile partitioning ──
        tile_coord_mnk = (bidx, bidy, None)

        gA = cute.local_tile(
            input=mA_tma_tensor, tiler=self.tile_shape_mnk,
            coord=tile_coord_mnk, proj=(1, None, 1),
        )
        gB = cute.local_tile(
            input=mB_tma_tensor, tiler=self.tile_shape_mnk,
            coord=tile_coord_mnk, proj=(None, 1, 1),
        )
        gC = cute.local_tile(
            input=mC, tiler=self.tile_shape_mnk,
            coord=tile_coord_mnk, proj=(1, 1, None),
        )

        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        a_cta_crd = cluster_coord_mnk[1]
        sA_part = cute.group_modes(sA, 0, 2)
        gA_part = cute.group_modes(gA, 0, 2)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, a_cta_crd, a_cta_layout, sA_part, gA_part,
        )

        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )
        b_cta_crd = cluster_coord_mnk[0]
        sB_part = cute.group_modes(sB, 0, 2)
        gB_part = cute.group_modes(gB, 0, 2)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, b_cta_crd, b_cta_layout, sB_part, gB_part,
        )

        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        k_tile_cnt = mA_tma_tensor.shape[1] // self.BK
        num_k_blocks = self.BK // 16

        # ── MMA partitioning ──
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.warp_group_size
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.warp_group_size
        )
        thr_mma = tiled_mma.get_slice(
            warp_group_thread_layout(warp_group_idx)
        )

        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        acc_shape = tCgC.shape
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        if is_mma_warp:
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        # ════════════════════════════════════════════════════════
        # MAINLOOP
        # ════════════════════════════════════════════════════════
        for kidx in range(k_tile_cnt):
            if tidx == tma_leader:
                cute.arch.mbarrier_init(mbar_ptr, cnt=1)
                cute.arch.mbarrier_init_fence()

            cute.arch.sync_threads()

            # ── TMA LOAD ──
            if is_tma_warp:
                if tidx == tma_leader:
                    cute.arch.mbarrier_expect_tx(mbar_ptr, tma_transaction_bytes)
                    cute.arch.mbarrier_arrive(mbar_ptr)

                    if cutlass.const_expr(use_measure):
                        range_start(probe, tma_row, tma_cnt, sm, TAGS["tma_load"])

                cute.copy(
                    tma_atom_a, tAgA[None, kidx], tAsA[None, 0],
                    tma_bar_ptr=mbar_ptr,
                )
                cute.copy(
                    tma_atom_b, tBgB[None, kidx], tBsB[None, 0],
                    tma_bar_ptr=mbar_ptr,
                )

            cute.arch.mbarrier_wait(mbar_ptr, 0)

            if is_tma_warp:
                if tidx == tma_leader:
                    if cutlass.const_expr(use_measure):
                        tma_cnt = range_stop(probe, tma_row, tma_cnt)

            # ── WGMMA COMPUTE ──
            if is_mma_warp:
                if cutlass.const_expr(use_measure):
                    if tidx == 0:
                        range_start(probe, mma_row, mma_cnt, sm, TAGS["wgmma"])

                cute.nvgpu.warpgroup.fence()

                for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_block_coord = (None, None, k_block_idx, 0)
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        accumulators,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                if cutlass.const_expr(use_measure):
                    if tidx == 0:
                        mma_cnt = range_stop(probe, mma_row, mma_cnt)

        # ════════════════════════════════════════════════════════
        # EPILOGUE: accumulators → shared memory
        # ════════════════════════════════════════════════════════
        if is_mma_warp:
            if cutlass.const_expr(use_measure):
                if tidx == 0:
                    range_start(probe, mma_row, mma_cnt, sm, TAGS["frag_to_smem"])

            tv_layout_C_tiled = tiled_mma.tv_layout_C_tiled
            for reg_idx in range(cute.size(accumulators)):
                coord = cute.idx2crd(
                    (tidx, reg_idx), tv_layout_C_tiled.shape
                )
                mn_local_tile_flat = cute.crd2idx(
                    coord, tv_layout_C_tiled
                )
                m_local, n_local = cute.idx2crd(
                    mn_local_tile_flat, c_smem_layout.shape
                )
                sC[m_local, n_local] = cutlass.Float16(accumulators[reg_idx])

            if cutlass.const_expr(use_measure):
                if tidx == 0:
                    mma_cnt = range_stop(probe, mma_row, mma_cnt)

        cute.arch.sync_threads()
        cute.arch.fence_proxy("async.shared", space="cta")

        # ════════════════════════════════════════════════════════
        # TMA STORE
        # ════════════════════════════════════════════════════════
        gC_tma = cute.local_tile(
            input=mC_tma_tensor, tiler=self.tile_shape_mnk,
            coord=tile_coord_mnk, proj=(1, 1, None),
        )
        sC_part = cute.group_modes(sC, 0, 2)
        gC_part = cute.group_modes(gC_tma, 0, 2)
        tCsC, tCgC_store = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1), sC_part, gC_part,
        )

        if is_tma_warp:
            if cutlass.const_expr(use_measure):
                if tidx == tma_leader:
                    range_start(probe, tma_row, tma_cnt, sm,
                               TAGS["tma_store"])

            cute.copy(tma_atom_c, tCsC, tCgC_store)
            # _ = cp_async_bulk_wait_group_read_0()

            if cutlass.const_expr(use_measure):
                if tidx == tma_leader:
                    tma_cnt = range_stop(probe, tma_row, tma_cnt)

        # ── Finalize profiler ──
        if cutlass.const_expr(use_measure):
            if tidx == tma_leader:
                range_finalize(probe, tma_row, tma_cnt)
            if tidx == 0:
                range_finalize(probe, mma_row, mma_cnt)


# ── Main ────────────────────────────────────────────────────────────

def main():
    M, N, K = 128, 128, 256
    tile_shape = (128, 128, 64)
    bM, bN, bK = tile_shape

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # Initialize Probe
    num_blocks = (N // bN) * (M // bM)
    max_entries_per_role = (K // bK) + 1
    probe_cols = PROBE_HEADER + max_entries_per_role * PROBE_ENTRY
    probe = torch.zeros((num_blocks * NUM_PROBE_ROLES, probe_cols), device="cuda", dtype=torch.int64)

    A_, B_, C_, probe_ = [from_dlpack(t, assumed_align=16) for t in (A, B, C, probe)]
    gemm = Gemm_TC(cta_tiler=tile_shape)

    compiled_measure = cute.compile(gemm, A_, B_, C_, probe_, True)
    compiled_clean   = cute.compile(gemm, A_, B_, C_, probe_, False)

    compiled_measure(A_, B_, C_, probe_)
    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")

    dump_probe(probe, num_blocks)

    time = benchmark(compiled_clean, kernel_arguments=JitArguments(A_, B_, C_, probe_))
    time_probe = benchmark(compiled_measure, kernel_arguments=JitArguments(A_, B_, C_, probe_))
    
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")
    print(f"DURATION_PROBE: {time_probe:>5.4f} µs\nOVERHEAD: {((time_probe - time) / time) * 100:.2f}%")

if __name__ == "__main__":
    main()