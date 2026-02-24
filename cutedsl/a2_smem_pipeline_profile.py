import torch
from typing import Tuple
import math
import json

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.pipeline import PipelineAsync, CooperativeGroup, Agent
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm


# ── Profiler constants ──────────────────────────────────────────────
PROBE_HEADER = 1
PROBE_ENTRY  = 4

NUM_PROBE_ROLES = 2
PRODUCER_ROLE = 0
CONSUMER_ROLE = 1
ROLE_NAMES = {PRODUCER_ROLE: "Producer", CONSUMER_ROLE: "Consumer"}

TAGS = {
    "gmem2smem":  0,
    "mma":        2,
    "reg2gmem":   4,
}
TAG_NAMES = {v: k for k, v in TAGS.items()}


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
               out_path: str = "pipeline_trace.json"):
    probe_cpu = probe.cpu().contiguous().tolist()
    total_rows = num_blocks * NUM_PROBE_ROLES

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
    print(f"\nTrace: {len(events)} events from {num_sms} SMs → {out_path}")
    print("Open with chrome://tracing or https://ui.perfetto.dev")


# ── Kernel class ────────────────────────────────────────────────────

class GemmPipeAsync:
    def __init__(self):
        self.tile_shape_mnk = (8, 4, 16)
        self.BM, self.BN, self.BK = self.tile_shape_mnk
        self.padding = 8
        self.block_size = self.BM * self.BN

        self.num_stages = 2
        self.buffer_align_bytes = 1024
        self.shared_storage = None
        self.num_producer_threads = 32
        self.num_consumer_threads = 32

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
        probe: cute.Tensor,
        use_measure: cutlass.Constexpr,
    ):
        sA_layout_staged = cute.make_layout(
            shape=(self.num_stages, self.BM, self.BK),
            stride=(self.BM * (self.BK + self.padding),
                    self.BK + self.padding, 1),
        )
        sB_layout_staged = cute.make_layout(
            shape=(self.num_stages, self.BN, self.BK),
            stride=(self.BN * (self.BK + self.padding),
                    self.BK + self.padding, 1),
        )

        @cute.struct
        class SharedStorage:
            pipeline_mbarrier_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[mA.element_type,
                                     cute.cosize(sA_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[mB.element_type,
                                     cute.cosize(sB_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        M, N = mC.shape

        self.kernel(
            mA, mB, mC,
            sA_layout_staged, sB_layout_staged,
            probe, use_measure,
        ).launch(
            grid=[N // self.BN, M // self.BM, 1],
            block=[self.num_producer_threads + self.num_consumer_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
        sA_layout_staged: cute.Layout,
        sB_layout_staged: cute.Layout,
        probe: cute.Tensor,
        use_measure: cutlass.Constexpr,
    ):
        BM, BN, BK = self.tile_shape_mnk

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(layout=sA_layout_staged)
        sB = storage.sB.get_tensor(layout=sB_layout_staged)

        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        grid_x = cute.arch.grid_dim()[0]
        bid = bidy * grid_x + bidx
        sm = smid_u32()

        producer_row = bid * NUM_PROBE_ROLES + PRODUCER_ROLE
        consumer_row = bid * NUM_PROBE_ROLES + CONSUMER_ROLE
        producer_cnt = cutlass.Int32(0)
        consumer_cnt = cutlass.Int32(0)

        mainloop_pipeline = PipelineAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread,
                                            self.num_producer_threads),
            consumer_group=CooperativeGroup(Agent.Thread,
                                            self.num_consumer_threads),
            barrier_storage=storage.pipeline_mbarrier_ptr.data_ptr(),
        )

        producer, consumer = mainloop_pipeline.make_participants()

        # ── Producer warp ──
        if warp_idx == 0:
            tid, _, _ = cute.arch.thread_idx()

            for ctak in range(0, gA.shape[1], BK):
                handle = producer.acquire_and_advance()

                if cutlass.const_expr(use_measure):
                    if tid == 0:
                        range_start(probe, producer_row, producer_cnt,
                                    sm, TAGS["gmem2smem"])

                num_loads_A = BM * BK
                for i in range(tid, num_loads_A, self.num_producer_threads):
                    row = i // BK
                    col = i % BK
                    sA[handle.index, row, col] = gA[bidy * BM + row,
                                                     ctak + col]

                num_loads_B = BN * BK
                for i in range(tid, num_loads_B, self.num_producer_threads):
                    row = i // BK
                    col = i % BK
                    sB[handle.index, row, col] = gB[bidx * BN + row,
                                                     ctak + col]

                if cutlass.const_expr(use_measure):
                    if tid == 0:
                        producer_cnt = range_stop(probe, producer_row,
                                                  producer_cnt)

                handle.commit()

            producer.tail()

            if cutlass.const_expr(use_measure):
                if tid == 0:
                    range_finalize(probe, producer_row, producer_cnt)

        # ── Consumer warp ──
        if warp_idx == 1:
            tid, _, _ = cute.arch.thread_idx()
            tid = tid - self.num_producer_threads
            tidx = tid % BM
            tidy = tid // BM

            acc = cute.Float32(0)

            for ctak in range(0, gA.shape[1], BK):
                handle = consumer.wait_and_advance()

                if cutlass.const_expr(use_measure):
                    if tid == 0:
                        range_start(probe, consumer_row, consumer_cnt,
                                    sm, TAGS["mma"])

                for mmak in range(BK):
                    acc += (cute.Float32(sA[handle.index, tidx, mmak])
                            * cute.Float32(sB[handle.index, tidy, mmak]))

                if cutlass.const_expr(use_measure):
                    if tid == 0:
                        consumer_cnt = range_stop(probe, consumer_row,
                                                  consumer_cnt)

                handle.release()

            # ── epilogue ──
            if cutlass.const_expr(use_measure):
                if tid == 0:
                    range_start(probe, consumer_row, consumer_cnt,
                                sm, TAGS["reg2gmem"])

            gC[bidy * BM + tidx, bidx * BN + tidy] = cute.Float16(acc)

            if cutlass.const_expr(use_measure):
                if tid == 0:
                    consumer_cnt = range_stop(probe, consumer_row,
                                              consumer_cnt)
                    range_finalize(probe, consumer_row, consumer_cnt)


# ── Main ────────────────────────────────────────────────────────────

def main():
    M, N, K = 8, 4, 128
    BM, BN, BK = 8, 4, 16

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    num_blocks = (N // BN) * (M // BM)
    k_tiles = K // BK
    max_entries_per_role = k_tiles + 2
    probe_cols = PROBE_HEADER + max_entries_per_role * PROBE_ENTRY
    probe = torch.zeros(
        (num_blocks * NUM_PROBE_ROLES, probe_cols),
        device="cuda", dtype=torch.int64,
    )

    A_, B_, C_, probe_ = [
        from_dlpack(t, assumed_align=16) for t in (A, B, C, probe)
    ]

    gemm = GemmPipeAsync()

    compiled_measure = cute.compile(gemm, A_, B_, C_, probe_, True)
    compiled_clean   = cute.compile(gemm, A_, B_, C_, probe_, False)

    compiled_measure(A_, B_, C_, probe_)

    assert torch.allclose(
        C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1
    ), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")

    dump_probe(probe, num_blocks)

    time = benchmark(compiled_clean,
                     kernel_arguments=JitArguments(A_, B_, C_, probe_))
    time_probe = benchmark(compiled_measure,
                           kernel_arguments=JitArguments(A_, B_, C_, probe_))

    tflops = (2 * M * N * K) / (time * 1e6)
    print(f"\nDURATION:       {time:>8.4f} µs")
    print(f"TFLOPS:         {tflops:>8.4f}")
    print(f"DURATION_PROBE: {time_probe:>8.4f} µs")
    print(f"OVERHEAD:       {((time_probe - time) / time) * 100:.2f}%")


if __name__ == "__main__":
    main()