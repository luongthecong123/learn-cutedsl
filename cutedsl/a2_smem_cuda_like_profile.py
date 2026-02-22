import torch
import json

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from cutlass.cute.testing import benchmark, JitArguments

# ── Profiler constants ──────────────────────────────────────────────
PROBE_HEADER = 1
PROBE_ENTRY  = 4

TAGS = {
    "gmem_to_smem": 0,
    "smem_mma":     2,
    "store":        4,
}
TAG_LIST = ["gmem_to_smem", "", "smem_mma", "", "store"]
TAG_NAMES = {v: k for k, v in TAGS.items()}

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


# ── Profiler helpers ────────────────────────────────────────────────
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


# ── Trace output ────────────────────────────────────────────────────

def dump_probe(probe, num_blocks, out_path="naive_smem_trace.json"):
    cpu_list = probe.cpu().contiguous().tolist()

    # ── Per-block detail ──
    for bid in range(min(num_blocks, 2)):
        cnt = int(cpu_list[bid][0])
        print(f"\n--- Block {bid}: {cnt} entries ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id, tag = int(cpu_list[bid][off]), int(cpu_list[bid][off + 1])
            start, dur = int(cpu_list[bid][off + 2]), int(cpu_list[bid][off + 3])
            print(f"  sm={sm_id} {TAG_NAMES.get(tag, f'tag_{tag}'):20s} "
                  f"start={start} dur={dur} ns")

    # ── Chrome trace JSON (1 block per SM) ──
    events, global_base, sm_seen = [], None, set()
    for row_idx in range(num_blocks):
        for i in range(int(cpu_list[row_idx][0])):
            s = int(cpu_list[row_idx][PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (global_base is None or s < global_base):
                global_base = s
    global_base = global_base or 0

    for row_idx in range(num_blocks):
        data = cpu_list[row_idx]
        cnt = int(data[0])
        if cnt == 0:
            continue
        sm_id = int(data[PROBE_HEADER])
        if sm_id in sm_seen:
            continue
        sm_seen.add(sm_id)
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag, start, dur = int(data[off+1]), int(data[off+2]), int(data[off+3])
            if start == 0 and dur == 0:
                continue
            events.append(dict(
                name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - global_base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id, tid=row_idx))

    with open(out_path, "w") as f:
        json.dump({"traceEvents": events}, f)
    print(f"Trace: {len(events)} events from {len(sm_seen)} SMs → {out_path}")
    print("Open with chrome://tracing or https://ui.perfetto.dev")


# ── Kernel ──────────────────────────────────────────────────────────

@cute.jit
def naive_smem_launcher(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    probe: cute.Tensor,
    use_measure: cutlass.Constexpr,
):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]

    BM, BN, BK = 32, 32, 64
    assert BM == BN

    naive_smem_kernel(mA, mB, mC, M, N, K, probe, use_measure).launch(
        grid=[N // BN, M // BM, 1],
        block=[BM, BN, 1],
    )


@cute.kernel
def naive_smem_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    M: int,
    N: int,
    K: int,
    probe: cute.Tensor,
    use_measure: cutlass.Constexpr,
):
    BM, BN, BK = 32, 32, 64
    PAD = 8

    allocator = cutlass.utils.SmemAllocator()
    layout_sA = cute.make_layout((BM, BK), stride=(BK + PAD, 1))
    layout_sB = cute.make_layout((BN, BK), stride=(BK + PAD, 1))
    sA = allocator.allocate_tensor(cutlass.Float16, layout_sA, 16, None)
    sB = allocator.allocate_tensor(cutlass.Float16, layout_sB, 16, None)

    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    tidx, tidy, _ = cute.arch.thread_idx()
    tid = tidy * bdimx + tidx
    num_threads = bdimx * bdimy

    # if cutlass.const_expr(use_measure):
    grid_x = cute.arch.grid_dim()[0]
    bid = bidy * grid_x + bidx
    sm = smid_u32()
    cnt = cutlass.Int32(0)

    acc = cute.Float32(0)

    for ctak in range(0, K, BK):
        if cutlass.const_expr(use_measure) and tid == 0:
            range_start(probe, bid, cnt, sm, TAGS["gmem_to_smem"])

        num_loads = BM * BK
        for i in range(tid, num_loads, num_threads):
            row = i // BK
            col = i % BK
            sA[row, col] = gA[bidy * BM + row, ctak + col]
            sB[row, col] = gB[bidx * BN + row, ctak + col]

        cute.arch.sync_threads()

        if cutlass.const_expr(use_measure) and tid == 0:
            cnt = range_stop(probe, bid, cnt)
            range_start(probe, bid, cnt, sm, TAGS["smem_mma"])

        for mmak in range(BK):
            acc += cute.Float32(sA[tidy, mmak]) * cute.Float32(sB[tidx, mmak])

        cute.arch.sync_threads()

        if cutlass.const_expr(use_measure) and tid == 0:
            cnt = range_stop(probe, bid, cnt)

    if cutlass.const_expr(use_measure) and tid == 0:
        range_start(probe, bid, cnt, sm, TAGS["store"])

    gC[bidy * bdimy + tidy, bidx * bdimx + tidx] = cute.Float16(acc)

    if cutlass.const_expr(use_measure) and tid == 0:
        cnt = range_stop(probe, bid, cnt)
        range_finalize(probe, bid, cnt)


# ── Main ────────────────────────────────────────────────────────────

def main():
    M, N, K = 32, 32, 256
    BM, BN, BK = 32, 32, 64

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # Initialize Probe
    num_blocks = (N // BN) * (M // BM)
    max_entries = (K // BK) * 2 + 1
    probe = torch.zeros((num_blocks, PROBE_HEADER + max_entries * PROBE_ENTRY), device="cuda", dtype=torch.int64)
    A_, B_, C_, probe_ = [from_dlpack(t, assumed_align=16) for t in (A, B, C, probe)]

    compiled_measure = cute.compile(naive_smem_launcher, A_, B_, C_, probe_, True)
    compiled_clean   = cute.compile(naive_smem_launcher, A_, B_, C_, probe_, False)

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