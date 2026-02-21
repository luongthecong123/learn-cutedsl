# trace_cute_naive_gemm_globaltimer.py

import json
import torch
import sys
sys.path.insert(0, 'cutedsl')

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm


TAGS = [
    "gmem_to_smem",  # LOAD_B = 0
    "UNUSED_1",
    "smem_to_reg",   # S2R_B = 2
    "UNUSED_3",
    "compute",       # COMP_B = 4
    "UNUSED_5",
    "reg_to_gmem",   # STORE_B = 6
    "UNUSED_7",
]

LOAD_B      = 0
LOAD_E      = 1
S2R_B       = 2
S2R_E       = 3
COMP_B      = 4
COMP_E      = 5
STORE_B     = 6
STORE_E     = 7
NUM_TAGS    = 8


@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(
        T.i32(),
        [],
        "mov.u32 $0, %smid;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(t)


# profiler buffer layout per warp/block:
#   [0]         = cnt (how many entries written)
#   [1 + i*4+0] = sm_id
#   [1 + i*4+1] = tag
#   [1 + i*4+2] = start timestamp
#   [1 + i*4+3] = duration
PROFILER_HEADER = 1
PROFILER_ENTRY  = 4  # fields per entry


@cute.jit
def naive_smem_launcher(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, profiler: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]

    BM, BN, BK = 32, 32, 64

    naive_smem_kernel(mA, mB, mC, profiler, M, N, K).launch(
        grid=[N // BN, M // BM, 1],
        block=[BM, BN, 1],
    )


@cute.kernel
def naive_smem_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    profiler: cute.Tensor,  # int64 [num_blocks, 1 + num_entries * 4]
    M: int,
    N: int,
    K: int,
):
    BM, BN, BK = 32, 32, 64
    assert BM == BN, f"BM ({BM}) must equal BN ({BN})"
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

    # block id into profiler buffer
    grid_x = cute.arch.grid_dim()[0]
    bid = bidy * grid_x + bidx
    num_entries = (profiler.shape[1] - PROFILER_HEADER) // PROFILER_ENTRY

    rA = cute.make_rmem_tensor((BK,), cutlass.Float16)
    rB = cute.make_rmem_tensor((BK,), cutlass.Float16)

    acc  = cute.Float32(0)
    cnt  = cutlass.Int32(0)
    sm   = smid_u32()
    k_tiles = K // BK

    # ---- helper: profiler start/stop ----
    # inlined manually since no closures in kernel scope

    for ctak in range(0, K, BK):
        kt = ctak // BK

        # LOAD start
        if tid == 0:
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 0] = cutlass.Int64(sm)
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 1] = cutlass.Int64(LOAD_B)
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2] = globaltimer_u64()

        num_loads = BM * BK
        for i in range(tid, num_loads, num_threads):
            row = i // BK
            col = i % BK
            sA[row, col] = gA[bidy * BM + row, ctak + col]
            sB[row, col] = gB[bidx * BN + row, ctak + col]

        cute.arch.sync_threads()

        # LOAD stop
        if tid == 0:
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 3] = globaltimer_u64() - profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2]
            cnt = cnt + cutlass.Int32(1)

        # S2R start
        if tid == 0:
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 0] = cutlass.Int64(sm)
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 1] = cutlass.Int64(S2R_B)
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2] = globaltimer_u64()

        for kk in range(BK):
            rA[kk] = sA[tidy, kk]
            rB[kk] = sB[tidx, kk]

        cute.arch.sync_threads()

        # S2R stop
        if tid == 0:
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 3] = globaltimer_u64() - profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2]
            cnt = cnt + cutlass.Int32(1)

        # COMP start
        if tid == 0:
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 0] = cutlass.Int64(sm)
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 1] = cutlass.Int64(COMP_B)
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2] = globaltimer_u64()

        for kk in range(BK):
            acc += cute.Float32(rA[kk]) * cute.Float32(rB[kk])

        cute.arch.sync_threads()

        # COMP stop
        if tid == 0:
            profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 3] = globaltimer_u64() - profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2]
            cnt = cnt + cutlass.Int32(1)

    # STORE start
    if tid == 0:
        profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 0] = cutlass.Int64(sm)
        profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 1] = cutlass.Int64(STORE_B)
        profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2] = globaltimer_u64()

    gC[bidy * BM + tidy, bidx * BN + tidx] = cute.Float16(acc)

    cute.arch.sync_threads()

    # STORE stop + flush cnt
    if tid == 0:
        profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 3] = globaltimer_u64() - profiler[bid, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2]
        cnt = cnt + cutlass.Int32(1)
        profiler[bid, 0] = cutlass.Int64(cnt)


def write_chrome_trace_json(profiler_cpu, out_path: str, block_id: int = 0):
    if isinstance(profiler_cpu, torch.Tensor):
        profiler_cpu = profiler_cpu.contiguous().cpu().tolist()

    events = []
    data = profiler_cpu[block_id]
    cnt = data[0]
    base = None

    for i in range(cnt):
        sm_id = data[PROFILER_HEADER + i * PROFILER_ENTRY + 0]
        tag   = data[PROFILER_HEADER + i * PROFILER_ENTRY + 1]
        start = data[PROFILER_HEADER + i * PROFILER_ENTRY + 2]
        dur   = data[PROFILER_HEADER + i * PROFILER_ENTRY + 3]
        if base is None:
            base = start
        if tag % 2 != 0:  # skip _E entries (odd indices are unused stops)
            continue
        events.append(dict(
            name=TAGS[tag], ph="X",
            ts=(start - base) / 1000.0,
            dur=dur / 1000.0,
            pid=sm_id, tid=0,
        ))

    with open(out_path, "w") as f:
        import json
        json.dump({"traceEvents": events}, f)


def main():
    M, N, K = 64, 64, 512
    BM, BN, BK = 32, 32, 64

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    grid_x = N // BN
    grid_y = M // BM
    num_blocks = grid_x * grid_y
    k_tiles = K // BK
    # entries per block: 3 per k-tile (load, s2r, comp) + 1 store
    num_entries = k_tiles * 3 + 1
    profiler = torch.zeros((num_blocks, PROFILER_HEADER + num_entries * PROFILER_ENTRY),
                           device="cuda", dtype=torch.int64)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)
    profiler_ = from_dlpack(profiler, assumed_align=16)

    compiled = cute.compile(naive_smem_launcher, A_, B_, C_, profiler_)

    # warmup
    compiled(A_, B_, C_, profiler_)
    torch.cuda.synchronize()

    # clean capture
    profiler.zero_()
    compiled(A_, B_, C_, profiler_)
    torch.cuda.synchronize()

    C_ref = torch.matmul(A, B.T)
    assert torch.allclose(C, C_ref, atol=1e-2, rtol=1e-2), \
        f"Correctness check failed\nmax abs diff: {(C - C_ref).abs().max().item():.6f}"
    print("Correctness check PASSED")

    out_path = "cute_globaltimer_trace.json"
    write_chrome_trace_json(profiler, out_path, block_id=0)
    print(f"Wrote trace JSON: {out_path}")
    print("Open with chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()