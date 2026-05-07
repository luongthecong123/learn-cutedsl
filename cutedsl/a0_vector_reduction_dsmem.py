"""
a0_vector_reduction_dsmem.py — three-level reduction of an Int32 vector using
Distributed Shared Memory (DSMEM) on Blackwell.

Levels of reduction:
  1. Warp reduction: each thread accumulates strided partial sum, then warp
     shuffle butterfly reduces it; lane 0 writes one int to SMEM.
  2. Block reduction: warp 0 loads the per-warp partials from SMEM and warp-
     reduces them — thread 0 of each CTA now holds its CTA's partial sum.
  3. Grid reduction (single cluster): each CTA's thread 0 atomically adds its
     partial sum into CTA 0's SMEM accumulator via DSMEM (`mapa` + cluster-
     scoped `nvvm.red`). After a `cluster_arrive`/`cluster_wait`, CTA 0
     writes the final sum to global memory.

References:
  - histogram_dsmem.py (DSMEM atomic-add via nvvm.red)
  - fused_kernel/dsa_attn.py (warp_reduce shuffle helper)
"""
import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import nvvm

CLUSTER_SIZE    = 4
NUM_WARPS       = 8
THREADS_PER_CTA = NUM_WARPS * 32   # 256


@dsl_user_op
def red_shared_cluster_add_i32(ptr_ir, val: cutlass.Int32, *, loc=None, ip=None) -> None:
    """DSMEM reduce-add: `red.relaxed.cluster.shared::cluster.add.s32 [ptr], val`.

    `ptr_ir` must be a `!llvm.ptr<3>` obtained via `cute.arch.mapa`.
    """
    nvvm.red(
        op=nvvm.ReductionOp.ADD,
        type_=nvvm.ReductionType.S32,
        a=ptr_ir,
        b=val.ir_value(loc=loc, ip=ip),
        mem_order=nvvm.MemOrderKind.RELAXED,
        shared_space=nvvm.SharedSpace.shared_cluster,
        mem_scope=nvvm.MemScopeKind.CLUSTER,
        loc=loc, ip=ip,
    )


@cute.jit
def warp_reduce_sum(val: cutlass.Int32, width: cutlass.Constexpr = 32) -> cutlass.Int32:
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.kernel
def vector_reduction_kernel(
    data: cute.Tensor,   # [N] int32
    out:  cute.Tensor,   # [1] int32
    n:    cutlass.Int32,
):
    tid  = cute.arch.thread_idx()[0]
    warp = tid // cutlass.Int32(32)
    lane = tid % cutlass.Int32(32)
    rank = cute.arch.block_in_cluster_idx()[0]    # 0..CLUSTER_SIZE-1
    cluster_threads = cutlass.Int32(CLUSTER_SIZE * THREADS_PER_CTA)
    gtid = rank * cutlass.Int32(THREADS_PER_CTA) + tid

    # ── SMEM allocation ──
    # `warp_partials`: NUM_WARPS scratch slots for level-1 results.
    # `cluster_acc`:   single int32, target of DSMEM atomic adds (only CTA 0's
    #                  copy is the real accumulator, but every CTA must allocate
    #                  it so the SMEM layout matches across the cluster — `mapa`
    #                  remaps offsets, it doesn't allocate).
    allocator = cutlass.utils.SmemAllocator()
    warp_partials = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None,
    )
    cluster_acc = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None,
    )

    # CTA 0 zeroes its accumulator before any peer atomic can land
    if rank == cutlass.Int32(0) and tid == cutlass.Int32(0):
        cluster_acc[0] = cutlass.Int32(0)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    # ── Level 1: per-thread strided sum + warp shuffle reduction ──
    thread_sum = cutlass.Int32(0)
    pos = gtid
    while pos < n:
        thread_sum = thread_sum + data[pos]
        pos = pos + cluster_threads

    warp_sum = warp_reduce_sum(thread_sum, width=32)
    if lane == cutlass.Int32(0):
        warp_partials[warp] = warp_sum
    cute.arch.sync_threads()

    # ── Level 2: warp 0 reduces the NUM_WARPS partials ──
    if warp == cutlass.Int32(0):
        v = cutlass.Int32(0)
        if lane < cutlass.Int32(NUM_WARPS):
            v = warp_partials[lane]
        cta_sum = warp_reduce_sum(v, width=32)

        # ── Level 3: thread 0 of each CTA adds into CTA 0's accumulator
        # via DSMEM (`mapa` → `red.relaxed.cluster.shared::cluster.add.s32`).
        if lane == cutlass.Int32(0):
            dst = cute.arch.mapa(cluster_acc.iterator, cutlass.Int32(0))
            red_shared_cluster_add_i32(dst, cta_sum)

    # All peer atomics must land before CTA 0 reads
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    # CTA 0 writes the cluster-wide sum to global memory
    if rank == cutlass.Int32(0) and tid == cutlass.Int32(0):
        out[0] = cluster_acc[0]


@cute.jit
def vector_reduction_launch(data: cute.Tensor, out: cute.Tensor, n: cutlass.Int32):
    vector_reduction_kernel(data, out, n).launch(
        grid=[CLUSTER_SIZE, 1, 1],
        block=[THREADS_PER_CTA, 1, 1],
        cluster=[CLUSTER_SIZE, 1, 1],
    )


def main():
    N = 1 << 20  # 1M elements
    torch.manual_seed(0)
    data = torch.randint(-8, 9, (N,), dtype=torch.int32, device="cuda")
    out  = torch.zeros(1, dtype=torch.int32, device="cuda")

    data_ = from_dlpack(data, assumed_align=16)
    out_  = from_dlpack(out,  assumed_align=16)

    compiled = cute.compile(vector_reduction_launch, data_, out_, cutlass.Int32(N))

    # correctness
    out.zero_()
    compiled(data_, out_, cutlass.Int32(N))
    torch.cuda.synchronize()
    ref = int(data.sum().item())
    got = int(out.item())
    print(f"N={N}  ref={ref}  got={got}  {'PASS' if ref == got else 'FAIL'}")

    # benchmark
    from cutlass.cute.testing import benchmark, JitArguments
    time = benchmark(compiled, kernel_arguments=JitArguments(data_, out_, cutlass.Int32(N)))
    bytes_read = N * 4
    bw = bytes_read / (time * 1e3)  # GB/s
    print(f"DURATION: {time:>8.2f} µs | BW: {bw:.2f} GB/s")


if __name__ == "__main__":
    main()
