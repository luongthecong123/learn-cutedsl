"""
a1_naive_cute_grouped_scheduler.py — Grouped GEMM variant using
StaticPersistentGroupTileScheduler.

Demonstrates how a single kernel launch can compute a batch of independent
GEMMs (potentially with different M, N, K per group) by letting the scheduler
distribute tiles across all groups in round-robin fashion.

For simplicity this example uses the same (M, N, K) for every group so that
operands can be packed into a single (group_count, M, K) / (group_count, N, K)
tensor. Real grouped-GEMM workloads with varying shapes typically pass arrays
of pointers / per-group strides instead.

Architecture: any (uses MmaUniversalOp); tested on B200.
"""
import math

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments


@cute.jit
def grouped_naive_launcher(
    mA: cute.Tensor,                # (group_count, M, K)
    mB: cute.Tensor,                # (group_count, N, K)
    mC: cute.Tensor,                # (group_count, M, N)
    problem_shape_mnkl: cute.Tensor,  # (group_count, 4) Int32 — [M, N, K, 1]
    group_count: cutlass.Constexpr,
    total_tiles: cutlass.Constexpr,
    max_active_clusters: cutlass.Constexpr,
):
    BM, BN, BK = 16, 32, 16

    cluster_shape_mnl = (1, 1, 1)
    # All groups in the round-robin pool live in the z dimension.
    num_ctas_mnl = (1, 1, total_tiles)

    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl,
        cluster_shape_mnl,
        swizzle_size=1,
        raster_along_m=True,
    )

    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    print("Grid size: ", grid)

    grouped_naive_kernel(
        mA, mB, mC, problem_shape_mnkl, tile_sched_params, group_count
    ).launch(
        grid=grid,
        block=[256, 1, 1],
    )


@cute.kernel
def grouped_naive_kernel(
    gA: cute.Tensor,                  # (group_count, M, K)
    gB: cute.Tensor,                  # (group_count, N, K)
    gC: cute.Tensor,                  # (group_count, M, N)
    problem_shape_mnkl: cute.Tensor,  # (group_count, 4) Int32
    tile_sched_params: utils.PersistentTileSchedulerParams,
    group_count: cutlass.Constexpr,
):
    BM, BN, BK = 16, 32, 16

    tidx, _, _ = cute.arch.thread_idx()

    tiler = (BM, BN, BK)

    # MMA setup (same as a1_naive_cute)
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)
    thr_mma = tiled_mma.get_slice(tidx)

    # Grouped persistent scheduler
    initial_search_state = utils.create_initial_search_state()
    scheduler = utils.StaticPersistentGroupTileScheduler.create(
        tile_sched_params,
        cute.arch.block_idx(),
        cute.arch.grid_dim(),
        cluster_tile_shape_mnk=(BM, BN, BK),
        initial_search_state=initial_search_state,
        group_count=group_count,
        problem_shape_mnkl=problem_shape_mnkl,
    )

    work_tile = scheduler.get_current_work()

    while work_tile.is_valid_tile:
        r = work_tile.group_search_result
        group_idx = r.group_idx
        tile_m = r.cta_tile_idx_m
        tile_n = r.cta_tile_idx_n

        # Per-group tile slices (all groups share the same shape here, so a
        # 3-D tensor view works; varying shapes would index via pointer arrays).
        gA_g = gA[group_idx, None, None]
        gB_g = gB[group_idx, None, None]
        gC_g = gC[group_idx, None, None]

        coord = (tile_m, tile_n, None)
        gA_tile = cute.local_tile(gA_g, tiler=tiler, coord=coord, proj=(1, None, 1))
        gB_tile = cute.local_tile(gB_g, tiler=tiler, coord=coord, proj=(None, 1, 1))
        gC_tile = cute.local_tile(gC_g, tiler=tiler, coord=coord, proj=(1, 1, None))

        tCgC = thr_mma.partition_C(gC_tile)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0)

        K_tiles = gA_tile.shape[2]
        for k in range(K_tiles):
            gA_k = gA_tile[None, None, k]
            gB_k = gB_tile[None, None, k]

            tCgA = thr_mma.partition_A(gA_k)
            tCgB = thr_mma.partition_B(gB_k)

            tCrA = tiled_mma.make_fragment_A(tCgA)
            tCrB = tiled_mma.make_fragment_B(tCgB)

            tCrA.store(tCgA.load())
            tCrB.store(tCgB.load())

            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

        tCgC.store(tCrC.load())

        scheduler.advance_to_next_work()
        work_tile = scheduler.get_current_work()


def main():
    GROUP_COUNT = 4
    M, N, K = 256, 256, 256
    BM, BN = 16, 32

    A = torch.randn((GROUP_COUNT, M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((GROUP_COUNT, N, K), device="cuda", dtype=torch.float32)
    C = torch.empty((GROUP_COUNT, M, N), device="cuda", dtype=torch.float32)

    # Per-group [M, N, K, L=1] descriptor table
    problem_shape_mnkl = torch.tensor(
        [[M, N, K, 1]] * GROUP_COUNT, device="cuda", dtype=torch.int32
    )

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)
    PS_ = from_dlpack(problem_shape_mnkl, assumed_align=16)

    total_tiles = GROUP_COUNT * math.ceil(M / BM) * math.ceil(N / BN)

    device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"SM COUNT: {sm_count}")
    print(f"GROUP_COUNT: {GROUP_COUNT}, total_tiles: {total_tiles}")
    max_active_clusters = sm_count  # cluster size = 1

    compiled = cute.compile(
        grouped_naive_launcher,
        A_, B_, C_, PS_,
        GROUP_COUNT,
        total_tiles,
        max_active_clusters,
    )
    compiled(A_, B_, C_, PS_)
    torch.cuda.synchronize()

    ref = torch.matmul(A, B.transpose(-1, -2))  # (G, M, N)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")

    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_, PS_))
    flops = 2 * GROUP_COUNT * M * N * K
    tflops = flops / (time * 1e6)
    print(f"DURATION: {time:>8.2f} µs | TFLOPS: {tflops:.4f}")


if __name__ == "__main__":
    main()
