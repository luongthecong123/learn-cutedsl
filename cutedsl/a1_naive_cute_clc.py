"""
a1_naive_cute_clc.py — Cluster Launch Control (CLC) variant of a1_naive_cute_persistent_kernel.

Architecture: Blackwell SM100a only.

Differences from a1_naive_cute_persistent_kernel:
  - Uses ClcDynamicPersistentTileScheduler instead of StaticPersistentTileScheduler
  - Grid is launched at FULL size (one CTA per output tile), not capped at sm_count
  - Each CTA starts on its own blockIdx tile, then steals subsequent tiles via CLC
  - CLC query fired EARLY (after tile decode) so response overlaps tile compute
  - Requires 2 extra SMEM tensors: smem_mbar (8 B) + smem_clc_rsp (16 B)
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments


@cute.jit
def clc_naive_launcher(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    BM, BN, BK = 64, 64, 16
    c_tile_shape = (BM, BN)

    # Compute total number of output tiles (one CTA per tile under CLC)
    gC = cute.zipped_divide(mC, tiler=c_tile_shape)
    num_ctas_mnl = (*gC[(0, (None, None))].shape, 1)

    cluster_shape_mnl = (1, 1, 1)

    clc_params = utils.ClcDynamicPersistentTileSchedulerParams(
        num_ctas_mnl,
        cluster_shape_mnl,
    )

    # Full-size grid: one CTA per output tile, no cap at max_active_clusters
    grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(clc_params)

    print("Grid size: ", grid)

    clc_naive_kernel(mA, mB, mC, clc_params).launch(
        grid=list(grid),
        block=[256, 1, 1],
    )


@cute.kernel
def clc_naive_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    clc_params: utils.ClcDynamicPersistentTileSchedulerParams,
):
    BM, BN, BK = 64, 64, 16

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    tiler = (BM, BN, BK)

    # Setup tiled MMA (same as a1_naive_cute)
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)

    thr_mma = tiled_mma.get_slice(tidx)

    # ── SMEM buffers required by CLC ──────────────────────────────────────────
    allocator = cutlass.utils.SmemAllocator()
    smem_mbar = allocator.allocate_tensor(
        cutlass.Int64, cute.make_layout((1,), stride=(1,)), 8, None
    )  # 8 B mbarrier
    smem_clc_rsp = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((4,), stride=(1,)), 16, None
    )  # 16 B CLC response

    # ── One-time mbarrier init ────────────────────────────────────────────────
    if tidx == 0:
        cute.arch.mbarrier_init(smem_mbar.iterator, 1)
    cute.arch.sync_threads()

    # ── Create CLC scheduler ──────────────────────────────────────────────────
    scheduler = utils.ClcDynamicPersistentTileScheduler.create(
        clc_params,
        cute.arch.block_idx(),
        cute.arch.grid_dim(),
        smem_clc_rsp.iterator,
    )

    phase = cutlass.Int32(0)
    work_tile = scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
        tile_m = work_tile.tile_idx[0]
        tile_n = work_tile.tile_idx[1]

        coord = (tile_m, tile_n, None)

        # CTA partitioning for this tile
        gA_tile = cute.local_tile(gA, tiler=tiler, coord=coord, proj=(1, None, 1))
        gB_tile = cute.local_tile(gB, tiler=tiler, coord=coord, proj=(None, 1, 1))
        gC_tile = cute.local_tile(gC, tiler=tiler, coord=coord, proj=(1, 1, None))

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

        # ── Issue CLC query, wait for response, fetch next tile ──────────────
        # advance_to_next_work uses elect_one (warp-level), so it must be
        # called from warp 0 only — issuing from every warp would fire 32
        # independent cancels and exhaust available tiles prematurely.
        cute.arch.sync_threads()
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(smem_mbar.iterator, 16)
            scheduler.advance_to_next_work(smem_mbar.iterator)

        cute.arch.mbarrier_wait(smem_mbar.iterator, phase)
        phase = phase ^ cutlass.Int32(1)
        work_tile = scheduler.get_current_work()


def main():
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((N, K), device="cuda", dtype=torch.float32)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"SM COUNT: {sm_count}")

    compiled = cute.compile(clc_naive_launcher, A_, B_, C_)
    compiled(A_, B_, C_)
    torch.cuda.synchronize()

    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")

    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    tflops = (2 * M * N * K) / (time * 1e6)
    print(f"DURATION: {time:>8.2f} µs | TFLOPS: {tflops:.4f}")


if __name__ == "__main__":
    main()
