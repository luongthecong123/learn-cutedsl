import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments


@cute.jit
def persistent_naive_launcher(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
):
    BM, BN, BK = 16, 32, 16
    c_tile_shape = (BM, BN)

    # Compute total number of output tiles
    gC = cute.zipped_divide(mC, tiler=c_tile_shape)
    num_ctas_mnl = (*gC[(0, (None, None))].shape, 1)

    cluster_shape_mnl = (1, 1, 1)

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

    persistent_naive_kernel(mA, mB, mC, tile_sched_params).launch(
        grid=grid,
        block=[256, 1, 1],
    )


@cute.kernel
def persistent_naive_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    BM, BN, BK = 16, 32, 16

    tidx, _, _ = cute.arch.thread_idx()

    tiler = (BM, BN, BK)

    # Setup tiled MMA (same as a1_naive_cute)
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)

    thr_mma = tiled_mma.get_slice(tidx)

    # Persistent tile scheduler: each CTA loops over multiple output tiles
    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

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

        # Advance to next tile assigned to this CTA
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()


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
    # max_active_clusters = num clusters that can run concurrently in one wave.
    # 1 cluster = cluster_m * cluster_n * cluster_k CTAs
    cluster_shape_m, cluster_shape_n, cluster_shape_k = 1, 1, 1
    ctas_per_cluster = cluster_shape_m * cluster_shape_n * cluster_shape_k
    max_active_clusters = sm_count // ctas_per_cluster

    compiled = cute.compile(persistent_naive_launcher, A_, B_, C_, max_active_clusters)
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
