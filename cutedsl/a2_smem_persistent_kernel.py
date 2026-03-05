import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments


@cute.jit
def persistent_smem_launcher(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
):
    BM, BN = 16, 16
    c_tile_shape = (BM, BN)

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

    persistent_smem_kernel(mA, mB, mC, tile_sched_params).launch(
        grid=grid,
        block=[BM, BN, 1],
    )


@cute.kernel
def persistent_smem_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    BM, BN, BK = 16, 16, 16
    PAD = 8
    K = gA.shape[1]

    allocator = cutlass.utils.SmemAllocator()
    layout_sA = cute.make_layout((BM, BK), stride=(BK + PAD, 1))
    layout_sB = cute.make_layout((BN, BK), stride=(BK + PAD, 1))
    sA = allocator.allocate_tensor(cutlass.Float16, layout_sA, 16, None)
    sB = allocator.allocate_tensor(cutlass.Float16, layout_sB, 16, None)

    bdimx, bdimy, _ = cute.arch.block_dim()
    tidx, tidy, _ = cute.arch.thread_idx()
    tid = tidy * bdimx + tidx
    num_threads = bdimx * bdimy

    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    while work_tile.is_valid_tile:
        tile_m = work_tile.tile_idx[0]
        tile_n = work_tile.tile_idx[1]

        acc = cute.Float32(0)

        for ctak in range(0, K, BK):
            num_loads_A = BM * BK
            for i in range(tid, num_loads_A, num_threads):
                row = i // BK
                col = i % BK
                sA[row, col] = gA[tile_m * BM + row, ctak + col]

            num_loads_B = BN * BK
            for i in range(tid, num_loads_B, num_threads):
                row = i // BK
                col = i % BK
                sB[row, col] = gB[tile_n * BN + row, ctak + col]

            cute.arch.sync_threads()

            for mmak in range(BK):
                acc += cute.Float32(sA[tidx, mmak]) * cute.Float32(sB[tidy, mmak])

            cute.arch.sync_threads()

        gC[tile_m * BM + tidx, tile_n * BN + tidy] = cute.Float16(acc)

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()


def main():
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    print("SM COUNT: ", sm_count)
    max_active_clusters = sm_count

    compiled = cute.compile(persistent_smem_launcher, A_, B_, C_, max_active_clusters)
    compiled(A_, B_, C_)
    torch.cuda.synchronize()

    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")

    time_us = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    time_s = time_us * 1e-6
    tflops = (2 * M * N * K) / time_s / 1e12

    print(f"DURATION: {time_us:>8.2f} µs")
    print(f"TFLOPS:  {tflops:>8.4f}")


if __name__ == "__main__":
    main()