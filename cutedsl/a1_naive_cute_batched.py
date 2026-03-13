import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

@cute.jit
def cute_naive(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    grid_m = (mA.shape[1] + BM - 1) // BM
    grid_n = (mB.shape[1] + BN - 1) // BN
    
    gemm_kernel(mA, mB, mC).launch(
        grid=[grid_m, grid_n, mA.shape[0]],
        block=[256, 1, 1]
    )

@cute.kernel
def gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    
    # Slice the batch dimension first, then use 2D tiling
    gA_batch = gA[bidz, None, None]  # (M, K)
    gB_batch = gB[bidz, None, None]  # (N, K)
    gC_batch = gC[bidz, None, None]  # (M, N)
    
    tiler = (BM, BN, BK)
    coord = (bidx, bidy, None)
    
    # CTA partitioning on the 2D slices
    gA_tile = cute.local_tile(gA_batch, tiler=tiler, coord=coord, proj=(1, None, 1))
    gB_tile = cute.local_tile(gB_batch, tiler=tiler, coord=coord, proj=(None, 1, 1))
    gC_tile = cute.local_tile(gC_batch, tiler=tiler, coord=coord, proj=(1, 1, None))
    
    # Tile the universal mma atom (1x1) to shape atoms_layout -> 16x16 tiles of this atom
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)
    print("tiled mma: ", tiled_mma)

    # Thread partitioning 
    # Each tile in C has shape BMxBN = 16x32   
    thr_mma = tiled_mma.get_slice(tidx)
    # tCgC: partitioning pattern tC applied to gC to produce tCgC
    tCgC = thr_mma.partition_C(gC_tile)
    tCrC = tiled_mma.make_fragment_C(tCgC)

    tCrC.fill(0)
    print("tCrC begin: ", tCrC)
    
    K_tiles = gA_tile.shape[2]
    for k in range(K_tiles):
        gA_k = gA_tile[None, None, k]
        gB_k = gB_tile[None, None, k]
        print("gA_k: ", gA_k)
        print("gB_k: ", gB_k)        
        
        tCgA = thr_mma.partition_A(gA_k)
        tCgB = thr_mma.partition_B(gB_k)
        
        tCrA = tiled_mma.make_fragment_A(tCgA)
        tCrB = tiled_mma.make_fragment_B(tCgB)
        
        print("tCrA: ", tCrA)
        print("tCrB: ", tCrB)
        
        tCrA.store(tCgA.load())
        tCrB.store(tCgB.load())
        
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
    
    print("tCrC final: ", tCrC)
    
    tCgC.store(tCrC.load())

def main():
    BS, M, N, K = 8, 1024, 1024, 1024

    A = torch.randn((BS, M, K), dtype=torch.float32, device="cuda")
    B = torch.randn((BS, N, K), dtype=torch.float32, device="cuda")
    C = torch.empty((BS, M, N), dtype=torch.float32, device="cuda")

    # Partial dynamic: only BS (mode=0) is dynamic, M/N/K stay static
    # A (BS,M,K):(M*K, K, 1) -> mark mode=0 -> (?,M,K):(?,K,1)
    # B (BS,N,K):(N*K, K, 1) -> mark mode=0 -> (?,N,K):(?,K,1)
    # C (BS,M,N):(M*N, N, 1) -> mark mode=0 -> (?,M,N):(?,N,1)
    A_ = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0)
    B_ = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0)
    C_ = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0)

    compiled = cute.compile(cute_naive, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.bmm(A, B.mT), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    tflops = (2 * BS * M * N * K) / (time * 1e6)
    print(f"DURATION: {time:>5.4f} µs | TFLOPS: {tflops:.4f}")

    # Reuse with different BS — no recompilation
    BS2 = 4
    A2 = torch.randn((BS2, M, K), dtype=torch.float32, device="cuda")
    B2 = torch.randn((BS2, N, K), dtype=torch.float32, device="cuda")
    C2 = torch.empty((BS2, M, N), dtype=torch.float32, device="cuda")

    A2_ = from_dlpack(A2, assumed_align=16).mark_compact_shape_dynamic(mode=0)
    B2_ = from_dlpack(B2, assumed_align=16).mark_compact_shape_dynamic(mode=0)
    C2_ = from_dlpack(C2, assumed_align=16).mark_compact_shape_dynamic(mode=0)

    compiled(A2_, B2_, C2_)

    assert torch.allclose(C2, torch.bmm(A2, B2.mT), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED 2"
    print("CORRECTNESS PASS 2 (BS=4, reused kernel)")
    time2 = benchmark(compiled, kernel_arguments=JitArguments(A2_, B2_, C2_))
    tflops2 = (2 * BS2 * M * N * K) / (time2 * 1e6)
    print(f"DURATION: {time2:>5.4f} µs | TFLOPS: {tflops2:.4f}")

if __name__ == "__main__":
    main()