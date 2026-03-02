import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

@cute.jit
def cute_naive(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    grid_m = (mA.shape[0] + BM - 1) // BM
    grid_n = (mB.shape[0] + BN - 1) // BN
    
    gemm_kernel(mA, mB, mC).launch(
        grid=[grid_m, grid_n, 1],
        block=[256, 1, 1]
    )

@cute.kernel
def gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    
    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    
    tiler = (BM, BN, BK)
    coord = (bidx, bidy, None)
    
    # CTA partitioning
    # gA_tile(m,n,k) = gA(bidx * BM + m, bidy * BK + k) = ptr_A + (bidx * BM + m) * K + bidy * BK + k
    gA_tile = cute.local_tile(gA, tiler=tiler, coord=coord, proj=(1, None, 1))
    gB_tile = cute.local_tile(gB, tiler=tiler, coord=coord, proj=(None, 1, 1))
    gC_tile = cute.local_tile(gC, tiler=tiler, coord=coord, proj=(1, 1, None))
    
    # Tile the universal mma atom (1x1) to shape atoms_layout -> 16x16 tiles of this atom
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float16)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)
    print("tiled mma: ", tiled_mma)

    # Thread partitioning 
    # Each tile in C has shape BMxBN = 16x32   
    thr_mma = tiled_mma.get_slice(tidx)
    # tCgC: partitioning pattern tC applied to gC to produce tCgC
    tCgC = thr_mma.partition_C(gC_tile)
    tCrC = tiled_mma.make_fragment_C(tCgC)

    tCrC.fill(cute.Float16(0))
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
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    compiled = cute.compile(cute_naive, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()