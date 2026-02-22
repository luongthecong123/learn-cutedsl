import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

@cute.jit
def smem_2D_launcher(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]
    
    # cfg = smem_2D_Config(BM=32, BN=32, BK=32, TILE_M=2, TILE_N=2, PAD=8)
    THRX = 16
    THRY = 16
    TILE_M = 1
    TILE_N = 1
    BM = THRY * TILE_M
    BN = THRX * TILE_N   
    BK = 32
    PAD = 8
    
    grid = [N // BN, M // BM, 1]
            
    smem_2D_kernel(mA, mB, mC, M, N, K).launch(
        grid=[N // BN, M // BM, 1],
        block=[THRX, THRY, 1])

@cute.kernel
def smem_2D_kernel(
    gA: cute.Tensor,  # [M, K]
    gB: cute.Tensor,  # [N, K]
    gC: cute.Tensor,  # [M, N]
    M: int,
    N: int,
    K: int
    # cfg: smem_2D_Config
):
    THRX = 16
    THRY = 16
    TILE_M = 1
    TILE_N = 1
    BM = THRY * TILE_M
    BN = THRX * TILE_N   
    BK = 32
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
    
    acc_layout = cute.make_layout((TILE_M, TILE_N))
    fragAcc = cute.make_rmem_tensor(acc_layout, cute.Float32)
    for tm in range(TILE_M):
        for tn in range(TILE_N):
            fragAcc[tm, tn] = cute.Float32(0)

    for ctak in range(0, K, BK):
        # Load sA, sB
        num_loads = BM * BK
        for i in range(tid, num_loads, num_threads):
            row = i // BK
            col = i % BK
            gRow = bidy * BM + row
            gCol = ctak + col

            if gRow < M and gCol < K:
                sA[row, col] = gA[gRow, gCol]

        num_loads = BN * BK
        for i in range(tid, num_loads, num_threads):
            row = i // BK
            col = i % BK
            gRow = bidx * BN + row
            gCol = ctak + col

            if gRow < N and gCol < K:
                sB[row, col] = gB[gRow, gCol]        
        
        cute.arch.sync_threads()
        
        # MMa on smem:
        for mmak in range(BK):
            for tm in range(TILE_M):
                for tn in range(TILE_N):
                    fragAcc[tm, tn] += \
                    cute.Float32(sA[tidy * TILE_M + tm, mmak]) * \
                    cute.Float32(sB[tidx * TILE_N + tn, mmak])    

        cute.arch.sync_threads()
        
    for tm in range(TILE_M):
        for tn in range(TILE_N):
            gRow = bidy * BM + tidy * TILE_M + tm
            gCol = bidx * BN + tidx * TILE_N + tn
            if gRow < M and gCol < N:
                gC[gRow, gCol] = cute.Float16(fragAcc[tm, tn])

def main():
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    compiled = cute.compile(smem_2D_launcher, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), f"CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()