import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

@cute.jit
def naive_smem_launcher(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]
    
    BM, BN, BK = 16, 16, 16
    assert BM == BN, f"[NAIVE SMEM ab_] BM ({BM}) must equal BN ({BN})"
    
    naive_smem_kernel(mA, mB, mC, M, N, K).launch(
        grid=[N // BN, M // BM, 1],
        block=[BM, BN, 1])

@cute.kernel
def naive_smem_kernel(
    gA: cute.Tensor,  # [M, K]
    gB: cute.Tensor,  # [N, K]
    gC: cute.Tensor,  # [M, N]
    M: int,
    N: int,
    K: int
):
    BM, BN, BK = 16, 16, 16
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
    acc = cute.Float32(0)

    for ctak in range(0, K, BK):
        # Load sA, sB
        num_loads = BM * BK
        for i in range(tid, num_loads, num_threads):
            row = i // BK
            col = i % BK
            sA[row, col] = gA[bidy * BM + row, ctak + col]
            sB[row, col] = gB[bidx * BN + row, ctak + col]
        
        cute.arch.sync_threads()
        
        # MMa on smem:
        for mmak in range(BK):
            acc += cute.Float32(sA[tidy, mmak]) * cute.Float32(sB[tidx, mmak])    

        cute.arch.sync_threads()
    
    gC[bidy * bdimy + tidy, bidx * bdimx + tidx] = cute.Float16(acc)

def main():
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    compiled = cute.compile(naive_smem_launcher, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), f"CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()