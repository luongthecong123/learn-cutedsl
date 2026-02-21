import cutlass
import cutlass.cute as cute

@cute.jit
def naive(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]
    
    BM, BN = 16, 16
    
    naive_kernel(mA, mB, mC, M, N, K).launch(
        grid=[N // BN, M // BM, 1],
        block=[BM, BN, 1])

@cute.kernel
def naive_kernel(
    gA: cute.Tensor,  # [M, K]
    gB: cute.Tensor,  # [N, K]
    gC: cute.Tensor,  # [M, N]
    M: int,
    N: int,
    K: int
):

    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    tidx, tidy, _ = cute.arch.thread_idx()
    
    acc = cute.Float32(0)
    
    for k in range(K):
        acc += cute.Float32(gA[bidy * bdimy + tidy, k]) * cute.Float32(gB[bidx * bdimx + tidx, k])
    
    gC[bidy * bdimy + tidy, bidx * bdimx + tidx] = cute.Float16(acc)