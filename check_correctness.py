import torch
import sys
sys.path.insert(0, 'cuda')
from gemm_cuda import gemm_cuda_func

M, K, N = 1024, 1024, 1024

matA = torch.randn(M, K, dtype=torch.float16, device='cuda')
matB = torch.randn(N, K, dtype=torch.float16, device='cuda')

c_test = torch.matmul(matA, matB.T)

abs_tol = 1e-2
rel_tol = 1e-2

algos = ["naive", "smem", "smem_2D", "wmma_simple", "wmma_smem", "wmma_smem_vec", "wmma_smem_vec_2D"]

for algo in algos:
    output = gemm_cuda_func(matA, matB, algo)

    abs_diff = torch.abs(c_test - output)
    rel_diff = abs_diff / (torch.abs(c_test) + 1e-8)

    mean_abs = abs_diff.mean().item()
    mean_rel = rel_diff.mean().item()

    print(f"\nAlgorithm: {algo}")
    print(f"Mean absolute error: {mean_abs:.6f}")
    print(f"Mean relative error: {mean_rel:.6f}")

    if mean_abs > abs_tol or mean_rel > rel_tol:
        print(f"ERROR: Tolerance exceeded - abs_diff: {mean_abs:.6f} (tol: {abs_tol}), rel_diff: {mean_rel:.6f} (tol: {rel_tol})")
        print(f"10 first values Reference: {c_test[0,:10]}, Output: {output[0,:10]}")
        print(f"10 last values Reference: {c_test[-1,:10]}, Output: {output[-1,:10]}")
    else:
        print("PASSED")
