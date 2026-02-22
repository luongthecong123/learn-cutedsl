import torch
import sys
sys.path.insert(0, 'cutedsl')
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from a1_naive_cuda_like import naive
from a2_smem_cuda_like import naive_smem_launcher
from a3_smem_2D_cuda_like import smem_2D_launcher

M, K, N = 1024, 1024, 1024
device = 'cuda'

a = torch.randn(M, K, device=device, dtype=torch.float16)
b = torch.randn(N, K, device=device, dtype=torch.float16)

c_test = torch.matmul(a, b.T)

abs_tol = 1e-2
rel_tol = 1e-2

algos = {
    "naive": naive,
    "naive_smem": naive_smem_launcher,
    "smem_2D": smem_2D_launcher,
}

for algo_name, algo_func in algos.items():
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c = torch.empty(M, N, device=device, dtype=torch.float16)
    c_ = from_dlpack(c, assumed_align=16)

    compiled_code = cute.compile(algo_func, a_, b_, c_)
    compiled_code(a_, b_, c_)

    abs_diff = torch.abs(c_test - c)
    rel_diff = abs_diff / (torch.abs(c_test) + 1e-8)

    mean_abs = abs_diff.mean().item()
    mean_rel = rel_diff.mean().item()

    print(f"\nAlgorithm: {algo_name}")
    print(f"Mean absolute error: {mean_abs:.6f}")
    print(f"Mean relative error: {mean_rel:.6f}")

    if mean_abs > abs_tol or mean_rel > rel_tol:
        print(f"ERROR: Tolerance exceeded - abs_diff: {mean_abs:.6f} (tol: {abs_tol}), rel_diff: {mean_rel:.6f} (tol: {rel_tol})")
        print(f"10 first values Reference: {c_test[0,:10]}, Output: {c[0,:10]}")
        print(f"10 last values Reference: {c_test[-1,:10]}, Output: {c[-1,:10]}")
    else:
        print("PASSED")
