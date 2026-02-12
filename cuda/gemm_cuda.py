import torch
from torch.utils.cpp_extension import load
import pathlib
import os
import sys

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not found")

force_rebuild = False
capability = torch.cuda.get_device_capability(torch.cuda.current_device())
name = torch.cuda.get_device_name(torch.cuda.current_device())

os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
print(f"GPU: {name}, compute capability: {capability[0]}.{capability[1]}")

if capability[0] < 8:
    raise RuntimeError(f"GPU compute capability {capability[0]}.{capability[1]} is below minimum required (8.0)")

dir_path = pathlib.Path(__file__).parent.absolute()
print(f"dir_path: {dir_path}")

build_dir = f"{dir_path}/build"

build_path = pathlib.Path(build_dir)
build_path.mkdir(parents=True, exist_ok=True)
if force_rebuild:
    for file in build_path.glob("*"):
        file.unlink()

extra_ldflags = []

gemm_cuda_compiled = load(
    name='gemm',
    sources=[f"{dir_path}/glue_code.cu", f"{dir_path}/glue_code.cpp"],
    verbose=True,
    build_directory=build_dir,
    extra_cuda_cflags=[
        "-lineinfo",
        "-keep",
        f"-arch=sm_{capability[0]}{capability[1]}"
    ],
    extra_ldflags=extra_ldflags
)

ALGO_MAP = {
    "naive": 0,
    "wmma_simple": 1,
    "wmma_smem": 2,
    "wmma_smem_vec": 3,
    "wmma_smem_vec_2D": 4,
}

def gemm_cuda_func(
    matA: torch.Tensor,
    matB: torch.Tensor,
    algo: str,
):
    """
    Performs matrix multiplication C = A @ B^T using CUDA kernels.

    Args:
        matA: Input matrix A of shape (M, K), dtype fp16
        matB: Input matrix B of shape (N, K), dtype fp16 (will be transposed)
        algo: Algorithm selection - "naive", "wmma_simple", "wmma_smem",
              "wmma_smem_vec", "wmma_smem_vec_2D"

    Returns:
        Matrix C of shape (M, N), dtype fp16, where C = A @ B^T
    """
    assert algo in ALGO_MAP, f"algo must be one of {list(ALGO_MAP.keys())}, got {algo}"

    M, K = matA.shape
    N, _ = matB.shape

    matC = torch.empty(M, N, dtype=matA.dtype, device=matA.device)
    gemm_cuda_compiled.gemm(matA, matB, matC, M, N, K, ALGO_MAP[algo])

    return matC