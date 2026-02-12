# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a learning repository for CuTeDSL (CUDA tensor programming). The goal is to add hardware features and optimization techniques incrementally to GEMM (matrix multiplication) kernels and measure FLOPS speedups, making it easier to understand how to progressively optimize CUDA code.

## Architecture

The codebase uses PyTorch's JIT compilation system to compile and run CUDA kernels:

**Python Layer:**
- `cuda/gemm_cuda.py`: Main entry point that JIT-compiles CUDA code using torch.utils.cpp_extension.load() and exposes gemm_cuda() function
- Kernels are selected via an "algo" string parameter passed to gemm_cuda()

**C++/CUDA Layer:**
- `cuda/glue_code.cpp`: PyBind11 wrapper providing Python-C++ interface
- `cuda/glue_code.cu`: Dispatcher that routes to different kernel implementations based on option parameter (0=naive, 1=wmma_simple, 2=wmma_smem, 3=wmma_smem_vec, 4=wmma_smem_vec_2D)
- `cuda/cuda_common.cuh`: Common utilities including IndexWrapper (N-dimensional indexing helper similar to CuTe's Layout concept)
- `cuda/kernels/*.cuh`: Individual kernel implementations showing progressive optimizations:
  - naive.cuh: Basic GEMM implementation
  - wmma_simple.cuh: Using Tensor Cores (WMMA API)
  - wmma_smem.cuh: Adding shared memory tiling
  - wmma_smem_vec.cuh: Adding vectorized memory loads
  - wmma_smem_vec_2D.cuh: Adding 2D vectorized loads

## Running Code

The CUDA code is JIT-compiled automatically when importing gemm_cuda module. Just run Python files directly:

```bash
python cuda/gemm_cuda.py
```

**Requirements:**
- CUDA-capable GPU with compute capability >= 8.0 (Ampere or newer)
- PyTorch with CUDA support

The compilation happens on first import and cached in `cuda/build/` directory. Set `force_rebuild = True` in gemm_cuda.py to force recompilation.

## Code Rules

1. Do not use argsparse.
2. Code should be concise.
3. Don't add or modify code's comments
4. This is a learning, feature example repo, not a production code, so don't add protective guards (i.e. try except).
