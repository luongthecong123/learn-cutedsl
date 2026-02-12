#include "cuda_common.cuh"
#include "kernels/naive.cuh"
#include "kernels/wmma_simple.cuh"
#include "kernels/wmma_smem_vec_2D.cuh"
#include "kernels/wmma_smem_vec.cuh"
#include "kernels/wmma_smem.cuh"

void gemm_cuda(
    void* A,
    void* B,
    void* C,
    int M, int N, int K, int option
 );

void gemm_cuda(
    void* A,
    void* B,
    void* C,
    int M, int N, int K, int option
 ){
    switch (option) {
        case 0: naive::gemm_launcher(A, B, C, M, N, K); break;
        case 1: wmma_simple::gemm_launcher(A, B, C, M, N, K); break;
        case 2: wmma_smem::gemm_launcher(A, B, C, M, N, K); break;
        case 3: wmma_smem_vec::gemm_launcher(A, B, C, M, N, K); break;
        case 4: wmma_smem_vec_2D::gemm_launcher(A, B, C, M, N, K); break;
    }

    // or simply
    // wmma_smem_vec_2D::gemm_launcher(A, B, C, M, N, K);
}