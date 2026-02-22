#include "cuda_common.cuh"
#include "kernels/a1_naive.cuh"
#include "kernels/a2_smem.cuh"
#include "kernels/a3_smem_2D.cuh"
#include "kernels/b1_wmma_simple.cuh"
#include "kernels/b2_wmma_smem.cuh"
#include "kernels/b3_wmma_smem_vec.cuh"
#include "kernels/b4_wmma_smem_vec_2D.cuh"


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
        case 1: smem::gemm_launcher(A, B, C, M, N, K); break;
        case 2: thread_tiling::gemm_launcher(A, B, C, M, N, K); break;
        case 3: wmma_simple::gemm_launcher(A, B, C, M, N, K); break;
        case 4: wmma_smem::gemm_launcher(A, B, C, M, N, K); break;
        case 5: wmma_smem_vec::gemm_launcher(A, B, C, M, N, K); break;
        case 6: wmma_smem_vec_2D::gemm_launcher(A, B, C, M, N, K); break;
    }

    // or simply
    // wmma_smem_vec_2D::gemm_launcher(A, B, C, M, N, K);
}