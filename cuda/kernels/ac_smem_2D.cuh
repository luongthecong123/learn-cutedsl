#include "../cuda_common.cuh"

/*
matA @ matB.T = matC
where:
    matA: MxK row major (K-major) float16
    matB: NxK row major (K-major) float16
    matC: MxN row major (N-major) float16

Thread tiling matrix muliplication with shared memory.
Here, each block split matC into BMxBN tiles, inside, each thread calculates a smaller tile of shape TMxTN.
This resulted in spliting matC into larger tiles. Now the number threads per block < the number of matC elements
that this block calculates.

The purpose is to increase arthimetic intensity (computation per data transferred).
Also, since the amount of shared memory is limited per SM, an SM can only launch 2048 threads (Tesla GPUs)
or 1536 threads (GeForce GPUs), larger tiles allow us to use more shared memory per block/tile, leading to
more data reuse efficiency in the faster smem.

Additionally, we just use 1D thread block in this code, as it is sufficient, 2D thread block can be create
from 1D with integer division (/) and modulo (%) or use coorperative group.
Here, we split a block of 512 threads into 16 warps 
(Each warp = 32 threads scheduled together = SIMD size = physical unit of execution/computation). 
Each block calculates 64x64 elements, instead of 16x16 elements in example aa and ab  (8x arithmetic intensity)
*/

namespace thread_tiling {

    __global__ void gemm_kernel(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        const int BM = 64;
        const int BN = 64;
        const int BK = 64;
        const int TM = 2;
        const int TN = 4;
        const int PADDING = 4;

        __shared__ half sA[BM][BK + PADDING];
        __shared__ half sB[BN][BK + PADDING];

        cg::thread_block cta = cg::this_thread_block();
        auto group32 = cg::tiled_partition<32>(cta);
        int t32x = group32.thread_rank(); // 0 -> 31, thread idx in a warp
        int t32y = group32.meta_group_rank(); // 0 -> 15, warp idx in a block

        float c[TM][TN] = {0.0f};

        for (int cta_k = 0; cta_k < K; cta_k += BK){
            // Load data to shared memory
            #pragma unroll
            for (int warp_k = 0; warp_k < BK; warp_k += 32){
                #pragma unroll
                for (int warp_mn = 0; warp_mn < BM; warp_mn += 16){
                    sA[warp_mn + t32y][warp_k + t32x] = A[(blockIdx.y * BM + warp_mn + t32y) * K + cta_k + warp_k + t32x];
                    sB[warp_mn + t32y][warp_k + t32x] = B[(blockIdx.x * BN + warp_mn + t32y) * K + cta_k + warp_k + t32x];
                }
            }

            __syncthreads();

            // MMA on 
            #pragma unroll
            for (int mma_k = 0; mma_k < BK; ++mma_k){
                #pragma unroll
                for (int mma_m = 0; mma_m < TM; mma_m++){
                    #pragma unroll
                    for (int mma_n = 0; mma_n < TN; mma_n++){
                        c[mma_m][mma_n] += __half2float(sA[t32y * TN + mma_n][mma_k]) * __half2float(sB[t32x * TM + mma_m][mma_k]);
                    }
                }
            }

            __syncthreads();
        } // end cta_k

        #pragma unroll
        for (int mma_m = 0; mma_m < TM; mma_m++){
            #pragma unroll
            for (int mma_n = 0; mma_n < TN; mma_n++){
                C[(blockIdx.y * BM + t32y * TN + mma_n) * N + blockIdx.x * BN + t32x * TM + mma_m] = __float2half(c[mma_m][mma_n]);
            }
        }
    }

    void gemm_host(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        const int BM = 64;
        const int BN = 64;
        const int TM = 2;
        const int TN = 4;
        dim3 blockSize(BM * BN / (TM * TN));
        dim3 gridSize(N / BN, M / BM);

        gemm_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    void gemm_launcher(
        void* A,
        void* B,
        void* C,
        int M, int N, int K
    ){
        gemm_host(
            static_cast<const half*>(A),
            static_cast<const half*>(B),
            static_cast<half*>(C),
            M, N, K
        );
    }
} // namespace thread_tiling