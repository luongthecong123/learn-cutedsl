#include "../cuda_common.cuh"

namespace naive {

    __global__ void __launch_bounds__(256) gemm_kernel(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[row * K + k]) * __half2float(B[col * K + k]);
            }
            C[row * N + col] = __float2half(sum);
        }
    }

    void gemm_host(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        dim3 blockSize(16, 16);  // 256 threads total (16*16)
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);

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
} // namespace naive