// #include "cuda_common.cuh"
#include "../cuda_common.cuh"

namespace wmma_smem {

    template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K>
    __global__ void __launch_bounds__((BN / WMMA_N) * WARPSIZE * (BM / WMMA_M)) gemm_kernel(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        IndexWrapper<const half, 2> A_idx(A, M, K);
        IndexWrapper<const half, 2> B_idx(B, N, K);
        IndexWrapper<      half, 2> C_idx(C, M, N);

        // Shared memory padding to avoid bank conflict
        constexpr int lds = BK + 8;
        __shared__ half sA[BM][lds];
        __shared__ half sB[BN][lds];

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
        wmma::fill_fragment(fragC, 0.f);

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;

        for (int ctak = 0; ctak < K; ctak += BK) {
            size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
            size_t numThreads = blockDim.x * blockDim.y;

            // Load sA, sB: each thread loads multiple elements
            for (size_t i = tid; i < BM * BK; i += numThreads) {
                int row = i / BK;
                int col = i % BK;
                int globalRow = blockIdx.y * BM + row;
                int globalCol = ctak + col;
                if (globalRow < M && globalCol < K) {
                    sA[row][col] = A_idx.at(globalRow, globalCol);
                } else {
                    sA[row][col] = __float2half(0.0f);
                }
            }

            for (size_t i = tid; i < BN * BK; i += numThreads) {
                int row = i / BK;
                int col = i % BK;
                int globalRow = blockIdx.x * BN + row;
                int globalCol = ctak + col;
                if (globalRow < N && globalCol < K) {
                    sB[row][col] = B_idx.at(globalRow, globalCol);
                } else {
                    sB[row][col] = __float2half(0.0f);
                }
            }

            __syncthreads();

            size_t warpIdx = threadIdx.x / WARPSIZE;
            size_t warpIdy = threadIdx.y;

            // mma on smem
            for (int k = 0; k < BK; k += WMMA_K) {
                wmma::load_matrix_sync(fragA, &sA[warpIdy * WMMA_M][k], lds);
                wmma::load_matrix_sync(fragB, &sB[warpIdx * WMMA_N][k], lds);
                wmma::mma_sync(fragC, fragA, fragB, fragC);
            }

            __syncthreads();
        }

        size_t warpIdx = threadIdx.x / WARPSIZE;
        size_t warpIdy = threadIdx.y;
        int mAg = blockIdx.y * BM + warpIdy * WMMA_M;
        int nBg = blockIdx.x * BN + warpIdx * WMMA_N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> fragC_out;

        for (size_t i = 0; i < fragC.num_elements; ++i) {
            fragC_out.x[i] = __float2half(fragC.x[i]);
        }

        wmma::store_matrix_sync(&C_idx.at(mAg, nBg), fragC_out, N, wmma::mem_row_major);
    }

    void gemm_host(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        constexpr int BM = 64;
        constexpr int BN = 64;
        constexpr int BK = 64;
    
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
    
        constexpr int WARPS_M = BM / WMMA_M;
        constexpr int WARPS_N = BN / WMMA_N;
    
        dim3 blockSize(WARPS_N * WARPSIZE, WARPS_M);
        dim3 gridSize((N + BN - 1) / BN, (M + BM - 1) / BM);
    
        static_assert(BM % WMMA_M == 0, "BM must be divisible by WMMA_M");
        static_assert(BN % WMMA_N == 0, "BN must be divisible by WMMA_N");
        static_assert(BK % WMMA_K == 0, "BK must be divisible by WMMA_K");
        static_assert(WARPS_M * WARPS_N * WARPSIZE <= 1024, "Too many threads per block");
    
        gemm_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K><<<gridSize, blockSize>>>(A, B, C, M, N, K);
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
} // namespace wmma_smem