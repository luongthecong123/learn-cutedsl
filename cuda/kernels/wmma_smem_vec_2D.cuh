// #include "cuda_common.cuh"
#include "../cuda_common.cuh"

namespace wmma_smem_vec_2D {

    template<int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WARP_TILING_X, int WARP_TILING_Y>
    __global__ void __launch_bounds__((BN / WMMA_N / WARP_TILING_Y) * WARPSIZE * (BM / WMMA_M / WARP_TILING_X))
    gemm_kernel_2d_warp_tiling(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        IndexWrapper<const half, 2> A_idx(A, M, K);
        IndexWrapper<const half, 2> B_idx(B, N, K);
        IndexWrapper<      half, 2> C_idx(C, M, N);

        constexpr int lds = BK + 8;
        __shared__ half sA[BM][lds];
        __shared__ half sB[BN][lds];

        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int numThreads = blockDim.x * blockDim.y;

        // 2D array of accumulator fragments
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_TILING_X][WARP_TILING_Y];

        // Initialize all accumulators to zero
        #pragma unroll
        for (int ti = 0; ti < WARP_TILING_X; ti++) {
            #pragma unroll
            for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                wmma::fill_fragment(frag_acc[ti][tj], 0.f);
            }
        }


        // Each warp now handles WARP_TILING_X x WARP_TILING_Y output tiles
        size_t warpIdx = threadIdx.x / WARPSIZE;
        size_t warpIdy = threadIdx.y;

        for (int ctak = 0; ctak < K; ctak += BK) {
            // Load sA, sB with vectorized 128-bit loads (8 half elements)
            constexpr int VEC_SIZE = 8;  // float4 = 8 halves
            {
                int numLoads = (BM * BK) / VEC_SIZE;
// #pragma unroll
                for (int i = tid; i < numLoads; i += numThreads) {
                    int row = i / (BK / VEC_SIZE);
                    int col = (i % (BK / VEC_SIZE)) * VEC_SIZE;
                    int globalRow = blockIdx.y * BM + row;
                    int globalCol = ctak + col;

                    if (globalRow < M && globalCol < K) {
                        float4 tmp = *reinterpret_cast<const float4*>(&A_idx.at(globalRow, globalCol));
                        *reinterpret_cast<float4*>(&sA[row][col]) = tmp;
                    } else {
                        // #pragma unroll
                        for (int j = 0; j < VEC_SIZE; j++) {
                            sA[row][col + j] = __float2half(0.0f);
                        }
                    }
                }
            }

            {
// #pragma unroll
                int numLoads = (BN * BK) / VEC_SIZE;
                for (int i = tid; i < numLoads; i += numThreads) {
                    int row = i / (BK / VEC_SIZE);
                    int col = (i % (BK / VEC_SIZE)) * VEC_SIZE;
                    int globalRow = blockIdx.x * BN + row;
                    int globalCol = ctak + col;

                    if (globalRow < N && globalCol < K) {
                        float4 tmp = *reinterpret_cast<const float4*>(&B_idx.at(globalRow, globalCol));
                        *reinterpret_cast<float4*>(&sB[row][col]) = tmp;
                    } else {
                        // #pragma unroll
                        for (int j = 0; j < VEC_SIZE; j++) {
                            sB[row][col + j] = __float2half(0.0f);
                        }
                    }
                }
            }

            __syncthreads();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a[WARP_TILING_X];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b[WARP_TILING_Y];

            // MMA computation with 2D tiling
            for (int k = 0; k < BK; k += WMMA_K) {
                // Load A, B fragments for this warp's tiles
                // #pragma unroll
                for (int ti = 0; ti < WARP_TILING_X; ti++) {
                    int sA_row = warpIdy * WMMA_M * WARP_TILING_X + ti * WMMA_M;
                    wmma::load_matrix_sync(frag_a[ti], &sA[sA_row][k], lds);
                }

                // #pragma unroll
                for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                    int sB_row = warpIdx * WMMA_N * WARP_TILING_Y + tj * WMMA_N;
                    wmma::load_matrix_sync(frag_b[tj], &sB[sB_row][k], lds);
                }

                // #pragma unroll
                for (int ti = 0; ti < WARP_TILING_X; ti++) {
                    // #pragma unroll
                    for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                        wmma::mma_sync(frag_acc[ti][tj], frag_a[ti], frag_b[tj], frag_acc[ti][tj]);
                    }
                }
            }

            __syncthreads();
        }

        // Store results with bounds checking
        #pragma unroll
        for (int ti = 0; ti < WARP_TILING_X; ti++) {
            #pragma unroll
            for (int tj = 0; tj < WARP_TILING_Y; tj++) {
                int mAg = blockIdx.y * BM + warpIdy * WMMA_M * WARP_TILING_X + ti * WMMA_M;
                int nBg = blockIdx.x * BN + warpIdx * WMMA_N * WARP_TILING_Y + tj * WMMA_N;

                if (mAg < M && nBg < N) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> fragC_out;

                    for (size_t i = 0; i < frag_acc[ti][tj].num_elements; ++i) {
                        fragC_out.x[i] = __float2half(frag_acc[ti][tj].x[i]);
                    }

                    wmma::store_matrix_sync(&C_idx.at(mAg, nBg), fragC_out, N, wmma::mem_row_major);
                }
            }
        }
    }

    void gemm_host(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        constexpr int BM = 64;
        constexpr int BN = 128;
        constexpr int BK = 16;

        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;

        constexpr int WARP_TILING_X = 1;
        constexpr int WARP_TILING_Y = 2;

        // Each warp now handles WARP_TILING_X x WARP_TILING_Y tiles
        constexpr int WARPS_M = BM / WMMA_M / WARP_TILING_X;
        constexpr int WARPS_N = BN / WMMA_N / WARP_TILING_Y;

        dim3 blockSize(WARPS_N * WARPSIZE, WARPS_M);
        dim3 gridSize((N + BN - 1) / BN, (M + BM - 1) / BM);

        static_assert(BM % (WMMA_M * WARP_TILING_X) == 0, "BM must be divisible by WMMA_M * WARP_TILING_X");
        static_assert(BN % (WMMA_N * WARP_TILING_Y) == 0, "BN must be divisible by WMMA_N * WARP_TILING_Y");
        static_assert(BK % WMMA_K == 0, "BK must be divisible by WMMA_K");
        static_assert(WARPS_M * WARPS_N * WARPSIZE <= 1024, "Too many threads per block");

        gemm_kernel_2d_warp_tiling<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_TILING_X, WARP_TILING_Y>
            <<<gridSize, blockSize>>>(A, B, C, M, N, K);
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
} // namespace wmma_smem_vec_2D