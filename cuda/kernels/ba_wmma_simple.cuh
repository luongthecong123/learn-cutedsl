// #include "cuda_common.cuh"
#include "../cuda_common.cuh"

namespace wmma_simple {

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

        // Define half tensor matrix-multiply-accumulate (mma) instruction and allocate registers
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
        wmma::fill_fragment(fragC, 0.f); // Fill accumulator with zeroes

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;

        int warpIdx = threadIdx.x / WARPSIZE;
        int warpIdy = threadIdx.y;
        int mAg = blockIdx.y * BM + warpIdy * WMMA_M;
        int nBg = blockIdx.x * BN + warpIdx * WMMA_N;

        for (int mmak = 0; mmak < K; mmak += WMMA_K) {
            // Load BMxBK and BNxBK tiles straight from global memory --> register
            wmma::load_matrix_sync(fragA, &A_idx.at(mAg, mmak), K);
            wmma::load_matrix_sync(fragB, &B_idx.at(nBg, mmak), K);
            // Perform tensor core mma on 
            wmma::mma_sync(fragC, fragA, fragB, fragC);
        }

        __syncthreads();

        // Cast back to fp16 and store results of accumulator back to global memory
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
        constexpr int BK = 16;

        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;

        // Each block has 4*4=16 warps, each warp calculates a 16*16 ouput tile in matC
        constexpr int WARPS_M = BM / WMMA_M;  // 64 = 4
        constexpr int WARPS_N = BN / WMMA_N;  // 64 = 4

        dim3 blockSize(WARPS_N * WARPSIZE, WARPS_M);  // (4*32, 4) = (128, 4) = 512 threads
        dim3 gridSize((N + BN - 1) / BN, (M + BM - 1) / BM);

        static_assert(BM % WMMA_M == 0, "BM must be divisible by WMMA_M");
        static_assert(BN % WMMA_N == 0, "BN must be divisible by WMMA_N");
        static_assert(BK == WMMA_K, "BK must be equal to WMMA_K");
        static_assert(WARPS_M * WARPS_N * WARPSIZE <= 1024, "Too many threads per block");

        gemm_kernel<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K>
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
} // namespace wmma_simple