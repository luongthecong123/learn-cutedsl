#include "../cuda_common.cuh"
/*
matA @ matB.T = matC
where:
    matA: MxK row major (K-major) float16
    matB: NxK row major (K-major) float16
    matC: MxN row major (N-major) float16

Multi-threaded parallel matrix muliplication with shared memory.
Each thread will calculate 1 element in matrix C.
Split matrix C into smaller matrices (aka tiles in C), 
because if matrix C is too large, we might not have enough threads to calculate in 1 go (1 wave).
Therfore each tile C is calculated by a thread block (max threads per block (aka cta) = 1024).
Here the number threads per block = the number of matC elements that this block calculates.

Where each block performs tiled matmul BMxK @ BNxK.T = BMxBN, ("B" is for block).
To promote data reuse in matrix multiplication, shared memory (smem) is used which is order of magnitudes
faster than global memory read/write. Due to smem size limitation, K dimension is further split into 
tiles of length BK to create smaller tiles in matA of shape BMxBK and BNxBK in matB. These 
smem buffers are iteratively filled and used to compute a tile of BMxBN in matC.

Naive with shared memory buffer implementation:
Read data from global memory to registers (slow speed), store data from register to smem in coalesced manner, 
synchronization to make sure data are fully written (to avoid race condition). 
Then we load data from smem to registers (high speed, high data reuse) and perform tiled matmul on CUDA cores.
*/
namespace smem {

    __global__ void gemm_kernel(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        const int BM = 16;
        const int BN = 16;
        const int BK = 16;
        const int PADDING = 2;

        __shared__ half sA[BM][BK + PADDING];
        __shared__ half sB[BN][BK + PADDING];

        int tm = threadIdx.y;
        int tn = threadIdx.x;

        int gm = blockIdx.y * BM + tm;
        int gn = blockIdx.x * BN + tn;

        float sum = 0.0f;

        for (int bk = 0; bk < K / BK; bk++) {
            int gk_a = bk * BK + tn;
            int gk_b = bk * BK + tm;

            sA[tm][tn] = A[gm * K + gk_a];
            sB[tn][tm] = B[gn * K + gk_b];

            __syncthreads();

            for (int k = 0; k < BK; k++) {
                sum += __half2float(sA[tm][k]) * __half2float(sB[tn][k]);
            }

            __syncthreads();
        }

        C[gm * N + gn] = __float2half(sum);
    }

    void gemm_host(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        const int BM = 16;
        const int BN = 16;        
        dim3 blockSize(BM, BN);
        // Note: you can map blockIdx.x and blockIdx.y to vertical or horizontal axis either way
        // Here I'm mapping blockIdx.x to the axis of N, and blockIdx.y to the axis of M
        // But in the cuteDSL, I'm mapping the opposite way for cleaner code there
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
} // namespace smem
