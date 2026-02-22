#include "../cuda_common.cuh"

/*
matA @ matB.T = matC
where:
    matA: MxK row major (K-major) float16
    matB: NxK row major (K-major) float16
    matC: MxN row major (N-major) float16

Naive multi-threaded parallel matrix muliplication.
Each thread will calculate 1 element in matrix C. 
Split matrix C into smaller matrices (aka tiles in C), 
because if matrix C is too large, we might not have enough threads to calculate in 1 go (1 wave).
Therfore each tile C is calculated by a thread block (max threads per block (aka cta) = 1024).
Here the number threads per block = the number of matC elements that this block calculates.

In this example we use thread block of size (16, 16) = 256 threads.
threadIdx.x and threadIdx.y will retrieve the coordinate of that thread in the block
blockIdx.x and blockIdx.y will retrieve the coordinate of that block in the grid (defined by problem size)

We pass in A and B as memory pointer, as memory is linear, we need to find a coordinate m,n equivalent
linear indexing. gm, gn is global memory coordinate of an element in matC and gm * N + gn is its linear
memory index (row-major). CuTeDSL can help abstract these coordinate mapping to make our life easier (or harder).

Naive implementatin:
Read data from global memory, reduction on the common dimension (K dimension), and save result 
to a register called "sum", then write the result back to global memory.
*/

namespace naive {

    __global__ void gemm_kernel(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        int gm = blockIdx.y * blockDim.y + threadIdx.y;
        int gn = blockIdx.x * blockDim.x + threadIdx.x;

        float sum = 0.0f; // Accumulate in float for higher precision
        for (int gk = 0; gk < K; gk++) {
            sum += __half2float(A[gm * K + gk]) * __half2float(B[gn * K + gk]);
        }
        C[gm * N + gn] = __float2half(sum);
    }

    void gemm_host(
        const half* A,
        const half* B,
        half* C,
        int M, int N, int K
    ){
        dim3 blockSize(16, 16);
        // Note: you can map blockIdx.x and blockIdx.y to vertical or horizontal axis either way
        // Here I'm mapping blockIdx.x to the axis of N, and blockIdx.y to the axis of M
        // But in the cuteDSL, I'm mapping the opposite way for cleaner code there
        dim3 gridSize(N / 16, M / 16);

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