#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>

#define WARPSIZE 32

namespace cg = cooperative_groups;
using namespace nvcuda;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Helper function to convert logical N-dimension coordinate to 1D physical memory index/offset
// Pretty similar to cuteDSL's Layout, can reduce register pressure in large, highly fused kernel
template <typename T, uint32_t num_dim>
struct IndexWrapper
{
    template <typename... Dims>
    constexpr __host__ __device__ explicit IndexWrapper(T* ptr, Dims... dims) : dimensions{static_cast<uint32_t>(dims)...}, m_ptr(ptr)
    {
        static_assert(sizeof...(dims) == num_dim);
    }

    __host__ __device__ T* ptr() { return m_ptr; }
    __host__ __device__ const T* ptr() const { return m_ptr; }

    template <typename... Idx>
    __host__ __device__ T& at(Idx... idx)
    {
        static_assert(sizeof...(Idx) == num_dim);
        return m_ptr[_calc_1D_idx<0, Idx...>(idx...)];
    }

    template <typename... Idx>
    __host__ __device__ const T& at(Idx... idx) const
    {
        static_assert(sizeof...(Idx) == num_dim);
        return m_ptr[_calc_1D_idx<0, Idx...>(idx...)];
    }

    template <uint32_t dim_idx>
    constexpr __host__ __device__ uint32_t stride_size() const {
        if constexpr (dim_idx < num_dim) {
            return dimensions[dim_idx] * stride_size<dim_idx + 1>();
        } else {
            return 1;
        }
    }

    template <uint32_t dim_idx, typename Idx>
    constexpr static __host__ __device__ uint32_t _calc_1D_idx(Idx idx)
    {
        static_assert(dim_idx == num_dim - 1);
        return idx;
    }

    template <uint32_t dim_idx, typename Idx, typename... Tail>
    constexpr __host__ __device__ uint32_t _calc_1D_idx(Idx idx, Tail... tail) const
    {
        return idx * stride_size<dim_idx+1>() + _calc_1D_idx<dim_idx + 1, Tail...>(tail...);
    }

    uint32_t dimensions[num_dim];
    T* m_ptr;
};
