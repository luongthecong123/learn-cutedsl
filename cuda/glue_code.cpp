#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>

void gemm_cuda(
    void* A,
    void* B,
    void* C,
    int M,
    int N,
    int K,
    int option
 );

void gemm(
    const at::Tensor& A, 
    const at::Tensor& B,
    const at::Tensor& C,
    int M,
    int N,
    int K,
    int option
) {
    gemm_cuda(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        M, N, K, option
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "GEMM (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("C"),
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("option"));
}