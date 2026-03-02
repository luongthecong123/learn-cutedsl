import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

# Vector addition: C[i] = A[i] + B[i]
# Each thread handles exactly one element.
# This is the simplest possible GPU kernel

@cute.jit
def vector_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    N = mA.shape[0]

    BLOCK_SIZE = 256

    vector_add_kernel(mA, mB, mC, N).launch(
        grid=[(N + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1],
        block=[BLOCK_SIZE, 1, 1])

@cute.kernel
def vector_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    N: int,
):
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    tidx, _, _ = cute.arch.thread_idx()

    i = bidx * bdimx + tidx

    gC[i] = gA[i] + gB[i]

def main():
    N = 1024 * 1024  # 1 M elements

    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.empty(N, device="cuda", dtype=torch.float32)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    compiled = cute.compile(vector_add, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, A + B, atol=1e-5, rtol=1e-5), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs")

if __name__ == "__main__":
    main()
