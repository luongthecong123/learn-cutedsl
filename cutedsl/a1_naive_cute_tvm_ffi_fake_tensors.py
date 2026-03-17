import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

@cute.jit
def cute_naive(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    grid_m = (mA.shape[1] + BM - 1) // BM
    grid_n = (mB.shape[1] + BN - 1) // BN
    
    gemm_kernel(mA, mB, mC).launch(
        grid=[grid_m, grid_n, mA.shape[0]],
        block=[256, 1, 1]
    )

@cute.kernel
def gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    
    gA_batch = gA[bidz, None, None]  # (M, K)
    gB_batch = gB[bidz, None, None]  # (N, K)
    gC_batch = gC[bidz, None, None]  # (M, N)
    
    tiler = (BM, BN, BK)
    coord = (bidx, bidy, None)

    gA_tile = cute.local_tile(gA_batch, tiler=tiler, coord=coord, proj=(1, None, 1))
    gB_tile = cute.local_tile(gB_batch, tiler=tiler, coord=coord, proj=(None, 1, 1))
    gC_tile = cute.local_tile(gC_batch, tiler=tiler, coord=coord, proj=(1, 1, None))
    
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)
 
    thr_mma = tiled_mma.get_slice(tidx)
    tCgC = thr_mma.partition_C(gC_tile)
    tCrC = tiled_mma.make_fragment_C(tCgC)

    tCrC.fill(0)
    
    K_tiles = gA_tile.shape[2]
    for k in range(K_tiles):
        gA_k = gA_tile[None, None, k]
        gB_k = gB_tile[None, None, k]

        tCgA = thr_mma.partition_A(gA_k)
        tCgB = thr_mma.partition_B(gB_k)
        
        tCrA = tiled_mma.make_fragment_A(tCgA)
        tCrB = tiled_mma.make_fragment_B(tCgB)
        
        tCrA.store(tCgA.load())
        tCrB.store(tCgB.load())
        
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
    
    tCgC.store(tCrC.load())

def main():
    BS = cute.sym_int()
    # K = cute.sym_int(divisibility=64)
    K = 1024
    M, N = 1024, 1024
    A_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (BS, M, K), stride_order=(2, 1, 0), assumed_align=16)
    B_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (BS, N, K), stride_order=(2, 1, 0), assumed_align=16)
    C_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (BS, M, N), stride_order=(2, 1, 0), assumed_align=16)

    print("TVM FFI compilation")
    compiled = cute.compile(cute_naive, A_fake, B_fake, C_fake, options="--enable-tvm-ffi")
    
    BS, M, N, K = 8, 1024, 1024, 1024

    A = torch.randn((BS, M, K), dtype=torch.float32, device="cuda")
    B = torch.randn((BS, N, K), dtype=torch.float32, device="cuda")
    C = torch.empty((BS, M, N), dtype=torch.float32, device="cuda")
    
    compiled(A, B, C)

    assert torch.allclose(C, torch.bmm(A, B.mT), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A, B, C))
    tflops = (2 * BS * M * N * K) / (time * 1e6)
    print(f"DURATION: {time:>5.4f} µs | TFLOPS: {tflops:.4f}")

    #===============================================================================

    BS, M, N, K = 8, 1024, 1024, 1024

    A = torch.randn((BS, M, K), dtype=torch.float32, device="cuda")
    B = torch.randn((BS, N, K), dtype=torch.float32, device="cuda")
    C = torch.empty((BS, M, N), dtype=torch.float32, device="cuda")

    A_ = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=A.dim_order())
    B_ = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=B.dim_order())
    C_ = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=C.dim_order())

    compiled = cute.compile(cute_naive, A_, B_, C_)
    compiled(A_, B_, C_)
    
    print("Default compilation")
    assert torch.allclose(C, torch.bmm(A, B.mT), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    tflops = (2 * BS * M * N * K) / (time * 1e6)
    print(f"DURATION: {time:>5.4f} µs | TFLOPS: {tflops:.4f}")    

if __name__ == "__main__":
    main()