import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

@cute.jit
def cute_naive(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    grid_m = (mA.shape[0] + BM - 1) // BM
    grid_n = (mB.shape[0] + BN - 1) // BN
    
    gemm_kernel(mA, mB, mC).launch(
        grid=[grid_m, grid_n, 1],
        block=[256, 1, 1]
    )

@cute.kernel
def gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    
    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    
    tiler = (BM, BN, BK)
    coord = (bidx, bidy, None)
    
    gA_tile = cute.local_tile(gA, tiler=tiler, coord=coord, proj=(1, None, 1))
    gB_tile = cute.local_tile(gB, tiler=tiler, coord=coord, proj=(None, 1, 1))
    gC_tile = cute.local_tile(gC, tiler=tiler, coord=coord, proj=(1, 1, None))
    
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

# ── Wrappers for the three compilation modes ──────────────────────────────

# cute_naive expects cute.Tensor typed args, so we need a wrapper without
# type annotations for fully-dynamic (torch.Tensor passed directly).
@cute.jit
def cute_naive_dynamic(mA, mB, mC):
    BM, BN, BK = 16, 32, 16
    grid_m = (mA.shape[0] + BM - 1) // BM
    grid_n = (mB.shape[0] + BN - 1) // BN
    
    gemm_kernel(mA, mB, mC).launch(
        grid=[grid_m, grid_n, 1],
        block=[256, 1, 1]
    )

def check(C, A, B):
    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"

def bench_static(M, N, K):
    """Fully static: from_dlpack, all shapes baked in. Must recompile per shape."""
    print(f"\n{'='*60}")
    print(f"FULLY STATIC  M={M} N={N} K={K}")
    print(f"{'='*60}")
    
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((N, K), device="cuda", dtype=torch.float32)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)
    
    compiled = cute.compile(cute_naive, A_, B_, C_)
    compiled(A_, B_, C_)
    check(C, A, B)
    print("CORRECTNESS PASS")
    
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    tflops = (2 * M * N * K) / (time * 1e6)
    print(f"DURATION: {time:>8.2f} µs | TFLOPS: {tflops:.4f}")
    
    # Reuse compiled kernel with WRONG shape — shows static bakes in constants
    K2 = K // 2
    A2 = torch.randn((M, K2), device="cuda", dtype=torch.float32)
    B2 = torch.randn((N, K2), device="cuda", dtype=torch.float32)
    C2 = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    A2_ = from_dlpack(A2, assumed_align=16)
    B2_ = from_dlpack(B2, assumed_align=16)
    C2_ = from_dlpack(C2, assumed_align=16)
    
    # ❌ Reusing compiled kernel — it still thinks K=1024, will read garbage
    compiled(A2_, B2_, C2_)
    ref = torch.matmul(A2, B2.T)
    is_correct = torch.allclose(C2, ref, atol=1e-1, rtol=1e-1)
    print(f"  Reused static kernel for K={K2}: {'CORRECT' if is_correct else 'WRONG (expected!)'}")
    if not is_correct:
        max_err = (C2 - ref).abs().max().item()
        print(f"  Max error: {max_err:.4f}  (shapes are baked in, reads out-of-bounds memory)")
    
    return tflops

def bench_dynamic(M, N, K):
    """Fully dynamic: pass torch.Tensor directly, all shapes are runtime ?."""
    print(f"\n{'='*60}")
    print(f"FULLY DYNAMIC  M={M} N={N} K={K}")
    print(f"{'='*60}")
    
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((N, K), device="cuda", dtype=torch.float32)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    # Pass torch.Tensor directly -> auto mark_layout_dynamic -> (?,?):(?,1)
    compiled = cute.compile(cute_naive_dynamic, A, B, C)
    compiled(A, B, C)
    check(C, A, B)
    print("CORRECTNESS PASS")
    
    time = benchmark(compiled, kernel_arguments=JitArguments(A, B, C))
    tflops = (2 * M * N * K) / (time * 1e6)
    print(f"DURATION: {time:>8.2f} µs | TFLOPS: {tflops:.4f}")
    
    # Reuse with different shape — no recompilation
    M2, N2, K2 = M // 2, N // 2, K // 2
    A2 = torch.randn((M2, K2), device="cuda", dtype=torch.float32)
    B2 = torch.randn((N2, K2), device="cuda", dtype=torch.float32)
    C2 = torch.empty((M2, N2), device="cuda", dtype=torch.float32)
    
    compiled(A2, B2, C2)
    check(C2, A2, B2)
    
    time2 = benchmark(compiled, kernel_arguments=JitArguments(A2, B2, C2))
    tflops2 = (2 * M2 * N2 * K2) / (time2 * 1e6)
    print(f"  Reused for M={M2} N={N2} K={K2}:")
    print(f"  DURATION: {time2:>8.2f} µs | TFLOPS: {tflops2:.4f}")
    return tflops

def bench_partial(M, N, K):
    """Partial: M,N static, K dynamic via mark_compact_shape_dynamic."""
    print(f"\n{'='*60}")
    print(f"PARTIAL (M,N static / K dynamic)  M={M} N={N} K={K}")
    print(f"{'='*60}")
    
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((N, K), device="cuda", dtype=torch.float32)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    # A (M,K):(K,1) -> mark mode=1 (K) dynamic -> (M, ?{div=16}):(?{div=16}, 1)
    # B (N,K):(K,1) -> mark mode=1 (K) dynamic -> (N, ?{div=16}):(?{div=16}, 1)
    # C (M,N):(N,1) -> fully static
    A_ = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=1, divisibility=16)
    B_ = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=1, divisibility=16)
    C_ = from_dlpack(C, assumed_align=16)
    
    compiled = cute.compile(cute_naive, A_, B_, C_)
    compiled(A_, B_, C_)
    check(C, A, B)
    print("CORRECTNESS PASS")
    
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    tflops = (2 * M * N * K) / (time * 1e6)
    print(f"DURATION: {time:>8.2f} µs | TFLOPS: {tflops:.4f}")
    
    # Reuse with different K — no recompilation (M, N must match)
    K2 = K // 2
    A2 = torch.randn((M, K2), device="cuda", dtype=torch.float32)
    B2 = torch.randn((N, K2), device="cuda", dtype=torch.float32)
    C2 = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    A2_ = from_dlpack(A2, assumed_align=16).mark_compact_shape_dynamic(mode=1, divisibility=16)
    B2_ = from_dlpack(B2, assumed_align=16).mark_compact_shape_dynamic(mode=1, divisibility=16)
    C2_ = from_dlpack(C2, assumed_align=16)
    
    compiled(A2_, B2_, C2_)
    check(C2, A2, B2)
    
    time2 = benchmark(compiled, kernel_arguments=JitArguments(A2_, B2_, C2_))
    tflops2 = (2 * M * N * K2) / (time2 * 1e6)
    print(f"  Reused for K={K2}:")
    print(f"  DURATION: {time2:>8.2f} µs | TFLOPS: {tflops2:.4f}")
    return tflops

def main():
    M, N, K = 1024, 1024, 1024
    
    tf_static  = bench_static(M, N, K)
    tf_dynamic = bench_dynamic(M, N, K)
    tf_partial = bench_partial(M, N, K)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY  M={M} N={N} K={K}")
    print(f"{'='*60}")
    print(f"  Fully static:    {tf_static:.4f} TFLOPS")
    print(f"  Fully dynamic:   {tf_dynamic:.4f} TFLOPS  ({tf_dynamic/tf_static:.2f}x vs static)")
    print(f"  Partial dynamic: {tf_partial:.4f} TFLOPS  ({tf_partial/tf_static:.2f}x vs static)")

if __name__ == "__main__":
    main()
