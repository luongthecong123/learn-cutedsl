import torch
import ray
import sys
import time

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
# from cutedsl.aa_naive_cuda_like import naive
# from cutedsl.ab_smem_cuda_like import naive_smem_launcher
# from cutedsl.ac_smem_2D_cuda_like import smem_2D_launcher
from cutedsl.cc_wgmma_warp_spec import Gemm_TC

@ray.remote(num_gpus=1)
class Gemm:
    def __init__(self, algo_name="wgmma_warp_spec", cta_tiler=(128, 128, 64)):
        self.algo_name = algo_name
        self.cta_tiler = cta_tiler
        self.gemm_instance = Gemm_TC(cta_tiler=cta_tiler)

    def run_gemm(self, M=8192, N=8192, K=8192, iterations=10, abs_tol=1e-1, rel_tol=1e-1):
        device = 'cuda'

        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(N, K, device=device, dtype=torch.float16)

        c_test = torch.matmul(a, b.T)

        a_ = from_dlpack(a, assumed_align=16)
        b_ = from_dlpack(b, assumed_align=16)
        c = torch.empty(M, N, device=device, dtype=torch.float16)
        c_ = from_dlpack(c, assumed_align=16)

        compiled_code = cute.compile(self.gemm_instance, a_, b_, c_)

        compiled_code(a_, b_, c_)
        torch.cuda.synchronize()

        abs_diff = torch.abs(c_test - c)
        rel_diff = abs_diff / (torch.abs(c_test) + 1e-8)
        mean_abs = abs_diff.mean().item()
        mean_rel = rel_diff.mean().item()

        passed = mean_abs <= abs_tol and mean_rel <= rel_tol
        correctness_status = "PASSED" if passed else "FAILED"

        result = {
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "dtype": "fp16",
            "algo": f"{self.algo_name}_{self.cta_tiler}",
            "matrix_shape": (M, N, K),
            "iterations": iterations,
            "correctness": correctness_status,
            "mean_abs_diff": mean_abs,
            "mean_rel_diff": mean_rel,
        }

        if not passed:
            result["reference_first_10"] = c_test[0, :10].tolist()
            result["output_first_10"] = c[0, :10].tolist()
            result["reference_last_10"] = c_test[-1, :10].tolist()
            result["output_last_10"] = c[-1, :10].tolist()
            return result

        start_time = time.time()
        for i in range(iterations):
            c = torch.empty(M, N, device=device, dtype=torch.float16)
            c_ = from_dlpack(c, assumed_align=16)
            compiled_code(a_, b_, c_)
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / iterations) * 1000

        ops_per_gemm = 2 * M * N * K
        tflops = ops_per_gemm / (avg_time_ms / 1000) / 1e12

        result["total_time_sec"] = elapsed_time
        result["avg_time_per_gemm_ms"] = avg_time_ms
        result["tflops"] = tflops

        return result

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
    num_gpus = int(ray.available_resources().get("GPU", 0))
    print(f"Available GPUs: {num_gpus}")

    # algos = ["naive", "naive_smem", "smem_2D"]
    cta_tilers = [(128, 128, 64)]

    M, N, K = 8192, 8192, 8192
    iterations = 10
    abs_tol = 1e-1
    rel_tol = 1e-1

    futures = []
    for cta_tiler in cta_tilers:
        worker = Gemm.remote(algo_name="wgmma_warp_spec", cta_tiler=cta_tiler)
        future = worker.run_gemm.remote(M=M, N=N, K=K, iterations=iterations, abs_tol=abs_tol, rel_tol=rel_tol)
        futures.append(future)

    results = ray.get(futures)

    for i, result in enumerate(results):
        print(f"\n{'='*70}")
        print(f"Worker {i} Results")
        print(f"{'='*70}")
        print(f"Algorithm:      {result['algo']}")
        print(f"Device:         {result['cuda_device_name']}")
        print(f"Matrix shape:   {result['matrix_shape']}")
        print(f"Data type:      {result['dtype']}")
        print(f"\nCorrectness Check (Warmup):")
        print(f"Status:         {result['correctness']}")
        print(f"Mean abs diff:  {result['mean_abs_diff']:.6f}")
        print(f"Mean rel diff:  {result['mean_rel_diff']:.6f}")

        if result['correctness'] == "FAILED":
            print(f"\nFirst 10 elements:")
            print(f"  Reference: {result['reference_first_10']}")
            print(f"  Output:    {result['output_first_10']}")
            print(f"\nLast 10 elements:")
            print(f"  Reference: {result['reference_last_10']}")
            print(f"  Output:    {result['output_last_10']}")
        else:
            print(f"\nPerformance:")
            print(f"Iterations:     {result['iterations']}")
            print(f"Total time:     {result['total_time_sec']:.4f} s")
            print(f"Avg time:       {result['avg_time_per_gemm_ms']:.4f} ms")
            print(f"TFLOPS:         {result['tflops']:.2f}")

        print(f"{'='*70}")

    ray.shutdown()
