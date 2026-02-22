import ray
from cutedsl.a2_smem_cuda_like import main

@ray.remote(num_gpus=1)
def run_kernel():
    main()

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
        
    num_gpus = int(ray.available_resources().get("GPU", 0))
    print(f"Available GPUs: {num_gpus}")

    ray.get(run_kernel.remote())
    ray.shutdown()