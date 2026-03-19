import modal
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

cuda_version = "13.0.1"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("torch", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi", "torch-c-dlpack-ext")
    .add_local_dir(CURRENT_DIR / "cutedsl", remote_path="/root/cutedsl")
    .add_local_dir(CURRENT_DIR / "cuda", remote_path="/root/cuda")
)

app = modal.App("learn-cutedsl", image=image)

@app.function(gpu="A100")
def run_kernel_sm80():
    import sys
    sys.path.insert(0, "/root")
    print("SM80 cuda/gemm_cuda.py")
    from cuda.gemm_cuda import main
    main()
    
    print("SM80 b2_wmma_smem.py")
    from cutedsl.b2_wmma_smem import main
    main()

@app.function(gpu="RTX-PRO-6000")
def run_kernel_sm120():
    # print("SM120 b2_wmma_smem.py")
    # from cutedsl.b2_wmma_smem import main
    # main()
    # print("SM120 b5_wmma_tma_load_store.py")
    # from cutedsl.b5_wmma_tma_load_store import main
    # main()
    print("SM120 b7_wmma_tma_specialized_pipeline.py")
    from cutedsl.b7_wmma_tma_specialized_pipeline import main
    main()


@app.function(gpu="H100")
def run_kernel_sm90():
    # print("SM90 c1_wgmma_tma_load_store.py")
    # from cutedsl.c1_wgmma_tma_load_store import main
    # main()
    print("SM90 c2_wgmma_tma_specialized_pipeline.py")
    from cutedsl.c2_wgmma_tma_specialized_pipeline import main
    main()
    

@app.function(gpu="B200")
def run_kernel_sm100():
    # print("SM100 d1_tcgen05_tma_umma.py")
    # from cutedsl.d1_tcgen05_tma_umma import main
    # main()
    # print("SM100 d1_tcgen05_tma_umma_ld.py")
    # from cutedsl.d1_tcgen05_tma_umma_ld import main
    # main()
    # print("SM100 d2_tcgen05_tma_specialized_pipeline.py")
    # from cutedsl.d2_tcgen05_tma_umma_specialized_pipeline import main
    # main()
    
    print("SM100 d3_tcgen05_tma_umma_2cta_specialized_pipeline.py")
    from cutedsl.d3_tcgen05_tma_umma_2cta_specialized_pipeline import main
    main()
      
    
    
@app.local_entrypoint()
def main():
    # run_kernel_sm80.remote()
    run_kernel_sm90.remote()
    # run_kernel_sm100.remote()
    # run_kernel_sm120.remote()
    
    