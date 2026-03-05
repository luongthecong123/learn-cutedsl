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
    .uv_pip_install("torch", "nvidia-cutlass-dsl", "ninja")
    .add_local_dir(CURRENT_DIR / "cutedsl", remote_path="/root/cutedsl")
)

app = modal.App("learn-cutedsl", image=image)

@app.function(gpu="H100")
def run_kernel_H100():
    # print("b2_wmma_smem.py")
    # from cutedsl.b2_wmma_smem import main
    # main()
    # print("b5_wmma_tma_load_store.py")
    # from cutedsl.b5_wmma_tma_load_store import main
    # main()
    # print("b6_wmma_colwise_scaling.py")
    # from cutedsl.b6_wmma_colwise_scaling import main
    # main()
    print("c1_wgmma_tma_load_store.py")
    from cutedsl.c1_wgmma_tma_load_store import main
    main()
    # print("c2_wgmma_tma_pipeline.py")
    # from cutedsl.c2_wgmma_tma_pipeline import main
    # main()
    # print("c3_wgmma_tma_specialized_pipeline.py")
    # from cutedsl.c3_wgmma_tma_specialized_pipeline import main
    # main()

@app.function(gpu="B200")
def run_kernel_B200():
    print("d1_tcgen05_tma.py")
    from cutedsl.d1_tcgen05_tma_umma import main
    main()

@app.local_entrypoint()
def main():
    # run_kernel_H100.remote()
    run_kernel_B200.remote()
    