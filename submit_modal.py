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
    .pip_install("torch", "nvidia-cutlass-dsl", "ninja")
    .add_local_dir(CURRENT_DIR / "cutedsl", remote_path="/root/cutedsl")
)

app = modal.App("learn-cutedsl", image=image)

@app.function(gpu="H100")
def run_kernel():
    from cutedsl.c1_wgmma_tma_pipeline import main
    main()

@app.local_entrypoint()
def main():
    run_kernel.remote()