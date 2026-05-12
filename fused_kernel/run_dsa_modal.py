"""Modal runner for DSA attn + indexer kernels with CUPTI timing.

Usage:
    modal run fused_kernel/run_dsa_modal.py
    modal run fused_kernel/run_dsa_modal.py --kernel attn
    modal run fused_kernel/run_dsa_modal.py --kernel indexer

    # Capture a perfetto trace from the pipeline-staging attn kernel
    # (writes <out>.json locally; default uuid=385742b2):
    modal run fused_kernel/run_dsa_modal.py --kernel trace
    modal run fused_kernel/run_dsa_modal.py --kernel trace --uuid 9d4a5f21 \\
        --out trace_attn_staging_9d4a5f21.json

Workload tensors (sparse_indices for attn; seq_lens + block_table for indexer)
are bundled as safetensors under ./workloads/. Other inputs (q, k, v, weights)
are generated with torch.randn / torch.randint using a fixed seed.
"""
import argparse
import modal
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent

SM_SCALE = 0.1352337788608801

ATTN_UUIDS = [
    "9d4a5f21",  # T=2  — fused single-block
    "385742b2",  # T=8  — KV-split + reduce
    "2207f0fd",  # T=23 — KV-split + reduce (large)
]
IDX_UUIDS = [
    "30cecff1",  # T=1 max_pages=1   — fast skip-GEMM
    "ef12ac76",  # T=4 max_pages=44  — indexer + topk
]

cuda_version = "13.0.1"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "torch",
        "nvidia-cutlass-dsl",
        "ninja",
        "apache-tvm-ffi",
        "torch-c-dlpack-ext",
        "numpy",
        "safetensors",
        "flashinfer-python",
        "cupti-python",
    )
    .add_local_dir(REPO_DIR / "fused_kernel", remote_path="/root/fused_kernel")
)

app = modal.App("run-dsa-modal", image=image)


def _load_st(name: str):
    from safetensors.torch import load_file
    return load_file(f"/root/fused_kernel/workloads/{name}")


def _build_attn_inputs(uuid_prefix: str):
    import torch
    from fused_kernel.dsa_attn import (
        NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, NUM_PAGES, PAGE_SIZE,
    )
    sparse_indices = _load_st(f"attn_{uuid_prefix}.safetensors")["sparse_indices"].to("cuda").to(torch.int32)
    T = sparse_indices.shape[0]

    g = torch.Generator(device="cuda").manual_seed(0)
    q_nope = torch.randn(T, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda", generator=g)
    q_pe   = torch.randn(T, NUM_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda", generator=g)
    ckv    = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda", generator=g)
    kpe    = torch.randn(NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda", generator=g)

    output = torch.zeros(T, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
    lse    = torch.full((T, NUM_HEADS), -float("inf"), dtype=torch.float32, device="cuda")
    label  = f"uuid={uuid_prefix} T={T}"
    return label, q_nope, q_pe, ckv, kpe, sparse_indices, output, lse


def _build_indexer_inputs(uuid_prefix: str):
    import torch
    from fused_kernel.dsa_indexer import NUM_HEADS, HEAD_DIM, TOP_K, PAGE_SIZE
    NUM_PAGES_IDX = 11923

    sf = _load_st(f"idx_{uuid_prefix}.safetensors")
    seq_lens    = sf["seq_lens"].to("cuda").to(torch.int32)
    block_table = sf["block_table"].to("cuda").to(torch.int32)
    T = seq_lens.shape[0]

    g = torch.Generator(device="cuda").manual_seed(0)
    q_index_fp8 = torch.randn(
        T, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device="cuda", generator=g,
    ).to(torch.float8_e4m3fn)
    k_index_cache_fp8 = torch.randint(
        0, 256, (NUM_PAGES_IDX, PAGE_SIZE, 1, HEAD_DIM + 4),
        dtype=torch.uint8, device="cuda", generator=g,
    ).view(torch.int8)
    weights = torch.randn(T, NUM_HEADS, dtype=torch.float32, device="cuda", generator=g)

    topk_indices = torch.full((T, TOP_K), -1, dtype=torch.int32, device="cuda")
    label = (f"uuid={uuid_prefix} T={T} max_pages={block_table.shape[1]} "
             f"max_sl={int(seq_lens.max())}")
    return label, q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices


def _bench(fn, label: str):
    import numpy as np
    from flashinfer.testing import bench_gpu_time
    times = bench_gpu_time(
        fn=fn, dry_run_iters=3, repeat_iters=100,
        enable_cupti=True, use_cuda_graph=False, cold_l2_cache=True,
    )
    med = float(np.median(times))
    print(f"  {label:<60s}  median = {med*1000:8.3f} us  ({len(times)} iters)", flush=True)
    return med


@app.function(gpu="B200", timeout=1800)
def bench_attn():
    import sys, torch
    sys.path.insert(0, "/root")
    print("=" * 78)
    print("DSA ATTN — HybridDSA")
    print("=" * 78)
    from fused_kernel.dsa_attn import run as run_attn

    for uuid_prefix in ATTN_UUIDS:
        label, q_nope, q_pe, ckv, kpe, si, out, lse = _build_attn_inputs(uuid_prefix)
        run_attn(q_nope, q_pe, ckv, kpe, si, SM_SCALE, out, lse)
        torch.cuda.synchronize()
        _bench(lambda: run_attn(q_nope, q_pe, ckv, kpe, si, SM_SCALE, out, lse), label)


@app.function(gpu="B200", timeout=1800)
def bench_indexer():
    import sys, torch
    sys.path.insert(0, "/root")
    print("=" * 78)
    print("DSA INDEXER — Indexer_kvsplit_v4_hist_pdl")
    print("=" * 78)
    from fused_kernel.dsa_indexer import run as run_idx

    for uuid_prefix in IDX_UUIDS:
        label, q, k, w, sl, bt, out = _build_indexer_inputs(uuid_prefix)
        run_idx(q, k, w, sl, bt, out)
        torch.cuda.synchronize()
        _bench(lambda: run_idx(q, k, w, sl, bt, out), label)


@app.function(gpu="B200", timeout=1800)
def trace_attn_staging(uuid_prefix: str = "385742b2") -> str:
    """Run the pipeline-staging attn kernel once with probe buffers and
    return a perfetto-compatible chrome trace JSON string."""
    import sys, torch
    sys.path.insert(0, "/root")
    print("=" * 78)
    print(f"DSA ATTN — pipeline_staging (probe trace)  uuid={uuid_prefix}")
    print("=" * 78)
    import fused_kernel.dsa_attn_tcgen05_warpspec_pipeline_staging as ps

    label, q_nope, q_pe, ckv, kpe, si, output, lse = _build_attn_inputs(uuid_prefix)
    print(f"  inputs: {label}")

    Bc = (ps.NUM_HEADS // ps.HEADS_PER_SPLIT) * ps.NUM_SPLITS
    Br = ps.LIMIT_REQUEST * ps.NUM_HEADS  # compile-time fixed
    probe_prod   = torch.zeros((Bc, ps.PROBE_COLS_PROD),   dtype=torch.int64, device="cuda")
    probe_cons   = torch.zeros((Bc, ps.PROBE_COLS_CONS),   dtype=torch.int64, device="cuda")
    probe_sgemm  = torch.zeros((Bc, ps.PROBE_COLS_SGEMM),  dtype=torch.int64, device="cuda")
    probe_reduce = torch.zeros((Br, ps.PROBE_COLS_REDUCE), dtype=torch.int64, device="cuda")

    def _run():
        ps._compiled(
            q_nope, q_pe, ckv, kpe, si,
            ps._hybrid.partial_out, ps._hybrid.partial_lse,
            output, lse,
            probe_prod, probe_cons, probe_sgemm, probe_reduce,
        )

    # Warmup
    for _ in range(3):
        output.zero_(); lse.fill_(-float("inf"))
        probe_prod.zero_(); probe_cons.zero_(); probe_sgemm.zero_(); probe_reduce.zero_()
        _run()
        torch.cuda.synchronize()

    # Measured run (also collects probes for trace)
    output.zero_(); lse.fill_(-float("inf"))
    probe_prod.zero_(); probe_cons.zero_(); probe_sgemm.zero_(); probe_reduce.zero_()
    _run()
    torch.cuda.synchronize()

    # Bench timing alongside the trace
    _bench(_run, f"pipeline_staging {label}")

    ep, bp, ec, bc, es, bs = ps.dump_compute(probe_prod, probe_cons, probe_sgemm, Bc, ps.NUM_SPLITS)
    er, br = ps.dump_reduce(probe_reduce, Br)
    trace_json = ps.build_combined_trace(ep, bp, ec, bc, es, bs, er, br)
    print(f"  trace events: {len(trace_json)} bytes")
    return trace_json


@app.local_entrypoint()
def main(kernel: str = "all", uuid: str = "385742b2", out: str = "trace_attn_staging.json"):
    if kernel not in ("attn", "indexer", "all", "trace"):
        raise SystemExit(f"unknown kernel '{kernel}' — use attn|indexer|all|trace")
    if kernel in ("attn", "all"):
        bench_attn.remote()
    if kernel in ("indexer", "all"):
        bench_indexer.remote()
    if kernel == "trace":
        trace_json = trace_attn_staging.remote(uuid)
        out_path = Path(out)
        out_path.write_text(trace_json)
        print(f"wrote {out_path.resolve()}  ({len(trace_json)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["attn", "indexer", "all", "trace"], default="all")
    parser.add_argument("--uuid", default="385742b2")
    parser.add_argument("--out",  default="trace_attn_staging.json")
    args = parser.parse_args()
    with modal.enable_output(), app.run():
        if args.kernel in ("attn", "all"):
            bench_attn.remote()
        if args.kernel in ("indexer", "all"):
            bench_indexer.remote()
        if args.kernel == "trace":
            trace_json = trace_attn_staging.remote(args.uuid)
            out_path = Path(args.out)
            out_path.write_text(trace_json)
            print(f"wrote {out_path.resolve()}  ({len(trace_json)} bytes)")
