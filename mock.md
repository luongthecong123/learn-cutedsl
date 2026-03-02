# learn-cutedsl

Adding hardware features and optimization techniques brick by brick and measure the FLOPS speed up, making it easier to add CuTeDSL to your codebase according to your needs.

Apart from educational purpose, this repo can be treated as commonly used CuTeDSL API examples. Readers can pick out a feature/API and add it to their code or inject them as context to LLMs for coding assistance, thanks to LLMs being few-shot learners. For example, feed script `a1` + `a2` to an LLM to generate the profiling variant `a2_profile`.

## Overview

CuTeDSL is a Python-embedded domain-specific language that wraps CUDA/PTX, providing quality-of-life APIs for writing efficient GPU kernels with seamless PyTorch integration and blazing-fast JIT compilation. This repo walks through a series of progressively optimized GEMM kernels — from naïve thread-per-element up to Hopper WGMMA + TMA specialized pipelines and Blackwell tcgen05 — so you can understand what each hardware feature buys you in FLOPS.

**Pros:**
- Provides quality of life APIs to help write efficient CUDA kernels
- Easier to learn compared to the CUTLASS CuTe C++ counterpart, which can be daunting due to its highly templated code
- Exposes low-level features for speed-of-light (SoL) optimization — write it the CuTe way or the CUDA/PTX way (very versatile)
- Seamless integration with PyTorch with JIT compilation
- Blazing fast compilation and faster development cycle thanks to Python
- Latest hardware features on new NVIDIA GPUs are supported (Hopper SM90, Blackwell SM100/SM120)
- Great SoL examples by Junkai Wu and the CUTLASS team

**Cons:**
- Too many APIs to do the same thing — can be confusing and hard to master (a trade-off of versatility)
- Lacks detailed documentation on many APIs
- Examples are often too complicated due to SoL requirements, making it hard to pick apart individual concepts

---

**Here is a quick overview of what lies ahead:**

Section 1 provides beginners with a learning roadmap from vector addition up to Hopper and Blackwell kernels. Section 2 explains the core CuTeDSL APIs you will encounter everywhere: Layout, Shared Memory, Copy Atoms, and MMA Atoms. Section 3 dives into TV Layout for mapping thread registers to `(m, n)` coordinates, enabling layer fusion without materializing large tensors. Sections 4 and 5 cover the Hopper-specific hardware features TMA and WGMMA respectively. Section 6 combines them into a full async warp-specialized pipeline, contrasting `PipelineAsync` and `PipelineTmaAsync` and the SM120 fallback. Section 7 shows how to profile the pipeline with inline PTX clock reads. Section 8 introduces the Blackwell tcgen05 tensor core instruction.

## Frequently used APIs explanation

<details>
<summary><strong>1. Learning Curve</strong></summary>

CuTeDSL and CUDA in general have a very steep but rewarding learning curve, so don't get frustrated the first time you try it. The best approach is to look at examples, write kernels yourself, and observe the performance speedup — and understand *why* it speeds up. Once you can wrap your head around the concept of massively parallel programming with CUDA, subsequent kernels become much easier to digest.

**Suggested progression:**

1. **Vector addition (a0)** — the classic gateway CUDA example. Plenty of online explanations exist. A CuTeDSL version is provided in this repo. CUDA has a broader ecosystem of YouTube videos and blog posts than CuTeDSL, but CuTeDSL is essentially a Python wrapper over CUDA/PTX so the concepts transfer directly.

2. **Naïve GEMM (a1)** — understand how to perform General Matrix Multiplication in a parallel fashion using one thread per output element (`tidx, tidy, _ = cute.arch.thread_idx()`).

3. **Shared memory GEMM (a2)** — load tiles of A and B into fast on-chip shared memory (SMEM) to reuse data and reduce global memory traffic, producing a significant FLOPS improvement.

4. **WMMA tensor core (b-series)** — use Warp Matrix Multiply-Accumulate instructions to leverage the dedicated tensor core hardware. Lei Mao's blog provides great explanation: https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/. Note: in CuTeDSL/CUTLASS the convention is to wrap PTX instructions directly (lower level than the CUDA C++ `wmma` API).

5. **Hopper TMA + WGMMA (c-series)** — move to Hopper SM90 with Tensor Memory Accelerator and Warp Group MMA, combined with the new async barrier primitives for true pipelined overlap of memory and compute.

6. **Blackwell tcgen05 (d-series)** — the next generation matrix instruction on SM100/SM120.

</details>

<details>
<summary><strong>2. CuTeDSL Fundamentals</strong></summary>

### Linear Indexing

When performing matrix multiplication we write `A[i, j] * B[j, k] = C[i, k]`, which uses 2-D indexing. Physical GPU memory is flat (1-D), so under the hood this multi-dimensional index is collapsed to a single pointer offset. In CUDA kernels we pass only a pointer to the first element of each memory block — no copies are made. CuTe's **Layout** abstraction makes this index arithmetic explicit and composable.

### Host Code vs. Device Code

| Decorator | Role |
|---|---|
| `@cute.jit` | Host-side entry point — calls into device code |
| `@cute.kernel` | Device-side kernel — runs on the GPU |

Compilation can be done in **JIT** (just-in-time, compiled on first call) or **AOT** (ahead-of-time, compiled once and cached) mode.

### Interfacing with PyTorch

PyTorch tensors on the GPU are simply pointers into VRAM with metadata. We use **DLPack** to hand these pointers to our custom CuTeDSL kernels without copying data. In classic CUDA C++ you need glue code (see `cuda/glue_code.cu`) to bridge PyTorch and the kernel; CuTeDSL removes this boilerplate via MLIR/NVVM, resulting in faster compilation.

### Layout

Arguably the most important concept in CUTLASS CuTe / CuTeDSL. A `Layout` pairs a **shape** (extents) with a **stride** (step sizes in linear memory). Layouts compose: you can tile, partition, and swizzle them algebraically. Key API surface:

- `cute.make_layout(shape, stride)` — construct a layout
- `cute.idx2crd(idx, shape)` — unflatten a linear index to coordinates
- `cute.crd2idx(coord, layout)` — flatten coordinates back to a linear index
- **TV Layout** — maps (Thread, Value/Register) pairs to (M, N) coordinates for tensor core fragments (see `z1_tv2mn.py`)
- **Swizzle-composed Layout** — eliminates shared memory bank conflicts by XOR-permuting the address mapping (see `a2_smem_cuda_swizzled.py`, `c3_wgmma_tma_specialized_pipeline.py`)

### Shared Memory

Shared memory (SMEM) is an on-chip scratchpad shared by all threads in a block — roughly 100× lower latency than global memory. Allocation in CuTeDSL:

```python
smem_buf = cute.shared_memory(dtype, shape, alignment=128)
```

Threads cooperatively load tiles from global memory into SMEM, synchronize with `cute.arch.syncthreads()` (or barriers for async), then each thread reads from SMEM rather than GMEM.

### Copy Atom

A **Copy Atom** describes a single hardware copy instruction (e.g., `cp.async`, TMA, `ldmatrix`). A **Tiled Copy Atom** tiles this atom across all threads in a block to fill a shared memory tile efficiently.

| Atom type | Notes |
|---|---|
| Universal Copy Atom | Generic register ↔ register or GMEM → register copy |
| `cp.async` atom | Async GMEM → SMEM without going through registers |
| TMA Copy Atom | Hardware-accelerated GMEM → SMEM, Hopper+ only |
| `ldmatrix` atom | SMEM → register in the layout expected by tensor core |

### MMA Atom

An **MMA Atom** wraps a single hardware matrix-multiply instruction. A **Tiled MMA Atom** composes multiple atoms and threads into the larger tile shape your kernel needs.

| Atom type | Notes |
|---|---|
| WMMA Atom | `wmma` warp-level MMA, Volta+ |
| WGMMA Atom | Warp-group async MMA, Hopper SM90 only |
| tcgen05 Atom | Blackwell SM100/SM120 tensor core instruction |

</details>

<details>
<summary><strong>3. TV Layout for Thread-Register → MN Coordinate Mapping and Layer Fusion</strong></summary>

In CUDA optimization, one strategy is to fuse multiple operations — for example, performing GEMM and immediately applying an element-wise transformation on the accumulator — without ever materializing the large output tensor to global memory. This requires knowing exactly which thread holds which output element.

For example, in a custom RNN kernel that achieved 90–110× speedup over PyTorch, the key insight was to avoid materializing the large matC, then performing a GEMV (matrix-vector multiplication) on it afterwards. Instead, the GEMV was fused directly into the accumulator registers. This requires mapping each thread's accumulator registers to logical `(m, n)` coordinates.

By reading the PTX documentation one can derive this mapping manually with modulo and integer division (see the C++ `tv2mn` template in `README.md`). CuTe provides this formula automatically through the **TV Layout** of the MMA atom.

Printing the atom and tiled MMA for the warp-level `F16/BF16 → F32` instruction (script `b2_wmma_smem.py`) gives:

```
TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
```

- `(4,8)` → 32 threads (one warp)
- `(2,2)` → 4 registers per thread for the accumulator fragment

After tiling this atom 4×4 (512 threads total), the tiled TV layout becomes:

```
tv_layout_C_tiled:
((4,8,4,4),((2,2),(1,2))):((128,1,16,512),((64,8),(0,2048)))
```

To loop over each accumulator register and recover its logical `(m, n)` coordinate:

```python
for reg_idx in range(cute.size(tCrC_out)):
    coord = cute.idx2crd((tid, reg_idx), tv_layout_C_tiled.shape)
    mn_flat = cute.crd2idx(coord, tv_layout_C_tiled)
    m, n = cute.idx2crd(mn_flat, fragC_layout.shape)
```

For example, with `tid=11` and `reg_idx=3`, `coord` is `((3,2,0,0),((1,1),(0,0)))`, `mn_flat` is `1547`, and the resulting logical coordinate is `m=11, n=24`. See `z1_tv2mn.py` for a runnable demo.

This same "do not materialize" pattern underpins the **Implicit GEMM** algorithm, where the `im2col` matrix is computed on the fly tile-by-tile and loaded into SMEM, allowing convolution to use tensor cores efficiently.

</details>

<details>
<summary><strong>4. Tensor Memory Accelerator (TMA)</strong></summary>

Hopper (SM90) and above provides a dedicated hardware unit called the **Tensor Memory Accelerator (TMA)**. Without TMA the data flow for loading a tile is:

```
GMEM → registers → SMEM
```

This consumes register file capacity. TMA short-circuits this:

```
GMEM → SMEM   (direct, asynchronous, no register allocation)
```

The issuing thread returns immediately after launching the TMA copy; the hardware writes the data into shared memory in the background and signals completion via a **barrier** (see Pipeline section for details). This leaves more registers free for computation.

TMA also handles multi-dimensional address calculations, stride, and boundary clamping in hardware, removing that logic from the kernel. Example `c1_wgmma_tma_load_store.py` shows TMA as a drop-in replacement for manual `cp.async` SMEM loading, yielding a clean speedup while remaining portable to SM90, SM100, and SM120.

Key API:
```python
tma_atom = cute.nvgpu.cpasync.TmaLoad(...)
cute.copy(tma_atom, gmem_src, smem_dst, tma_bar_ptr=barrier)
```

</details>

<details>
<summary><strong>5. Warp Group Matrix Multiplication (WGMMA)</strong></summary>

Hopper also introduces **WGMMA** — a faster tensor core instruction that requires a full **warpgroup** (4 warps = 128 threads) to issue together. Key properties:

- **Asynchronous**: `cute.gemm(tiled_mma, ...)` returns immediately while the tensor cores compute in the background, reading operands directly from **shared memory** (not registers).
- **Higher throughput** than the register-based `mma` / `wmma` instructions used on earlier GPUs.
- **Requires new synchronization primitives** to track when the hardware has finished reading from SMEM before that stage can be released back to the producer.

The three WGMMA fence/commit/wait primitives:

| Primitive | Purpose |
|---|---|
| `warpgroup.fence()` | Ensures prior memory ops are ordered before WGMMA issues |
| `warpgroup.commit_group()` | Seals all WGMMA instructions issued since the last commit into one group |
| `warpgroup.wait_group(N)` | Blocks until at most N committed groups are still in-flight |

For example, if `BK=64` and `WGMMA_K=16`, we loop 4 times and then call `commit_group()` to group all 4 WGMMA calls into one trackable unit. `wait_group(0)` then blocks until that group finishes, ensuring SMEM is safe to reuse.

</details>

<details>
<summary><strong>6. Asynchronous Pipeline: PipelineAsync vs PipelineTmaAsync</strong></summary>

Hopper's new barrier primitives allow us to overlap memory transactions and computation. CuTeDSL exposes this via `PipelineAsync` and `PipelineTmaAsync`.

### Warp Specialization

We split work by **role** rather than by data:

- **Producer warps** — handle memory (GMEM → SMEM)
- **Consumer warps** — handle computation (MMA on registers or WGMMA from SMEM)

Different warps can take entirely different code paths with zero divergence penalty because they are independently scheduled by the GPU. This is called **warp specialization**.

```python
if warp_group_idx == 0:    # Producer warps — handle memory
    ...
if warp_group_idx == 1:    # Consumer warps — handle computation
    ...
```

### Pipeline Communication via Barriers in Shared Memory

Producer and consumer warps communicate via **`mbarrier`** objects stored in shared memory (visible to all threads in a block). Each pipeline stage gets its own barrier, organized as a **circular buffer** — after the last stage we wrap back to stage 0.

Each barrier tracks a **phase** that alternates between even and odd. The two race conditions that barriers prevent:

- **Producer overwrite** — producer `acquire` blocks until the consumer has released that stage (data fully consumed).
- **Consumer underread** — consumer `wait` blocks until the producer has committed to that stage (data fully written).

With `S` stages the producer can run up to `S` iterations ahead of the consumer, hiding memory latency behind computation.

### PipelineAsync (Synchronous Writes + Synchronous MMA)

Used when the producer writes to SMEM via regular thread stores (synchronous) and the consumer uses register-based MMA (synchronous). The only concern is preventing race conditions between the two warpgroups.

```python
# Setup
pipeline = PipelineAsync.create(
    num_stages=S,
    producer_group=CooperativeGroup(Agent.Thread, 128),
    consumer_group=CooperativeGroup(Agent.Thread, 128),
    barrier_storage=mbar_ptr,
)
producer, consumer = pipeline.make_participants()

# Producer
for k in range(K):
    handle = producer.acquire_and_advance()   # wait for stage to be free
    smem[handle.index] = data[k]              # threads write to SMEM synchronously
    handle.commit()                           # signal "data ready"
producer.tail()

# Consumer
for k in range(K):
    handle = consumer.wait_and_advance()      # wait for "data ready"
    result += smem[handle.index]              # threads read from SMEM / issue MMA
    handle.release()                          # signal "stage free"
```

### PipelineTmaAsync (Async TMA Loads + Async WGMMA)

Used on Hopper when both the producer (TMA) and consumer (WGMMA) are asynchronous. Two complementary mechanisms track completion:

- **TMA completion**: tracked via **transaction byte counting** on barriers. The `tx_count` parameter tells the pipeline how many bytes to expect per stage. TMA hardware automatically decrements the barrier's counter as bytes land in SMEM — `producer_commit()` is a NOP.
- **WGMMA completion**: tracked via `commit_group()` / `wait_group()` as described above.

Because the consumer must wait for WGMMA to finish reading before releasing a stage, it uses **two separate state trackers**: `consumer_read_state` and `consumer_release_state`.

The producer uses a **prefetch phase** to fill all `S` pipeline stages before the steady-state loop, maximizing overlap:

```
Stage S lifecycle
─────────────────
Producer (TMA warp)                  Consumer (MMA warps)
      │                                     │
      ├─ producer_acquire(S)                │
      ├─ TMA copy A/B → smem[S]            │
      │   (tied to barrier via tma_bar_ptr) │
      ├─ producer_commit(S) [NOP]           │
      │    TMA hw decrements tx_count       │
      │         tx_count hits 0 ──────►     ├─ consumer_wait(S) unblocks
      │                                     ├─ warpgroup.fence()
      │                                     ├─ WGMMA reads smem[S] (async)
      │                                     ├─ warpgroup.commit_group() → G_k
      │                                     ├─ warpgroup.wait_group(0)
      │  ◄──────────────────────────────────├─ consumer_release(S)
```

### What About SM120 (Blackwell RTX)?

SM120 has TMA but **no WGMMA**. It uses register-based MMA (`ldmatrix` + `mma`). Key differences from SM90:

- `ldmatrix` and `mma` are **synchronous** — no `fence`/`commit_group`/`wait_group` needed.
- Only **one consumer state tracker** needed — the stage can be released as soon as `ldmatrix` finishes (MMA never reads SMEM).
- `consumer_release` happens **inside** the k_block loop at the boundary after the last `ldmatrix` of each stage.
- **Double-buffering** the `ldmatrix` for the next k_block with MMA of the current k_block hides `ldmatrix` latency — classic software pipelining.

See [Junkai Wu's dense_gemm SM120 example](https://github.com/NVIDIA/cutlass/blob/main/python/examples/blackwell_rtx/dense_gemm.py) for the full implementation.

</details>

<details>
<summary><strong>7. Profiling Async Pipelines with Probing</strong></summary>

To verify that our async pipeline is actually overlapping memory and compute, we need timing measurements inside the kernel. Taking inspiration from [gau-nernst's blog post](https://gau-nernst.github.io/tcgen05/), the probing technique uses **inline PTX** to call the `globaltimer` instruction, which returns the current GPU clock in nanoseconds:

```python
# Inline PTX to read GPU clock
clock = cute.arch.read_clock()   # wraps: mov.u64 %0, %globaltimer;
```

Each warp records timestamps at the start and end of its producer (TMA load) or consumer (WGMMA / MMA) work. The timestamps are written to a small global memory buffer, then post-processed on the CPU and visualized on [Perfetto](https://ui.perfetto.dev).

**Important caveats about the profiling kernel (`a2_smem_pipeline_profile.py`, `c3_wgmma_tma_specialized_pipeline_profile.py`):**

- Only **2 warps per block** are launched — very low occupancy by design so the timeline is readable.
- The SMEM loading is the main bottleneck; it can be accelerated with vectorized loads or TMA.
- The MMA completes very quickly (low Arithmetic Intensity = FLOPS / bytes transferred), so the pipeline profile shows a memory-bound workload.

**Profile of PipelineAsync (a2):**

The figure below shows producer (SMEM load) and consumer (MMA) timelines per warp. With only 2 warps there is limited overlap, but the barrier handoff between producer and consumer is clearly visible.

![PipelineAsync profile](./assets/a2_pipeline_profile.png)

**Profile of PipelineTmaAsync (c3, 64×128×64 tile):**

Running `c3_wgmma_tma_specialized_pipeline_profile.py` and visualizing on Perfetto shows the first 3 prefetched TMA loads (filling the pipeline), followed by well-overlapped WGMMA compute and TMA loads in the steady state. Launch overhead is still visible at the start.

![PipelineTmaAsync profile](./assets/c3_pipeline_profile_64x128x64.png)

</details>

<details>
<summary><strong>8. Blackwell tcgen05 Matrix Multiplication</strong></summary>

Blackwell (SM100) introduces **tcgen05**, a new generation of tensor core instruction designed for the new architecture. Like WGMMA on Hopper, it requires multiple warps working together and is issued asynchronously. CuTeDSL exposes it through the `tcgen05` MMA atom.

Script `d1_tcgen05_tma.py` provides an example of tcgen05 combined with TMA loads.

> Full documentation coming soon.

</details>

## Job Submission

**Using Ray:**
```bash
pip install ray
# Assume a Ray cluster is already running

ray job submit \
    --address 'http://localhost:8265' \
    --working-dir . \
    --runtime-env-json='{"pip":"./requirements.txt"}' \
    -- python submit_ray.py
```

**Using Modal:**
```bash
pip install modal
python3 -m modal setup

modal run submit_modal.py
```

## Reference

1. https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL
2. https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
3. https://research.colfax-intl.com/tutorial-hopper-tma/
4. https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
5. https://gau-nernst.github.io/tcgen05/
6. https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
7. https://github.com/LeiWang1999/CPPTorchExecutable
8. https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html