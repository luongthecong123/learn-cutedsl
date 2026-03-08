# Learn CuTeDSL

**CuTeDSL** is a Python-embedded domain-specific language that wraps CUDA/PTX. It gives you quality-of-life APIs for writing efficient GPU kernels — think Layout algebra, Swizzle, Copy Atoms, MMA Atoms, TMA descriptors — while staying close enough to the hardware to hit speed-of-light (SoL) performance. Kernels integrate seamlessly with PyTorch via DLPack and compile fast thanks to an MLIR/NVVM backend.

This repo adds hardware features and optimization techniques gradually, measuring the FLOPS gain at each step so you see exactly what you are buying. Beyond the kernels, the repo also serves as a **reference for commonly used CuTeDSL APIs**. Each concept — Layout arithmetic, Swizzle composition, TV-to-MN coordinate mapping, `mbarrier` synchronization — is isolated and explained with minimal surrounding noise, making it easy to lift a pattern into your own code or feed it as context to an LLM. Here, you can find kernels for Ampere (SM80), Hopper (SM90), Blackwell (SM100) and Blackwell RTX (SM120).

**Why CuTeDSL over CUTLASS C++?**
- No template metaprogramming maze, much faster iteration and easier to get started
- Python quality of life: Pylance, Intellisense, preferred language by AIs,...
- Same low-level control: you can drop to raw PTX whenever you need it
- JIT compilation (`@cute.jit`) or AOT (`cute.compile`) with PyTorch zero-copy interop

**Known rough edges:**
- Many APIs overlap in purpose — the versatility that makes it powerful can also be confusing
- Documentation is sparse; official examples tend to go straight for SoL complexity
- This repo exists partly to fill that gap: one concept per script, explained step by step

---

**What lies ahead in this document:**

Section 1 walks through the optimization progression with measured FLOPS gains at each step. Section 2 explains the core CuTeDSL APIs you will encounter everywhere: Layout, Shared Memory, Copy Atoms, and MMA Atoms. Section 3 dives into TV Layout for mapping thread registers to `(m, n)` coordinates, enabling layer fusion without materializing large tensors. Section 4 covers TMA and WGMMA together — the two Hopper hardware features that power the async pipeline. Section 5 builds a full async warp-specialized pipeline, contrasting `PipelineAsync` and `PipelineTmaAsync` and the SM120 fallback. Section 6 shows how to profile the pipeline with inline PTX clock reads. Section 7 introduces the Blackwell tcgen05 tensor core instruction.

## Frequently used APIs explanation 
(Click the arrow to expand section)

<details>
<summary><strong>1. Learning Curve and FLOPS Gain</strong></summary>

CuTeDSL and CUDA in general have a very steep but rewarding learning curve, so don't get frustrated the first time you try it. The best approach is to look at examples, write kernels yourself, and observe the performance speedup — and understand *why* it speeds up. Once you can wrap your head around the concept of massively parallel programming with CUDA, subsequent kernels become much easier to digest.

Suggested progression:

1. **Vector addition (a0)** — the classic gateway CUDA example. Plenty of online explanations exist. A CuTeDSL version is provided in this repo. CUDA has a broader ecosystem of YouTube videos and blog posts than CuTeDSL, but CuTeDSL is essentially a Python wrapper over CUDA/PTX so the concepts transfer directly.

2. **Naïve GEMM (a1)** — understand how to perform General Matrix Multiplication in a parallel fashion using one thread per output element (`tidx, tidy, _ = cute.arch.thread_idx()`).

3. **Shared memory GEMM (a2)** — load tiles of A and B into fast on-chip shared memory (SMEM) to reuse data and reduce global memory traffic, producing a significant FLOPS improvement.

4. **WMMA tensor core (b-series)** — use Warp Matrix Multiply-Accumulate instructions to leverage the dedicated tensor core hardware. Lei Mao's blog provides great explanation: https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/. Note: in CuTeDSL/CUTLASS the convention is to wrap PTX instructions directly (lower level than the CUDA C++ `wmma` API).

5. **Hopper TMA + WGMMA (c-series)** — move to Hopper SM90 with Tensor Memory Accelerator and Warp Group MMA, combined with the new async barrier primitives for true pipelined overlap of memory and compute.

6. **Blackwell tcgen05 (d-series)** — the next generation matrix instruction on SM100/SM120.

**FLOPS progression** (M=N=K=4096, dtype=float16):

| Stage | Script | Architecture | SM90 (H100) | SM100 (B200) | SM120 (RTX Pro 6K) |
|---|---|---|---|---|---|
| Naïve | [`a1`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cute.py) | Any | 0.58 | similar | similar |
| Shared memory | [`a2`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_cuda_like.py) | Any | 7.17 | similar | similar |
| WMMA | [`b2`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py) | Ampere+ (SM80+) | 203.29 | 241.30 | 295.26 |
| WMMA + TMA | [`b5`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b5_wmma_tma_load_store.py) | Hopper+ (SM90+) | 355.71 | 324.37 | 340.16 |
| WMMA + TMA warp-specialized pipeline | [`b7`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b7_wmma_tma_pipeline.py) | Hopper+ (SM90+) | 392.86 | 424.70 | 345.27 |
| WGMMA + TMA | [`c1`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py) | Hopper (SM90) | 532.10 | - | - |
| WGMMA + TMA warp-specialized pipeline | [`c2`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c2_wgmma_tma_specialized_pipeline.py) | Hopper (SM90) | 685.08 | - | - |
| tcgen05 MMA + TMA | [`d1`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d1_tcgen05_tma_umma.py) | Blackwell (SM100) | - | 717.30 | - |
| tcgen05 MMA + TMA warp-specialized pipeline | [`d2`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d2_tcgen05_tma_umma_specialized_pipeline.py) | Blackwell (SM100) | - | 1279.36 | - |
| tcgen05 MMA 2CTA + TMA warp-specialized pipeline | [`d3`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d3_tcgen05_tma_umma_2cta_specialized_pipeline.py) | Blackwell (SM100) | - | 1366.63 | - |

We can improve further by using techniques such as Persistent kernel to overlap epilogue with the start of the next tile, TMA Multi-cast, TMEM staging in Blackwell,... to reach Speed of Light (CuBLAS) which is around 720 TFLOPS for H100 and 1500 TFLOPS for B200.

</details>

<details>
<summary><strong>2. CuTeDSL Fundamentals</strong></summary>

The building blocks you will encounter in every CuTeDSL kernel: how host and device code are structured, how to interface with PyTorch, and the core abstractions — Layout, Shared Memory, Copy Atoms, and MMA Atoms.

### 2.1. Host Code vs. Device Code

```python
@cute.jit   # host-side entry point
def my_launcher(mA: cute.Tensor, ...):
    my_kernel(...).launch(grid=[...], block=[...])

@cute.kernel   # device-side kernel — runs on the GPU
def my_kernel(...):
    ...
```

- **[`@cute.jit`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a0_vector_addition.py#L11)** marks the host function. It triggers JIT compilation when first called and handles argument marshalling. You can also call `cute.compile(fn, *sample_args)` explicitly for **AOT** (ahead-of-time) compilation, which compiles once and caches the result for reuse — as done in the [`main()` of every script](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a0_vector_addition.py#L47).

- **[`@cute.kernel`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a0_vector_addition.py#L21)** marks the GPU kernel. Inside it, thread/block coordinates are queried with:
  ```python
  bidx, bidy, _ = cute.arch.block_idx()
  bdimx, bdimy, _ = cute.arch.block_dim()
  tidx, tidy, _ = cute.arch.thread_idx()
  ```

### 2.2. Interfacing with PyTorch

#### 2.2.1. Traditional CUDA C++ Extension Workflow

When writing custom CUDA kernels without CuTeDSL, you need glue code to bridge between PyTorch (Python) and the raw CUDA kernel. This involves three layers:

1. **The CUDA kernel** ([`glue_code.cu`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cuda/glue_code.cu)) — implements the GPU computation, receives raw `void*` pointers and dispatches to the chosen kernel variant:
   ```cpp
   // cuda/glue_code.cu
   void gemm_cuda(void* A, void* B, void* C, int M, int N, int K, int option) {
       switch (option) {
           case 0: naive::gemm_launcher(A, B, C, M, N, K); break;
           case 4: wmma_smem::gemm_launcher(A, B, C, M, N, K); break;
           // ...
       }
   }
   ```

2. **The Torch binding** ([`glue_code.cpp`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cuda/glue_code.cpp)) — unwraps `at::Tensor` objects into raw pointers and exposes the function to Python via `pybind11`:
   ```cpp
   // cuda/glue_code.cpp
   void gemm(const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
             int M, int N, int K, int option) {
       gemm_cuda(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K, option);
   }
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("gemm", &gemm, "GEMM (CUDA)", ...);
   }
   ```

3. **The Python loader** ([`gemm_cuda.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cuda/gemm_cuda.py)) — uses `torch.utils.cpp_extension.load` to JIT-compile and import the extension:
   ```python
   # cuda/gemm_cuda.py
   gemm_cuda_compiled = load(
       name='gemm',
       sources=[f"{dir_path}/glue_code.cu", f"{dir_path}/glue_code.cpp"],
       extra_cuda_cflags=["-arch=sm_90", "-lineinfo"],
       build_directory=build_dir,
   )
   ```

Why are the CUDA kernel and Torch binding split into two separate source files? Separating `glue_code.cu` (CUDA-only code) from `glue_code.cpp` (Torch/pybind11 binding) prevents Ninja from compiling to large binary. Credit to the profiling experiment from [Lei Wang](https://github.com/LeiWang1999/CPPTorchExecutable).

#### 2.2.2. CuTeDSL: No Glue Code Needed

With CuTeDSL, all of the above boilerplate disappears. PyTorch tensors on the GPU are pointers into VRAM with metadata. [`from_dlpack`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a0_vector_addition.py#L43) uses the DLPack protocol to share that pointer with CuTeDSL without copying data:

```python
from cutlass.cute.runtime import from_dlpack

A_ = from_dlpack(A, assumed_align=16)   # wrap a torch.Tensor as a cute.Tensor
```

CuTeDSL handles the bridge via MLIR/NVVM — no `pybind11` bindings, no `.cu`/`.cpp` glue files, no manual `data_ptr()` casting. Compilation is also faster thanks to an incremental JIT cache.

### 2.3. Threads and Blocks — the CUDA Execution Model

When you call `.launch(grid=[Gx, Gy, 1], block=[Bx, By, 1])`, the GPU spawns `Gx x Gy` blocks (also called CTAs — Cooperative Thread Arrays), each containing `Bx x By` threads. Every thread runs the same kernel function but gets a different `(block_idx, thread_idx)` combination, so each one can compute its own unique slice of the output. The key identities are:

```
global_row = block_idx.y * block_dim.y + thread_idx.y
global_col = block_idx.x * block_dim.x + thread_idx.x
```

Threads within the same block can communicate via shared memory and synchronize with `sync_threads()`. Threads in different blocks cannot directly synchronize — they run independently. Understanding why the work is partitioned this way is the first mental hurdle in CUDA; if it's new to you, working through [`a0_vector_addition.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a0_vector_addition.py) and [`a1_naive_cuda_like.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cuda_like.py) yourself — tracing exactly which thread handles which element — is the most effective way to build that intuition.

### 2.4. Layout and Linear indexing

When performing matrix multiplication we write `A[i, j] * B[j, k] = C[i, k]`, which uses 2-D indexing. Physical GPU memory is flat (1-D), so every element access is ultimately just a pointer plus an integer offset: `*(base_ptr + offset)`. A multi-dimensional index `(row, col)` must be collapsed to that single offset before the hardware can fetch the value. For a row-major matrix of width `K`, the formula is `offset = row * K + col` — "skip `row` full rows of `K` elements, then `col` more". The matrix itself is never copied or rearranged in memory; only this arithmetic changes depending on layout. In traditional CUDA C++ you compute these offsets by hand and pass raw pointers. CuTe's Layout abstraction — a pairing of shape and stride — encodes the formula once and applies it automatically every time you index into a tensor, making the arithmetic explicit, verifiable, and composable across tiling levels.

Arguably the most important concept in CUTLASS CuTe / CuTeDSL. A `Layout` pairs a **shape** (extents in each dimension) with a **stride** (step size in linear memory per dimension). The key formula is:

$$\text{offset} = \sum_i \text{coord}_i \times \text{stride}_i$$

For example, a row-major matrix `A` of shape `(M, K)` has `stride=(K, 1)`. Accessing element at row `r`, column `c` gives offset `r * K + c`. In memory this just means: "skip `r` full rows of `K` elements, then `c` more."

#### 2.4.1. 1-D case: vector addition ([`a0_vector_addition.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a0_vector_addition.py))

The simplest example — each thread computes a single linear index and indexes directly:

```python
# a0_vector_addition.py L29-L33
bidx, _, _ = cute.arch.block_idx()
bdimx, _, _ = cute.arch.block_dim()
tidx, _, _ = cute.arch.thread_idx()

i = bidx * bdimx + tidx   # global thread index = linear memory offset
gC[i] = gA[i] + gB[i]    # stride=1 → offset == index
```

Here the tensor has `shape=(N,), stride=(1,)`, so the offset formula reduces to just `i * 1 = i` — no multi-dimensional arithmetic needed.

#### 2.4.2. 2-D case: naïve GEMM ([`a1_naive_cuda_like.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cuda_like.py))

In a naïve GEMM, each thread owns one output element `C[row, col]`. The 2-D index must be converted to a linear memory offset:

```python
# a1_naive_cuda_like.py L37-L40
for k in range(K):
    acc += cute.Float32(gA[bidy * bdimy + tidy, k]) * cute.Float32(gB[bidx * bdimx + tidx, k])

gC[bidy * bdimy + tidy, bidx * bdimx + tidx] = cute.Float16(acc)
```

When you write `gA[row, col]`, CuTeDSL uses the Layout of `gA` to compute the pointer offset automatically:

$$\text{offset}_A = \text{row} \times K + \text{col} \times 1$$

For `A` of shape `(M, K)` stored row-major, `stride=(K, 1)`. So `gA[row, k]` reads from `base_ptr + row * K + k`. No manual offset arithmetic is needed — the Layout carries the stride information.

#### 2.4.3. Tiling: going from elements to blocks ([`a1_naive_cute.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cute.py))

Rather than computing per-element offsets manually, `cute.local_tile` slices the global tensor into CTA-sized tiles. The resulting sub-tensor still carries a Layout, so all further indexing remains clean:

```python
# a1_naive_cute.py L29-L31
tiler = (BM, BN, BK)
coord = (bidx, bidy, None)   # None = "keep this dimension" (K-iteration axis)

gA_tile = cute.local_tile(gA, tiler=tiler, coord=coord, proj=(1, None, 1))
gB_tile = cute.local_tile(gB, tiler=tiler, coord=coord, proj=(None, 1, 1))
gC_tile = cute.local_tile(gC, tiler=tiler, coord=coord, proj=(1, 1, None))
```

`proj` masks out which dimensions are "owned" by the CTA. For `gA_tile`, `proj=(1, None, 1)` means: fix the M and K tile coordinates to `(bidx, _, bidy)` respectively, and leave the inner K-iteration axis free so the kernel can loop over it. The pointer arithmetic for the tile origin is:

$$\text{base\_A\_tile} = \text{base\_A} + \text{bidx} \times BM \times K + \text{bidy} \times BK$$

This is computed once by `local_tile` and encoded into the sub-tensor's Layout; subsequent indexing within the tile uses only small local coordinates.

**Construction:**
```python
# cute.make_layout(shape, stride)
layout = cute.make_layout((BM, BK), stride=(BK + PAD, 1))
```
*Used in [`a2_smem_cuda_like.py` L32](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_cuda_like.py#L32) to define padded shared memory layouts that avoid bank conflicts.*

**Index arithmetic:**
```python
# Flatten multi-dim coordinate → linear offset
offset = cute.crd2idx(coord, layout)

# Unflatten linear index → coordinate
coord = cute.idx2crd(idx, shape)
```
*Used in [`z1_tv2mn.py` L16–L25](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/z1_tv2mn.py#L16) to decode thread/register indices into logical (m, n) output coordinates.*

**Tiling a global tensor into CTA-sized tiles:**
```python
# cute.local_tile(tensor, tiler, coord, proj)
# Returns the sub-tile of `tensor` that this CTA is responsible for.
gA_tile = cute.local_tile(gA, tiler=tiler, coord=coord, proj=(1, None, 1))
```
*Used in [`a1_naive_cute.py` L30](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cute.py#L30). `proj` masks out dimensions so the tile retains the K-iteration axis.*

**Swizzle-composed Layout:**

Bank conflicts occur when multiple threads in a warp access different addresses that map to the same shared memory bank. A **Swizzle** applies an XOR permutation to the row address, spreading accesses across banks. `make_swizzle(B, M, S)` defines the XOR pattern via three bit-field parameters.

```python
# cute.make_swizzle(B, M, S) — XOR permutation parameters
# cute.make_composed_layout(inner=swizzle, offset=0, outer=base_layout)
layout_sA_swizzled = cute.make_composed_layout(
    inner=cute.make_swizzle(3, 4, 3),
    offset=0,
    outer=layout_sA
)
```
*Used in [`a2_smem_cuda_swizzled.py` L35–L39](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_cuda_swizzled.py#L35). See [`z0_swizzle.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/z0_swizzle.py#L7) for a standalone swizzle demo.*


### 2.5. Shared Memory

Shared memory (SMEM) is an on-chip scratchpad shared by all threads in a block — roughly 100x lower latency than global memory. In CuTeDSL, SMEM is allocated with:

```python
# cutlass.utils.SmemAllocator — manages a contiguous SMEM buffer
# .allocate_tensor(dtype, layout, alignment, init) → cute.Tensor backed by SMEM
allocator = cutlass.utils.SmemAllocator()
sA = allocator.allocate_tensor(cutlass.Float16, layout_sA, 16, None)
```
*Used in [`a2_smem_cuda_like.py` L31–L35](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_cuda_like.py#L31).*

After cooperatively loading a tile from global memory into SMEM, threads must synchronize before reading:

```python
cute.arch.sync_threads()   # equivalent to __syncthreads() in CUDA C++
```
*Used in [`a2_smem_cuda_like.py` L58](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_cuda_like.py#L58). For async pipelines (TMA), `mbarrier`-based synchronization replaces this — see Section 6.*


### 2.6. Copy Atom

A **Copy Atom** is the smallest unit of a hardware copy operation. A **Tiled Copy** wraps the atom and distributes the work across all threads in a CTA to fill or drain a tile efficiently.

```python
# cute.make_copy_atom(op, dtype, num_bits_per_copy) → CopyAtom
# cute.make_tiled_copy(atom, thread_layout, value_layout) → TiledCopy
atom_copy_A = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    mA.element_type,
    num_bits_per_copy=mA.element_type.width * num_vectorized   # vectorized load
)
```
*Used in [`b2_wmma_smem.py` L63–L67](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L63) for GMEM → SMEM copies with vectorization.*

Once you have a `TiledCopy`, you slice it per-thread and partition source/destination tensors:

```python
thr_copy = tiled_copy.get_slice(tid)        # per-thread view of the tiled copy
tAsA = thr_copy.partition_S(sA)             # source partition (this thread's slice of sA)
tAgA = thr_copy.partition_D(gA)             # destination partition (this thread's slice of gA)
cute.copy(tiled_copy, tAgA, tAsA)           # every thread copies its own subtensor
```

> Naming convention — the prefix encodes both the partitioner and the tensor. `tAsA` reads as "partitioning pattern `tA` applied to tensor `sA`". The same partitioner `tA` is applied to both `sA` (shared memory) and `gA` (global memory) to produce `tAsA` and `tAgA`. Because both tensors use the same partitioning pattern, CuTe can assert that corresponding logical elements match across the two tensors, even if their physical data layouts differ. When you write `cute.copy(tiled_copy, tAgA, tAsA)`, you can verify that source and destination are partitioned consistently — a naming convention borrowed from CUTLASS CuTe C++. The prefix letter encodes the memory space: `s` = shared memory, `g` = global memory, `r` = register.

| Copy Op | Direction | Notes |
|---|---|---|
| `CopyUniversalOp` | any | Generic register-to-register or GMEM → register |
| `LdMatrix8x8x16bOp` | SMEM → register | `ldmatrix` instruction; loads data in the exact layout tensor cores expect. Used in [`b2_wmma_smem.py` L156–L161](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L156) |
| `CopyBulkTensorTileG2SOp` | GMEM → SMEM | TMA async load, Hopper+. Used in [`c1_wgmma_tma_load_store.py` L424](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L424) |
| `CopyBulkTensorTileS2GOp` | SMEM → GMEM | TMA async store. Used in [`c1_wgmma_tma_load_store.py` L92](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L92) |


### 2.7. MMA Atom

An **MMA Atom** wraps a single hardware matrix-multiply-accumulate instruction. A **Tiled MMA** tiles this atom across threads and repeats it to cover a larger output tile.

```python
# Define the hardware instruction
mma_op = cute.nvgpu.warp.MmaF16BF16Op(
    ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))

# Tile it: atom_layout_mnk=(4,4,1) means 4x4 warp-level tiles → 512 threads
tiled_mma = cute.make_tiled_mma(
    op_or_atom=mma_op,
    atom_layout_mnk=(4, 4, 1),
    permutation_mnk=(64, 64, 16))
```
*Used in [`b2_wmma_smem.py` L36–L53](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L36).*

Once you have a `TiledMma`, slice it per-thread and partition the operand tiles:

```python
thr_mma = tiled_mma.get_slice(tid)    # per-thread view
tCsA = thr_mma.partition_A(sA)        # A operand partition for this thread (partitioner tC, tensor sA)
tCsB = thr_mma.partition_B(sB)        # B operand partition (partitioner tC, tensor sB)
tCgC = thr_mma.partition_C(gC)        # C accumulator partition (partitioner tC, tensor gC)

tCrC = tiled_mma.make_fragment_C(tCgC)   # allocate register fragment for C
tCrC.fill(cute.Float32(0))

cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)   # issue MMA
```

| MMA Op | GPU | Notes |
|---|---|---|
| `MmaUniversalOp` | all | Scalar FMA tiled to any shape — used for naïve cute GEMM in [`a1_naive_cute.py` L37](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cute.py#L37) |
| `MmaF16BF16Op` (warp) | Ampere+ | `wmma`-style warp-level MMA — used in [`b2_wmma_smem.py` L36](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L36) |
| WGMMA atom | Hopper SM90 | Warp-group async MMA reading operands from SMEM — used in [`c1_wgmma_tma_load_store.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py) |
| tcgen05 atom | Blackwell SM100 | Next-gen tensor core — used in [`d1_tcgen05_tma.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d1_tcgen05_tma.py) |

</details>

<details>
<summary><strong>3. TV Layout for Thread-Register → MN Coordinate Mapping and Layer Fusion</strong></summary>

In CUDA optimization, one strategy is to fuse multiple operations — for example, performing GEMM and immediately applying an element-wise transformation on the accumulator — without ever materializing the large output tensor to global memory. This requires knowing exactly which thread holds which output element.

For example, in my custom RNN kernel that achieved 90–110x speedup over PyTorch [Repo link](https://github.com/chongxi/rnn_train_ring_attractor/blob/main/cpp/kernels/fwd_1loop_tc_idx.cuh#L172), the key insight was to avoid materializing the large matC, and performing a GEMV (matrix-vector multiplication) directly on the fragC on register file. This requires mapping each thread's accumulator registers to logical `(m, n)` coordinates.

By reading the PTX documentation one can derive this mapping manually with modulo and integer division [CUDA C++ Example](https://github.com/chongxi/rnn_train_ring_attractor/blob/main/cpp/kernels/fwd_1loop_tc_idx.cuh#L6). 

```c++
template<int WMMA_M, int WMMA_N, int WMMA_K, typename T>
__device__ void tv2mn(wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& fragC)
{
    /*
    *16x16 FragC Register Layout (4 quadrants, each with 2 registers)
    *Only support m16n16k16 and m32n8k16.
            +---------+---------+
            | r0  r1  | r4  r5  |
    0-7  |         |         |
            | Block 0 | Block 1 |
            +---------+---------+
            | r2  r3  | r6  r7  |
    8-15 |         |         |
            | Block 2 | Block 3 |
            +---------+---------+
            0-7       8-15
    */
    constexpr int NUM_REGBLOCK_ROWS = WMMA_M / 8;
    constexpr int NUM_REGBLOCK_COLS = WMMA_N / 8;
    constexpr int REGS_PER_BLOCK = 2;
    
    size_t threadID_in_warp = threadIdx.x % WARPSIZE;
    size_t groupID_in_warp = threadID_in_warp / 4;
    size_t threadID_in_group = threadID_in_warp % 4;
    
    for (int regBlockRow = 0; regBlockRow < NUM_REGBLOCK_ROWS; ++regBlockRow) {
        for (int regBlockCol = 0; regBlockCol < NUM_REGBLOCK_COLS; ++regBlockCol) {
            for (int i = 0; i < REGS_PER_BLOCK; ++i) {
                int regID = (regBlockRow * REGS_PER_BLOCK) +
                        (regBlockCol * NUM_REGBLOCK_ROWS * REGS_PER_BLOCK) + i;
                size_t m = regBlockRow * 8 + groupID_in_warp;
                size_t n = regBlockCol * 8 + threadID_in_group * 2 + i;
                // use m, n for indexing
            }
        }
    }
}    
```

CUTLASS devs made our life much easier by providing the **TV Layout** (Thread-Value Layout) of the MMA atom.
Printing the atom and tiled MMA for the warp-level `F16/BF16 → F32` instruction (script `b2_wmma_smem.py`) gives:

```
TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
```

- `(4,8)` → 32 threads (one warp)
- `(2,2)` → 4 registers per thread for the accumulator fragment

After tiling this atom 4x4 (512 threads total), the tiled TV layout becomes:

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

This same "do not materialize" pattern underpins the **Implicit GEMM** algorithm, where the `im2col` matrix is computed on the fly tile-by-tile and loaded into SMEM, allowing convolution to use tensor cores efficiently and minimizing round-trip gmem reads and writes.

</details>

<details>
<summary><strong>4. TMA and WGMMA (Hopper)</strong></summary>

Hopper (SM90) introduced two complementary hardware features that together enable the high-throughput async pipeline: **TMA** offloads the memory side and **WGMMA** accelerates the compute side. They are described together because understanding both is necessary to make sense of the pipeline in Section 5.

---

### 4.1. Tensor Memory Accelerator (TMA)

Without TMA the data flow for loading a tile is:

```
GMEM → registers → SMEM
```

This consumes register file capacity. TMA short-circuits this:

```
GMEM → SMEM   (direct, asynchronous, no register allocation)
```

The issuing thread returns immediately after launching the TMA copy; the hardware writes data into shared memory in the background and signals completion via an **mbarrier** (see Section 5 for details). This leaves more registers free for computation. TMA also handles multi-dimensional address calculations, stride, and boundary clamping in hardware, removing that logic from the kernel.

Example `b5` shows TMA as a drop-in replacement for manual SMEM loading, yielding a nice speedup while remaining portable to SM90, SM100, and SM120.

#### 4.1.1. Two-step Process

TMA is always set up in two distinct steps — unlike a regular `TiledCopy` where everything lives in kernel code:

1. **Host side**: Build a `cuTensorMap` descriptor encoding the GMEM tensor's base pointer, shape, strides, and swizzle mode. This is done once, at Python (host) level, and passed to the kernel as `__grid_constant__ const`. The descriptor cannot be modified by the device.
2. **Kernel side**: A single thread issues the copy instruction referencing the descriptor. The hardware does all address arithmetic and predication (out-of-bounds clamping) without any thread-level register involvement.

#### 4.1.2. TMA Load: GMEM → SMEM

```python
# Host: build the TMA atom + "ArithTuple" GMEM coordinate tensor
tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
    cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),  # direction: Global→Shared
    gmem_tensor_a,                                  # source GMEM tensor
    smem_layout,                                    # destination SMEM layout per CTA
    (BM, BK),                                       # tile extents
)

# Kernel: partition then copy
tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(tma_atom_a, cta_crd, cta_layout, sA, gA)
cute.copy(tma_atom_a, tAgA[None, kidx], tAsA[None, 0], tma_bar_ptr=mbar_ptr)
```
*Used in [`c1_wgmma_tma_load_store.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L391).*

The GMEM-side of `tma_partition` returns **ArithTuples**, not raw pointers. These are coordinate objects — the TMA descriptor resolves them to actual addresses. This is how TMA handles OOB predication: it never performs pointer arithmetic that could go out-of-bounds.

mbarrier synchronization protocol (only for TMA load — 4 steps):

| Step | CuTeDSL call | Underlying PTX | Who |
|---|---|---|---|
| Init | `mbarrier_init(mbar_ptr, cnt=1)` | `mbarrier.init.shared.b64` | thread 0 |
| Expect | `mbarrier_expect_tx(mbar_ptr, tx_bytes)` | `mbarrier.arrive.expect_tx.shared::cta.b64` | thread 0 — sets expected byte count |
| Issue | `cute.copy(..., tma_bar_ptr=mbar_ptr)` | `cp.async.bulk.tensor.Nd ...mbarrier::complete_tx::bytes` | warp 0 — TMA hw decrements counter as bytes land |
| Wait | `mbarrier_wait(mbar_ptr, phase=0)` | `mbarrier.try_wait.parity.shared::cta.b64` | **all** threads — sleep until barrier flips phase |

The **phase bit** alternates (0 → 1 → 0 → …) each time the barrier is reused. In `c1` the barrier is re-initialized each K iteration (single-stage). In the multi-stage pipeline (`c2`/`c3`) the pipeline API cycles phases automatically.

```python
# Excerpt from the K-tile loop in c1_wgmma_tma_load_store.py
if tidx == 0:
    cute.arch.mbarrier_init(mbar_ptr, cnt=1)
    cute.arch.mbarrier_init_fence()

cute.arch.sync_threads()   # all threads wait for barrier to finish init on smem, as they will need to access this later

if warp_idx == 0:
    if tidx == 0:
        cute.arch.mbarrier_expect_tx(mbar_ptr, tma_transaction_bytes)
        cute.arch.mbarrier_arrive(mbar_ptr)
    cute.copy(tma_atom_a, tAgA[None, kidx], tAsA[None, 0], tma_bar_ptr=mbar_ptr)
    cute.copy(tma_atom_b, tBgB[None, kidx], tBsB[None, 0], tma_bar_ptr=mbar_ptr)

cute.arch.mbarrier_wait(mbar_ptr, 0)   # all threads block here
```
*[`c1_wgmma_tma_load_store.py` lines 287–305](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L287).*

Descriptor prefetch — at kernel start, warp 0 warms the constant cache holding the descriptor before the K-loop:

```python
if warp_idx == 0:
    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
```

#### 4.1.3. TMA Store: SMEM → GMEM

```python
# Host: build TMA atom for the output tile
tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
    cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),  # direction: Shared→Global
    gmem_tensor_c,
    c_smem_layout,
    (BM, BN),
)

# Kernel: fence BEFORE the copy, then issue from one warp
cute.arch.sync_threads()
cute.arch.fence_proxy("async.shared", space="cta")   # fence.proxy.async.shared::cta
if warp_idx == 0:
    cute.copy(tma_atom_c, tCsC, tCgC_store)
```
*[`c1_wgmma_tma_load_store.py` lines 354–379](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L354).*

Why a proxy fence and not an mbarrier?
TMA store operates in the async proxy (with respect to the CTA). Without `fence.proxy.async.shared::cta`, the TMA hardware could read stale values from SMEM.

The fence goes before the store (opposite of mbarrier for loads, which waits after).

If the SMEM buffer for `sC` needs to be reused (e.g., multi-wave pipeline), add arrive/wait around the store:

```python
# cute.arch.tma_store_arrive()   # commit the store into a bulk group
# cute.arch.tma_store_wait(0)    # wait until 0 groups remain in-flight
```

#### 4.1.4. SMEM Layout Requirements and Stride Constraints

The SMEM layouts for A and B use **swizzled layouts** to avoid bank conflicts and satisfy the memory format that WGMMA instructions expect. These are built automatically by `sm90_utils.make_smem_layout_a/b`.

For the GMEM tensor, TMA requires:
- One dimension must be **stride-1** (contiguous).
- All other strides must be **multiples of 16 bytes**. For `float16` this means the leading dimension must be divisible by 8 (e.g., `K % 8 == 0` for a row-major A of shape `(M, K)`).

This is why inputs are created with `assumed_align=16`:
```python
A_ = from_dlpack(A, assumed_align=16)
```

---

### 4.2. Warp Group Matrix Multiplication (WGMMA)

**WGMMA** (`wgmma.mma_async`) is Hopper's tensor core instruction. It supersedes the warp-level `wmma` instruction used on Ampere and earlier. Key differences from `wmma`:

- **Warpgroup-wide**: requires all **128 threads** in a warpgroup (4 contiguous warps whose first warp rank is a multiple of 4) to issue the instruction collectively.
- **Asynchronous**: `cute.gemm(tiled_mma, ...)` returns immediately; the tensor cores continue computing in the background, reading operands directly from **SMEM via matrix descriptors** (not registers), and writing to register-backed accumulators.
- **B operand always comes from SMEM**. A can come from SMEM (`SS` mode) or registers (`RS` mode). In `c1`, both A and B are sourced from SMEM (SS mode).
- **Higher throughput** than `wmma` — responsible for the 2.3x jump from `b5` to `c1` in the FLOPS table.

#### 4.2.1. WGMMA Atom Shape Constraints

The underlying PTX instruction is `wgmma.mma_async.sync.aligned.M64xNxK`. The tile shape is constrained:
- **M is always 64** — one WGMMA atom always covers 64 rows.
- **K x sizeof(dtype) = 32 bytes** — so for `float16`, K = 16; for `float8`, K = 32.
- **N is a multiple of 8, from 8 to 256**.

This imposes the assertions in `c1`:
```python
assert self.BM % 64 == 0, "bM must be divisible by 64 for WGMMA"
assert self.BN % 64 == 0, "bN must be divisible by 64 for WGMMA"
```

#### 4.2.2. TiledMMA Construction

In `c1`, the `TiledMMA` is built on the host via a CuTeDSL helper:

```python
# c1_wgmma_tma_load_store.py — host (__call__)
self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
    self.a_dtype,
    self.b_dtype,
    self.a_layout.sm90_mma_major_mode(),   # MN-major or K-major (matches memory layout)
    self.b_layout.sm90_mma_major_mode(),
    self.acc_dtype,                         # Float32 accumulator
    self.atom_layout_mnk,                   # e.g. (1,1,1) = one warpgroup
    tiler_mn=(64, self.tile_shape_mnk[1]), # the MxN output tile per warpgroup
)
```
*[`c1_wgmma_tma_load_store.py` L125–L133](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L125).*

`atom_layout_mnk` tiles the single 64xN atom across multiple warpgroups. For `BM=64`, it is `(1,1,1)` (one warpgroup). For `BM=128`, setting it to `(2,1,1)` assigns 2 warpgroups to split the M-dimension, each computing a 64xN subtile independently. The total thread count becomes:

```python
self.threads_per_cta = math.prod(self.atom_layout_mnk) * 128  # warpgroups x 128 threads
```

For `float16`, both MN-major and K-major are supported. For other dtypes, only K-major is allowed (no transpose support in PTX for non-16-bit).

#### 4.2.3. Per-Thread Slicing and Fragments

On device, slice the `TiledMMA` per warpgroup (not per thread as with `wmma`):

```python
# c1_wgmma_tma_load_store.py — kernel
warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
warp_group_thread_layout = cute.make_layout(
    self.mma_warp_groups, stride=self.num_threads_per_warp_group
)
thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

tCsA = thr_mma.partition_A(sA)   # shape: (MMA_ATOM, MMA_M, MMA_K, stages)
tCsB = thr_mma.partition_B(sB)   # shape: (MMA_ATOM, MMA_N, MMA_K, stages)
tCrA = tiled_mma.make_fragment_A(tCsA)
tCrB = tiled_mma.make_fragment_B(tCsB)
```

`tCrA` and `tCrB` are not register tensors — they are `GMMA::DescriptorIterator` objects. Internally each is a 64-bit matrix descriptor held in registers that points into SMEM. The descriptor encodes the SMEM base address, leading byte offset (LBO), stride byte offset (SBO), and swizzle mode. `wgmma.mma_async` reads the descriptor from registers but fetches the actual matrix data directly from SMEM via the L2 cache, bypassing registers entirely. Only the current descriptor (not all of them) is held in registers at any given time — hence "Iterator".

In contrast, `tCrC` (the accumulator) is a register-backed tensor — the 32 output values per thread for a 64x64 output atom are stored in registers.

#### 4.2.4. SMEM Layout Constraints

The SMEM layouts for `sA` and `sB` must satisfy two requirements for WGMMA:

1. **Tile divisibility**: `BM` must be a multiple of 64, `BN` a multiple of 64 (or the chosen N), `BK` a multiple of 16 (for fp16).
2. **Swizzle compatibility**: the SMEM layout must use one of the canonical GMMA swizzle atoms (`SW128`, `SW64`, `SW32`, or no-swizzle) tiled to shape `(BM, BK)` or `(BN, BK)`.

In `c1` these are built automatically:

```python
self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
    a_layout=self.a_layout,         # determines MN-major vs K-major atom
    mma_tiler_mnk=self.tile_shape_mnk,
    a_dtype=self.a_dtype,
    num_stages=self.num_stages
)
# Prints: S<3,4,3> o 0 o ((8,16),(64,1),(1,1)):((64,512),(1,0),(0,0))
# i.e., Sw<3,4,3> (128-byte swizzle) composed over the base layout
```

Swizzled layouts eliminate SMEM bank conflicts and also ensure the matrix descriptor's LBO/SBO fields are set correctly by `make_gmma_desc` inside CUTLASS.

#### 4.2.5. Synchronization Protocol

WGMMA executes in the async proxy (same as TMA). The correct sequence (wrapped by CuTeDSL) maps to PTX as follows:

```python
# 1. Before the first GEMM — fence RMEM/SMEM accesses across warpgroup
cute.nvgpu.warpgroup.fence()          # PTX: wgmma.fence.sync.aligned

# 2. Issue WGMMA (returns immediately; runs async in background)
#    Inner loop over MMA_K automatically issues MMA_K/k_atom many wgmma instructions
cute.gemm(tiled_mma, accumulators, tCrA_k, tCrB_k, accumulators)

# 3. Seal all outstanding wgmma instructions into one trackable group
cute.nvgpu.warpgroup.commit_group()   # PTX: wgmma.commit_group.sync.aligned

# 4. Block until at most N groups are still in-flight
cute.nvgpu.warpgroup.wait_group(0)    # PTX: wgmma.wait_group.sync.aligned 0
```
*[`c1_wgmma_tma_load_store.py` L305–L326](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L305).*

| Primitive | PTX instruction | Purpose |
|---|---|---|
| `warpgroup.fence()` | `wgmma.fence.sync.aligned` | Orders prior RMEM/SMEM writes before WGMMA reads them. Required at the start of each GEMM call when the accumulator is being reused |
| `cute.gemm(...)` | `wgmma.mma_async.sync.aligned.*` | Issues the async MMA; each inner K-block loop issues one instruction |
| `warpgroup.commit_group()` | `wgmma.commit_group.sync.aligned` | Groups all prior uncommitted `wgmma.mma_async` instructions into one wgmma-group for tracking |
| `warpgroup.wait_group(N)` | `wgmma.wait_group.sync.aligned N` | Blocks until ≤ N wgmma-groups remain in-flight |

Why `wait_group(0)` here? In `c1` there is only one pipeline stage, so the consumer must wait for WGMMA to finish reading `sA`/`sB` before the next K-iteration overwrites them. A multi-stage pipeline (`c2`, `c3`) can set `wait_group(1)` — wait for the previous group while the current group is still running — enabling overlap of SMEM loads and compute.

Note: `fence.proxy.async` (used for TMA store) is not needed before WGMMA when SMEM is populated by TMA. TMA itself is in the async proxy, and the `mbarrier_wait` that gates TMA load completion already ensures visibility to WGMMA. `fence.proxy.async` would only be needed if SMEM were written by ordinary thread stores.

#### 4.2.6. The ACCUMULATE Flag

```python
# Before the very first GEMM tile — zero-initialize the accumulators
tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)  # C = A*B (not A*B + C)

# After the first tile — accumulate into existing registers
tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)   # C = A*B + C
```
*[`c1_wgmma_tma_load_store.py` L284 and L323](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L284).*

This maps to the `scale_D` parameter in the PTX instruction: `0` zero-initializes the accumulator, `1` adds to it. Setting `False` before the first tile avoids the overhead of a separate `fill(0)` on the accumulator register fragment.

</details>

<details>
<summary><strong>5. Asynchronous Pipeline: PipelineAsync vs PipelineTmaAsync</strong></summary>

Hopper's new barrier primitives allow us to overlap memory transactions and computation. CuTeDSL exposes this via `PipelineAsync` and `PipelineTmaAsync`.

### 5.1. Warp Specialization

We split work by **role** rather than by data:

- **Producer warps** — handle memory (GMEM → SMEM)
- **Consumer warps** — handle computation (MMA on registers or WGMMA from SMEM)

Different warps can take entirely different code paths with zero divergence penalty because they are independently scheduled by the warp scheduler. By doing this, we can overlap the computation done by the consumer with the SMEM filling of the producer. This technique is called **warp specialization**.

### 5.2. Pipeline Communication via Barriers in Shared Memory

Producer and consumer warps communicate via **`mbarrier`** objects stored in shared memory (visible to all threads in a block), which acts as a communication channel for different warps so they can signal each other when data is ready to be consumed or when the data on that smem parition is consumed and ready to be filled with new data. Each pipeline stage gets its own barrier, organized as a **circular buffer** — after the last stage we wrap back to stage 0.

Each barrier tracks a **phase** that alternates between even and odd. The two race conditions that barriers prevent:

- **Producer overwrite** — producer `acquire` blocks until the consumer has released that stage (data fully consumed).
- **Consumer underread** — consumer `wait` blocks until the producer has committed to that stage (data fully written).

With `S` stages the producer can run up to `S` iterations ahead of the consumer, hiding memory latency behind computation.

### 5.3. PipelineAsync (Synchronous Writes + Synchronous MMA)

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

### 5.4. PipelineTmaAsync (Async TMA Loads + Async WGMMA)

Used on Hopper when both the producer (TMA) and consumer (WGMMA) are asynchronous. Two complementary mechanisms track completion:

- **TMA completion**: tracked via **transaction byte counting** on barriers. The `tx_count` parameter tells the pipeline how many bytes to expect per stage. TMA hardware automatically decrements the barrier's counter as bytes land in SMEM — `producer_commit()` is effectively a NOP.
- **WGMMA completion**: tracked via `commit_group()` / `wait_group()` as described in Section 4.

Because the consumer must wait for WGMMA to finish reading SMEM before it is safe to release a stage for the producer to overwrite, it needs **two separate state trackers**: `consumer_read_state` (advances when data is ready to consume) and `consumer_release_state` (advances after WGMMA finishes reading).

The producer uses a **prefetch phase** to fill all `S` pipeline stages before the steady-state loop, maximizing overlap:

```python
# Setup
tma_transaction_bytes = cute.size_in_bytes(a_dtype, a_smem_layout) \
                      + cute.size_in_bytes(b_dtype, b_smem_layout)

pipeline = PipelineTmaAsync.create(
    num_stages=S,
    producer_group=CooperativeGroup(Agent.Thread, num_tma_warps),
    consumer_group=CooperativeGroup(Agent.Thread, num_mma_warps),
    barrier_storage=mbar_ptr,
    tx_count=tma_transaction_bytes,   # bytes TMA hw will write per stage
    cta_layout_vmnk=cute.make_layout((1, *cluster_shape)),
)

producer_state       = make_pipeline_state(PipelineUserType.Producer, S)
consumer_read_state  = make_pipeline_state(PipelineUserType.Consumer, S)
consumer_release_state = make_pipeline_state(PipelineUserType.Consumer, S)

# ── TMA warp (producer) ────────────────────────────────────────────────────
if is_tma_warp:
    # Prefetch: eagerly fill all S stages before consumer starts
    for kidx in range(S):
        pipeline.producer_acquire(producer_state)            # wait until stage is free

        bar = pipeline.producer_get_barrier(producer_state)
        cute.copy(tma_atom_a, tAgA[None, producer_state.count],
                  tAsA[None, producer_state.index], tma_bar_ptr=bar)
        # Same for sB

        pipeline.producer_commit(producer_state)             # NOP — TMA hw signals via tx_count
        producer_state.advance()

    # Steady-state: produce remaining K tiles
    for kidx in range(S, K // BK):
        pipeline.producer_acquire(producer_state)

        bar = pipeline.producer_get_barrier(producer_state)
        cute.copy(tma_atom_a, tAgA[None, producer_state.count],
                  tAsA[None, producer_state.index], tma_bar_ptr=bar)
        # Same for sB

        pipeline.producer_commit(producer_state)
        producer_state.advance()

# ── MMA warps (consumer) ───────────────────────────────────────────────────
if is_mma_warp:
    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False) # Set to False before first MMA

    for kidx in range(K // BK):
        pipeline.consumer_wait(consumer_read_state)          # wait for tx_count to hit 0

        cute.nvgpu.warpgroup.fence()

        for k_block_idx in range(num_k_blocks):              # loop over K sub-tiles
            cute.gemm(
                tiled_mma,
                accumulators,
                tCrA[None, None, k_block_idx, consumer_read_state.index],
                tCrB[None, None, k_block_idx, consumer_read_state.index],
                accumulators,
            )   
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        cute.nvgpu.warpgroup.commit_group()                  # seal all WGMMA into one group
        cute.nvgpu.warpgroup.wait_group(0)                   # wait until group finishes

        pipeline.consumer_release(consumer_release_state)    # signal stage is free
        consumer_read_state.advance()
        consumer_release_state.advance()
```

```
Stage lifecycle
───────────────
TMA warp (producer)                   MMA warps (consumer)
      │                                      │
      ├─ producer_acquire()                  │
      ├─ TMA copy A/B → sA/sB[stage]         │
      │    hw writes bytes, decrements       │
      │    tx_count on barrier               │
      ├─ producer_commit()  [NOP]            │
      │    tx_count hits 0 ─────────────►    ├─ consumer_wait() unblocks
      │                                      ├─ warpgroup.fence()
      │                                      ├─ WGMMA reads sA/sB (async)
      │                                      ├─ commit_group() → seals group
      │                                      ├─ wait_group(0)  → MMA done
      │  ◄───────────────────────────────────├─ consumer_release()
```

### 5.5. What About SM120 (Blackwell RTX)?

SM120 has TMA but **no WGMMA**. It uses register-based MMA (`ldmatrix` + `mma`). Key differences from SM90:

- `ldmatrix` and `mma` are **synchronous** — no `fence`/`commit_group`/`wait_group` needed.
- Only **one consumer state tracker** needed — the stage can be released as soon as `ldmatrix` finishes (MMA never reads SMEM).
- `consumer_release` happens **inside** the k_block loop at the boundary after the last `ldmatrix` of each stage.
- **Double-buffering** the `ldmatrix` for the next k_block with MMA of the current k_block hides `ldmatrix` latency — classic software pipelining.

See [Junkai Wu's dense_gemm SM120 example](https://github.com/NVIDIA/cutlass/blob/main/python/examples/blackwell_rtx/dense_gemm.py) for the full implementation.

</details>

<details>
<summary><strong>6. Profiling Async Pipelines with Inline PTX Probing</strong></summary>

To verify that our async pipeline is actually overlapping memory and compute, we need timing measurements inside the kernel. Taking inspiration from [gau-nernst's blog post](https://gau-nernst.github.io/tcgen05/), the probing technique uses **inline PTX** to call the `globaltimer` instruction, which returns the current GPU clock in nanoseconds.

CuTeDSL exposes this via LLVM IR, which lets you embed raw PTX instructions directly in Python kernel code:

```python
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(
        T.i64(), [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int64(t)
```

Each warp records timestamps at the start and end of its producer (TMA load) or consumer (WGMMA / MMA) work. The timestamps are written to a small global memory buffer, then post-processed on the CPU and visualized on [Perfetto](https://ui.perfetto.dev).

**Important caveats about the profiling kernel (`a2_smem_pipeline_profile.py`, `c3_wgmma_tma_specialized_pipeline_profile.py`):**

- Only **2 warps per block** are launched — very low occupancy by design so the timeline is readable.
- The SMEM loading is the main bottleneck; it can be accelerated with vectorized loads or TMA.
- The MMA completes very quickly (low Arithmetic Intensity = FLOPS / bytes transferred), so the pipeline profile shows a memory-bound workload.

**Profile of PipelineAsync (a2):**

The figure below shows producer (SMEM load) and consumer (MMA) timelines per warp. With only 2 warps there is limited overlap, but the barrier handoff between producer and consumer is clearly visible.

![PipelineAsync profile](./assets/a2_pipeline_profile.png)

**Profile of PipelineTmaAsync (c3, 64x128x64 tile):**

Running `c3_wgmma_tma_specialized_pipeline_profile.py` and visualizing on Perfetto shows the first 3 prefetched TMA loads (filling the pipeline), followed by well-overlapped WGMMA compute and TMA loads in the steady state. Launch overhead is still visible at the start.

![PipelineTmaAsync profile](./assets/c3_pipeline_profile_64x128x64.png)

</details>

<details>
<summary><strong>7. Blackwell tcgen05 Matrix Multiplication</strong></summary>

Blackwell (SM100) introduces **tcgen05**, a new generation of tensor core instruction designed for the new architecture. Like WGMMA on Hopper, it requires multiple warps working together and is issued asynchronously. CuTeDSL exposes it through the `tcgen05` MMA atom.

Script `d1_tcgen05_tma.py` provides an example of tcgen05 combined with TMA loads.

Q: Why make_fragment_A(sA) instead of make_fragment_A(tCsA)?

For WGMMA (SM90), thr_mma = tiled_mma.get_slice(tidx) maps a thread to its subset of A. So partition_A(sA) gives the per-thread view, and make_fragment_A on that makes sense.

For UMMA (SM100), tiled_mma.get_slice(0) is a CTA-level slice — there's no thread partitioning of A/B operands because UMMA reads SMEM via hardware descriptors, not per-thread register loads. The fragment it creates (tCrA) is a tensor of SMEM matrix descriptors, not actual register data. Those descriptors point into the full sA, so you pass sA directly. partition_A(gA) is still needed but only to give the right GMEM-side shape for TMA partitioning — not for making the MMA fragment.



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
Recommended approach: they gives 30$ credit upon account creation, each run costs 10 cents, and I love using Modal (this repo is not sponsored by Modal :))
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
5. https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
6. https://gau-nernst.github.io/tcgen05/
7. https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
8. https://github.com/LeiWang1999/CPPTorchExecutable
9. https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html
10. https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
11. https://veitner.bearblog.dev/persistent-gemm-in-cutedsl-on-hopper/