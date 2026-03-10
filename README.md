# Learn CuTeDSL

 CuTeDSL provides quality of life APIs while making sure you have access to the low level hardware to write performant kernels. Here, you can find kernels for Ampere SM80, Hopper SM90, Blackwell SM100 (B200) and Blackwell SM120 (RTX Pro 6000 Blackwell, RTX 5090, 50s series). Beyond the kernels, the repo also serves as a reference for commonly used CuTeDSL APIs. Each concept is isolated and explained with minimal surrounding noise, making it easy to lift a pattern into your own code or feed it as context to an LLM. 

## Frequently used APIs explanation 
*Click ▶ to expand the section*

<details>
<summary>1. Overview</summary>

CuTeDSL and CUDA in general have a very steep but rewarding learning curve, so don't get frustrated the first time you try it. The best approach is to look at examples, write kernels yourself, and observe the performance speedup — and understand why it speeds up. Once you can wrap your head around the concept of massively parallel programming with CUDA, subsequent kernels become much easier to digest. 

Each generation of GPU shifts the programmer's burden. On Ampere, you hand-tune everything: coalesced memory access patterns, shared memory padding or XOR swizzling to avoid bank conflicts, and the exact thread-to-register mapping that ldmatrix demands to feed tensor core fragments. Hopper's TMA eliminates the memory side of that equation — address arithmetic, boundary clamping, and swizzle remapping all move into hardware — but replaces it with a new obligation: asynchronous barrier protocols (mbarrier_expect_tx, phase bits, transaction byte counts) that are harder to debug than the pointer math they replace. Blackwell pushes further: TMEM absorbs the accumulator out of the register file entirely, freeing hundreds of registers per thread, but now you manage a six-step TMEM lifecycle, cross-CTA cooperative instructions, and a tcgen05_fence ordering constraint that has no Hopper equivalent. The net trajectory is clear — each generation automates the tedious data-movement work but demands deeper understanding of the PTX instruction semantics underneath. And with CuTeDSL providing apis that are wrappers to these hardwares' PTX instructions, our life has become easier than ever.

The suggested progression through this repository starts with vector addition ([`a0`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a0_vector_addition.py)), the classic gateway CUDA example. Plenty of online explanations exist for this pattern. CUDA has a broader ecosystem of YouTube videos and blog posts than CuTeDSL, but CuTeDSL is essentially a Python wrapper over CUDA/PTX so the concepts transfer directly. From there, naïve GEMM ([`a1`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cute.py)) introduces how to perform General Matrix Multiplication in a parallel fashion using one thread per output element. The shared memory GEMM ([`a2`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_cuda_like.py)) then loads tiles of A and B into fast on-chip shared memory to exploit data reuse and reduce global memory traffic, producing the first dramatic FLOPS jump. You can check the folder `cuda/kernels/` to explore CUDA C++ kernels for naive, smem and WMMA, with the files naming convention that map to equivalent ones in `cutedsl/` folder, these CUDA kernels are compiled JIT with ninja, you can run them directly by launching `cuda/gemm_cuda.py` or through modal submission, check function `run_kernel_sm80` in `submit_modal.py` and section `Job_submission` in this repo.

The b-series introduces WMMA (Warp Matrix Multiply-Accumulate) tensor core instructions available from Ampere SM80 onwards. Lei Mao's blog provides excellent tutorial for WMMA instruction: https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/. In CuTeDSL and CUTLASS generally, the convention is to map PTX instructions directly rather than going through the CUDA C++ `wmma` API. The c-series moves to Hopper SM90 with TMA (Tensor Memory Accelerator) and WGMMA (Warp Group MMA), combined with new async barrier primitives for true pipelined overlap of memory and compute. Finally, the d-series covers Blackwell's tcgen05 instruction for SM100 and SM120.

</details>

<details>
<summary>2. CuTeDSL Fundamentals</summary>

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

Threads within the same block can communicate via shared memory and synchronize with `sync_threads()`. Threads in different blocks cannot directly synchronize — they run independently. Understanding why the work is partitioned this way is the first mental hurdle in CUDA; if it's new to you, working through [`a0_vector_addition.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a0_vector_addition.py) and [`a1_naive_cuda_like.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cuda_like.py) yourself is the most effective way to build that intuition.

### 2.4. Layout and Linear indexing

When performing matrix multiplication we write `A[i, j] * B[j, k] = C[i, k]`, which uses 2-D indexing. Physical GPU memory is flat (1-D), so every element access is ultimately just a pointer plus an integer offset: `*(base_ptr + offset)`. A multi-dimensional index `(row, col)` must be collapsed to that single offset before the hardware can fetch the value.

Two conventions dominate GPU matrix storage. Row-major (C-order) stores elements along the last dimension contiguously, so for a matrix of shape `(M, K)` the stride is `(K, 1)` and offset `= row * K + col`. Column-major (Fortran-order) stores elements along the first dimension contiguously, giving stride `(1, M)` and offset `= row + col * M`. GEMM kernels CUBLAS typically expect B to be in column-major layout (equivalently, stored row-major but transposed), which is why you see `B.T` or the `NN`/`NT`/`TN`/`TT` GEMM variants in the literature. CuTe makes this explicit through strides: the matrix itself is never copied or rearranged in memory; only the stride arithmetic changes.

In traditional CUDA C++ you compute these offsets by hand and pass raw pointers. CuTe's Layout abstraction — a pairing of shape and stride — encodes the formula once and applies it automatically every time you index into a tensor, making the arithmetic explicit, verifiable, and composable across tiling levels.

Arguably the most important concept in CUTLASS CuTe / CuTeDSL. A `Layout` pairs a **shape** (extents in each dimension) with a **stride** (step size in linear memory per dimension). The key formula is:

$$\text{offset} = \sum_i \text{coord}_i \times \text{stride}_i$$

For example, a row-major matrix `A` of shape `(M, K)` has `stride=(K, 1)`. Accessing element at row `r`, column `c` gives offset `r * K + c`. In memory this just means: "skip `r` full rows of `K` elements, then `c` more."

You can implement Layout in CUDA C++ yourself [cuda/cuda_common.cuh #L33](https://github.com/luongthecong123/learn-cutedsl/blob/main/cuda/cuda_common.cuh#L33)

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
coord = (bidx, bidy, None)   #  We launched 2D grid, so the third dim is None

gA_tile = cute.local_tile(gA, tiler=tiler, coord=coord, proj=(1, None, 1))
gB_tile = cute.local_tile(gB, tiler=tiler, coord=coord, proj=(None, 1, 1))
gC_tile = cute.local_tile(gC, tiler=tiler, coord=coord, proj=(1, 1, None))
```

`proj` masks out which dimensions are "owned" by the CTA. For `gA_tile`, `proj=(1, None, 1)` means use BM and BK of `tiler` respectively, and leave the inner K-iteration axis free so the kernel can loop over it. The pointer arithmetic for the tile origin is:

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
*Used in [`a2_smem_cuda_like.py` L58](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_cuda_like.py#L58). For async pipelines (TMA), `mbarrier`-based synchronization replaces this — see Section 5.*


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
*Used in [`b2_wmma_smem.py` L74–L78](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L74) for GMEM → SMEM copies with vectorization.*

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
| `LdMatrix8x8x16bOp` | SMEM → register | `ldmatrix` instruction; loads data in the exact layout tensor cores expect. Used in [`b2_wmma_smem.py` L169–L186](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L169) |
| `CopyBulkTensorTileG2SOp` | GMEM → SMEM | TMA async load, Hopper+. Used in [`c1_wgmma_tma_load_store.py` L435](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L435) |
| `CopyBulkTensorTileS2GOp` | SMEM → GMEM | TMA async store. Used in [`c1_wgmma_tma_load_store.py` L105](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L105) |


### 2.7. MMA Atom

An **MMA Atom** wraps a single hardware matrix-multiply-accumulate instruction. A **Tiled MMA** tiles this atom across threads and repeats it to cover a larger output tile.

```python
# Define the hardware instruction
mma_op = cute.nvgpu.warp.MmaF16BF16Op(
    ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))

# Tile it: atom_layout_mnk=(2,2,1) means 2x2 warp-level tiles → 128 threads
tiled_mma = cute.make_tiled_mma(
    op_or_atom=mma_op,
    atom_layout_mnk=(2, 2, 1),
    permutation_mnk=(32, 32, 16))
```
*Used in [`b2_wmma_smem.py` L48–L63](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L48).*

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
| `MmaF16BF16Op` (warp) | Ampere+ | `wmma`-style warp-level MMA — used in [`b2_wmma_smem.py` L48](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L48) |
| WGMMA atom | Hopper SM90 | Warp-group async MMA reading operands from SMEM — used in [`c1_wgmma_tma_load_store.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py) |
| tcgen05 atom | Blackwell SM100 | Next-gen tensor core — used in [`d1_tcgen05_tma_umma.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d1_tcgen05_tma_umma.py) |

</details>

<details>
<summary>3. Tensor Memory Accelerator (TMA)</summary>

Hopper (SM90) introduced TMA, a dedicated hardware unit that offloads bulk data movement between global and shared memory. TMA handles multi-dimensional address calculations, stride arithmetic, boundary clamping, and swizzle remapping entirely in hardware, freeing threads and registers for computation. Understanding TMA is essential before moving on to the tensor core instructions (Section 4) and the asynchronous pipeline (Section 5) that depend on it.

### 3.1. How TMA Works

Without TMA the data flow for loading a tile is:

```
GMEM → registers → SMEM
```

This consumes register file capacity. TMA short-circuits this:

```
GMEM → SMEM   (direct, asynchronous, no register allocation)
```

The issuing thread returns immediately after launching the TMA copy; the hardware writes data into shared memory in the background and signals completion via an **mbarrier** (see below). This leaves more registers free for computation. TMA also handles multi-dimensional address calculations, stride, and boundary clamping in hardware, removing that logic from the kernel.

Example `b5` shows TMA as a drop-in replacement for manual SMEM loading, yielding a nice speedup while remaining portable to SM90, SM100, and SM120.

#### 3.1.1. Two-step Process

TMA is always set up in two distinct steps — unlike a regular `TiledCopy` where everything lives in kernel code:

1. **Host side**: Build a `cuTensorMap` descriptor encoding the GMEM tensor's base pointer, shape, strides, and swizzle mode. This is done once, at Python (host) level, and passed to the kernel as `__grid_constant__ const`. The descriptor cannot be modified by the device.
2. **Kernel side**: The underlying PTX instruction (`cp.async.bulk.tensor`) is issued only by a single thread — only one thread supplies the coordinates and barrier pointer. However, in CuTeDSL, `cute.copy` for TMA must be called from a full warp (`if warp_idx == 0:`). The reason is that CuTeDSL's code generation internally selects which thread within the calling warp actually issues the PTX instruction. If you gate the call to `tidx == 0` alone, which will cause a deadlock.

### 3.2. TMA Load: GMEM → SMEM

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
*Used in [`c1_wgmma_tma_load_store.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L296).*

The GMEM-side of `tma_partition` returns **ArithTuples** instead of raw pointers. These are coordinate objects — the TMA descriptor resolves them to actual addresses. This is how TMA handles OOB predication: it never performs pointer arithmetic that could go out-of-bounds.

mbarrier synchronization protocol (only for TMA load — 4 steps):

| Step | CuTeDSL call | Underlying PTX | Who |
|---|---|---|---|
| Init | `mbarrier_init(mbar_ptr, cnt=1)` | `mbarrier.init.shared.b64` | thread 0 (once, before the K-loop) |
| Issue | `cute.copy(..., tma_bar_ptr=mbar_ptr)` | `cp.async.bulk.tensor.Nd ...mbarrier::complete_tx::bytes` | warp 0 (CuTeDSL) — decrements counter as bytes land |
| Arrive + Expect | `mbarrier_arrive_and_expect_tx(mbar_ptr, tx_bytes)` | `mbarrier.arrive.expect_tx.shared::cta.b64` | thread 0 — sets expected byte count and arrives |
| Wait | `mbarrier_wait(mbar_ptr, phase)` | `mbarrier.try_wait.parity.shared::cta.b64` | **all** threads — sleep until barrier flips phase |

The **phase bit** alternates (0 → 1 → 0 → …) each time the barrier is reused. In `c1` the barrier is initialized once before the K-loop, and phases cycle with `phase ^= 1` after each wait. In the multi-stage pipeline (`c2`) the pipeline API manages phase cycling automatically across all `num_stages` barriers.

```python
# Descriptor prefetch — at kernel start, warp 0 warms the constant cache holding the descriptor before the K-loop:

if warp_idx == 0:
    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

# c1_wgmma_tma_load_store.py — mbarrier init (before K-loop, lines 282-287)
if warp_idx == 0 and tidx == 0:
    cute.arch.mbarrier_init(mbar_ptr, cnt=1)
    cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()   # all threads must see the mbarrier init in SMEM

phase = 0

# K-tile loop (lines 293-320)
for kidx in range(num_k_tiles):
    # TMA issue — must be gated to a full warp, cutedsl internal issue TMA using a thread
    if warp_idx == 0:
        cute.copy(tma_atom_a, tAgA[None, kidx], tAsA[None, 0], tma_bar_ptr=mbar_ptr)
        cute.copy(tma_atom_b, tBgB[None, kidx], tBsB[None, 0], tma_bar_ptr=mbar_ptr)

        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_transaction_bytes)

    cute.arch.mbarrier_wait(mbar_ptr, phase)   # all threads block here
    phase ^= 1                                 # flip for next iteration
```
*Used in [`c1_wgmma_tma_load_store.py` lines 282–324](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L282).*

Notice the overall pattern: the `cute.copy` calls are inside `if warp_idx == 0:` (all 32 threads of warp 0 enter the call), while the `mbarrier_arrive_and_expect_tx` is inside the additional `if tidx == 0:` (only one thread updates the barrier). This distinction is critical — reversing it (gating `cute.copy` to `tidx == 0`) will cause deadlock.

When a cluster of CTAs is launched, TMA can broadcast the same tile from GMEM to multiple CTAs' SMEM in one operation — called TMA multicast. This is used in [`d3_tcgen05_tma_umma_2cta_specialized_pipeline.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d3_tcgen05_tma_umma_2cta_specialized_pipeline.py), where a 2x1 cluster of CTAs cooperates on a single Blackwell 2-CTA UMMA instruction. Both CTAs compute with the same B tile (they differ only in their M-row assignment), so using `CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)` broadcasts the B tile to both CTAs' SMEM from a single TMA issue, saving half the B-tile GMEM bandwidth per CTA. The `ctarank` of each CTA within the cluster is used to slice the tiled copy, so each CTA receives the correct half. In the warp-specialized pipeline (`c2`, `d2`, `d3`), TMA copies run on a dedicated TMA warp — the entire warp enters the `cute.copy` call, and the pipeline API (`PipelineTmaAsync` or `PipelineTmaUmma`) handles the mbarrier bookkeeping internally via transaction byte counting (`tx_count`), so there is no manual `mbarrier_arrive_and_expect_tx` call.

### 3.3. TMA Store: SMEM → GMEM

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
*[`c1_wgmma_tma_load_store.py` lines 383–395](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L383).*

Why a proxy fence and not an mbarrier?
TMA store operates in the async proxy (with respect to the CTA). Without `fence.proxy.async.shared::cta`, the TMA hardware could read stale values from SMEM.

The fence goes before the store (opposite of mbarrier for loads, which waits after).

If the SMEM buffer for `sC` needs to be reused (e.g., multi-wave pipeline), add arrive/wait around the store:

```python
# cute.arch.tma_store_arrive()   # commit the store into a bulk group
# cute.arch.tma_store_wait(0)    # wait until 0 groups remain in-flight
```

In [`c1_wgmma_tma_load_store.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py), the epilogue uses exactly this path. After WGMMA finishes accumulating results in registers, the kernel uses the TV layout to scatter each thread's accumulator values into the appropriate `sC` location in shared memory, then issues a `sync_threads()` to ensure all threads have written before the proxy fence and TMA store. In benchmarks, this RMEM → SMEM staging + TMA store path proved faster than direct per-thread RMEM → GMEM stores, because TMA issues a single bulk hardware transfer aligned on 128-byte boundaries, whereas scattered per-thread stores from 256 warps interact with the L2 cache at fine granularity with less coalescing.

To carry TMA store into the warp-specialized pipeline in [`c2_wgmma_tma_specialized_pipeline.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c2_wgmma_tma_specialized_pipeline.py). We need to use barrier instruction `cute.arch.barrier(barrier_id, number_of_threads)` where `number_of_threads` is equal to number of consumer threads, using `sync_threads()` here would cause a deadlock as the producer thread will never be able reach this sync, as they operate under the producer `if` branch. Another thing is we have to redefine a new `thr__mma` that pass in the threads under the consumer branch.

### 3.4. SMEM Layout Requirements and Stride Constraints

The SMEM layouts for A and B use swizzled layouts to avoid bank conflicts and satisfy the memory format that tensor core instructions expect. These are built automatically by `sm90_utils.make_smem_layout_a/b` for Hopper and `sm100_utils.make_smem_layout_a/b` for Blackwell.

For the GMEM tensor, TMA has one hard constraint: one dimension must have stride 1 (contiguous), and all other strides must be multiples of 16 bytes. For float16, this means the leading dimension must be divisible by 8. For a row-major A of shape `(M, K)`, this translates to `K % 8 == 0`. This is why inputs are created with `assumed_align=16`:

```python
A_ = from_dlpack(A, assumed_align=16)
```

</details>

<details>
<summary>4. Tensor Cores: WMMA (Ampere), WGMMA (Hopper), and tcgen05 UMMA (Blackwell)</summary>

Why do tensor cores exist? Because moving data is far more expensive than computing on it. A single HFMA (half-precision fused multiply-add) instruction costs roughly 1.5 pJ, while a single register file access costs around 30 pJ — a 20x gap. When a datacenter GPU operates under a 700 W thermal envelope, the energy budget for feeding operands through the register file becomes the binding constraint long before the ALUs run out of throughput. Tensor cores solve this by amortizing the data-movement cost across many MACs in a single instruction: the hardware fetches a tile of operands once and performs hundreds of multiply-accumulates on those values behind a single instruction dispatch. Deep learning workloads are the ideal target because 80–95% of their compute is dense matrix multiplication — operations that map directly onto fixed-shape MMA tiles.

Each generation of NVIDIA datacenter GPU ships a new tensor core instruction with a larger MMA shape, a wider thread participation group, and a progressively more asynchronous execution model. The largest dense fp16/bf16 instruction shape tells the story: Ampere's WMMA atom is 16x8x16 — a single warp of 32 threads issues a synchronous multiply that completes before the next instruction begins, accumulating into per-thread registers. Hopper's WGMMA pushes this to 64x256x16 — a warpgroup of 128 threads issues an asynchronous instruction whose operands are read directly from shared memory via matrix descriptors rather than from registers, and whose completion is tracked through an explicit commit/wait protocol. Blackwell's tcgen05 MMA reaches 256x256x16 across a cooperative pair of CTAs — operands come from shared memory, results land in a dedicated Tensor Memory (TMEM) scratchpad per SM, and a single thread can initiate the entire operation. Smaller precisions scale the K dimension: fp8 doubles K to 32, and fp4 quadruples it to 64, keeping the operand bandwidth per instruction constant while multiplying arithmetic throughput. Each successive generation trades programming simplicity for throughput — the synchronization protocol grows more involved, but the reward is a dramatic reduction in data movement overhead and tensor core idle time. This section walks through all three instructions as implemented in CuTeDSL, covering the API, the fragment layout, and the synchronization barriers each one requires.

### 4.1. Warp Matrix Multiply-Accumulate (WMMA)

WMMA is the tensor core instruction available starting from Ampere (SM80). Unlike its successor WGMMA, it operates at warp granularity — all 32 threads in a warp collectively execute the instruction, and it completes synchronously before subsequent instructions begin. This makes WMMA straightforward to reason about: no commit-group, no wait-group, no async proxy fencing. A simple `sync_threads()` before reading SMEM is sufficient to ensure TMA or manual copy operations have landed, and the `cute.gemm` call itself returns only after the MMA is complete.

The PTX instruction underneath is `wmma.mma.sync.aligned.row.col.m16n8k16.f32.f16.f16.f32` for fp16 inputs with fp32 accumulation. A single warp-level atom covers a 16x8 output tile. In CuTeDSL, this is exposed as `cute.nvgpu.warp.MmaF16BF16Op`, and you tile it across multiple warps with `cute.make_tiled_mma` and the `atom_layout_mnk` parameter. In [`b2_wmma_smem.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py), a 2x2 warp arrangement with computed `permutation_mnk` covers a 32x32 output tile using 4 warps (128 threads):

```python
# b2_wmma_smem.py L48-L63
mma_op = cute.nvgpu.warp.MmaF16BF16Op(
    ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))
permutation_mnk = (
    atom_layout_mnk[0] * mma_inst_shape[0],      # 2*16 = 32
    atom_layout_mnk[1] * mma_inst_shape[1] * 2,  # 2*8*2 = 32
    atom_layout_mnk[2] * mma_inst_shape[2],       # 1*16 = 16
)
tiled_mma = cute.make_tiled_mma(
    op_or_atom=mma_op,
    atom_layout_mnk=(2, 2, 1),
    permutation_mnk=permutation_mnk)
```

The `atom_layout_mnk` of `(2, 2, 1)` arranges 4 warps in a 2x2 grid along M and N. Since the base atom is 16x8, the N factor gets a multiplier of 2 in `permutation_mnk` to account for how CuTeDSL lays out the atom — this is specific to the 16x**8**x16 atom shape. The result is `(32, 32, 16)`: each warpgroup invocation computes a 32x32 output tile in one K-step of 16.

On device, each thread gets its own slice:

```python
# b2_wmma_smem.py L157-L165
thr_mma = tiled_mma.get_slice(tid)
tCsA = thr_mma.partition_A(sA)
tCsB = thr_mma.partition_B(sB)
tCgC = thr_mma.partition_C(gC)
tCrA = tiled_mma.make_fragment_A(tCsA)
tCrB = tiled_mma.make_fragment_B(tCsB)
tCrC = tiled_mma.make_fragment_C(tCgC)
tCrC.fill(0.0)
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
```

#### 4.1.1. Register Fragments and the Tensor Core Data Layout

It is important to understand what `make_fragment_A`, `make_fragment_B`, and `make_fragment_C` are doing under the hood. When you write `a = 1.0` in a scalar register, every thread's register slot holds the literal value `1.0`. Tensor core fragments are fundamentally different: they are distributed data structures where the **same register index** across different threads holds **different values** dictated by the tensor core's fixed thread-to-data mapping. For the 16x8x16 WMMA atom, each of the 32 threads in a warp holds 4 accumulator registers that collectively tile a 16x8 output. Thread 0's register 0 might hold element `(0, 0)`, while thread 1's register 0 holds element `(1, 0)` — the mapping depends on the specific PTX instruction layout.

This is why you cannot simply load data into fragment registers using a flat `register[i] = value` pattern. The data must arrive in registers in the exact layout the tensor core expects. The `ldmatrix` PTX instruction exists precisely for this purpose.

#### 4.1.2. LdMatrix: Loading SMEM into Tensor Core Fragments

`LdMatrix8x8x16bOp` (PTX: `ldmatrix.sync.aligned.m8n8.x4`) is a cooperative, warp-synchronous instruction that loads data from shared memory directly into registers in the precise layout that `wmma.mma` expects. Each of the 32 threads in a warp provides an SMEM address, and the hardware shuffles the loaded values so that each thread's register file ends up holding the correct fragment elements — no explicit data rearrangement needed. The `x4` suffix means four 8x8 matrices (each 16-bit elements) are loaded in one instruction, covering 4 × 8 × 8 × 2 bytes = 256 bytes total.

In [b2_wmma_smem.py](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py#L169), the LdMatrix copy atom is constructed and tiled to match the MMA layout:

```python
# b2_wmma_smem.py L169-L186
atom_copy_s2r_A = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
    mA.element_type,
)
tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tid)
tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
cute.copy(atom=tiled_copy_s2r_A, src=tCsA_copy_view, dst=tCrA_copy_view)
```

`make_tiled_copy_A(atom, tiled_mma)` automatically derives the correct thread layout and value layout from the TiledMMA, ensuring the `ldmatrix` addresses align with the WMMA fragment layout. The `retile` call reshapes the register fragment `tCrA` to match the copy's destination layout without moving data.

#### 4.1.3. StMatrix: Storing Registers Back to SMEM

The reverse operation — writing from tensor core register layout back into shared memory — is `StMatrix8x8x16bOp` (PTX: `stmatrix.sync.aligned.m8n8.x4`). This is useful when you want to stage accumulator results through SMEM before a bulk TMA store to GMEM. In [c1_wgmma_tma_load_store.py](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L345), StMatrix is shown (commented out) as an alternative to the TV-layout-based register-to-SMEM scatter:

```python
# c1_wgmma_tma_load_store.py L345-L364 (commented out alternative)
copy_atom_C = cute.make_copy_atom(
    cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
    self.c_dtype,
)
tiled_copy_r2s_C = cute.make_tiled_copy_C(copy_atom_C, tiled_mma)
thr_copy_stmatrix_C = tiled_copy_r2s_C.get_slice(tidx)
tCrC_copy_view = thr_copy_stmatrix_C.retile(tCrC_out)
tCsC_copy_view = thr_copy_stmatrix_C.partition_D(sC)
cute.copy(atom=tiled_copy_r2s_C, src=tCrC_copy_view, dst=tCsC_copy_view)
```

For understanding how register indices map to logical `(m, n)` coordinates — essential for layer fusion patterns like performing element-wise ops directly on the accumulator without materializing to GMEM — refer to Section 6 (TV Layout).

### 4.2. Warp Group Matrix Multiplication (WGMMA)

WGMMA (`wgmma.mma_async`) is Hopper's tensor core instruction. It supersedes the warp-level WMMA used on Ampere and earlier. WGMMA requires all 128 threads in a warpgroup (4 contiguous warps whose first warp rank is a multiple of 4) to issue the instruction collectively. It is asynchronous: `cute.gemm(tiled_mma, ...)` returns immediately, the tensor cores continue computing in the background reading operands directly from SMEM via matrix descriptors rather than registers, and the accumulator is held in per-thread registers. The B operand always comes from SMEM; A can come from SMEM (SS mode) or registers (RS mode). In c1, both are sourced from SMEM.

#### 4.2.1. WGMMA Atom Shape Constraints

The underlying PTX instruction is `wgmma.mma_async.sync.aligned.M64xNxK`. M is always 64. K times the element size must equal 32 bytes, so for float16 K is 16 and for float8 K is 32. N is a multiple of 8 from 8 to 256.

#### 4.2.2. TiledMMA Construction

In c1, the TiledMMA is built on the host via a CuTeDSL helper:

```python
self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
    self.a_dtype,
    self.b_dtype,
    self.a_layout.sm90_mma_major_mode(),   # MN-major or K-major (matches memory layout)
    self.b_layout.sm90_mma_major_mode(),
    self.acc_dtype,                         # Float32 accumulator
    self.atom_layout_mnk,                   # e.g. (1,1,1) = one warpgroup
    tiler_mn=(64, self.tile_shape_mnk[1]), # the MxN output tile shape per warpgroup
)
```
*[`c1_wgmma_tma_load_store.py` L137–L147](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L137).*

`atom_layout_mnk` tiles the single 64xN atom across multiple warpgroups. For `BM=64`, it is `(1,1,1)` (one warpgroup). For `BM=128`, setting it to `(2,1,1)` assigns 2 warpgroups to split the M-dimension, each computing a 64xN subtile independently. The total thread count is `math.prod(self.atom_layout_mnk) * 128`. For float16, both MN-major and K-major are supported. For other dtypes, only K-major is allowed.

#### 4.2.3. Per-Thread Slicing and Fragments

On device, slice the TiledMMA per warpgroup rather than per individual thread:

```python
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

`tCrA` and `tCrB` are not register tensors, instead, they are `GMMA::DescriptorIterator` objects. Each is a 64-bit matrix descriptor held in registers that points into SMEM, encoding the base address, leading byte offset (LBO), stride byte offset (SBO), and swizzle mode. `wgmma.mma_async` reads that descriptor from registers but fetches matrix data directly from SMEM through the L2 cache, bypassing registers entirely. Only the current descriptor is held in registers at any given time — hence "Iterator". The accumulator `tCrC` is a genuine register-backed tensor: for instruction shape 64x256xK, each thread holds 128 32-bit registers, this high register usage will reduce our achieved occupancy, but should make the code faster due to better artithmetic intensity.

#### 4.2.4. SMEM Layout Constraints

The SMEM layouts for sA and sB must use one of the canonical GMMA swizzle atoms (SW128, SW64, SW32, or no-swizzle) tiled to shape (BM, BK) or (BN, BK). BM must be a multiple of 64, BN a multiple of 64, and BK a multiple of 16 for fp16. In c1 these are built automatically by `sm90_utils.make_smem_layout_a/b`. Swizzled layouts eliminate SMEM bank conflicts and ensure the matrix descriptor LBO/SBO fields are set correctly by `make_gmma_desc` inside CUTLASS.

#### 4.2.5. Synchronization Protocol

WGMMA executes in the async proxy. The correct sequence maps to PTX:

```python
# 1. Before the first GEMM — fence RMEM/SMEM accesses across warpgroup
cute.nvgpu.warpgroup.fence()          # PTX: wgmma.fence.sync.aligned

# 2. Issue WGMMA (returns immediately; runs async in background)
cute.gemm(tiled_mma, accumulators, tCrA_k, tCrB_k, accumulators)

# 3. Seal all outstanding wgmma instructions into one trackable group
cute.nvgpu.warpgroup.commit_group()   # PTX: wgmma.commit_group.sync.aligned

# 4. Block until at most N groups are still in-flight
cute.nvgpu.warpgroup.wait_group(0)    # PTX: wgmma.wait_group.sync.aligned 0
```
*[`c1_wgmma_tma_load_store.py` L324–L341](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py#L324).*

`warpgroup.fence()` orders prior RMEM/SMEM writes before WGMMA reads them — required at the start of each GEMM call when the accumulator is reused from a previous iteration. `commit_group()` batches all outstanding uncommitted `wgmma.mma_async` instructions into one trackable group. `wait_group(N)` blocks until at most N groups remain in-flight. In c1 with a single stage, `wait_group(0)` ensures WGMMA finishes before the next K-iteration can overwrite sA and sB. A multi-stage pipeline like c2 uses `wait_group(1)` to keep one group running while the next is loading, enabling true compute-memory overlap. `fence.proxy.async` is not needed before WGMMA when SMEM is populated by TMA, since the `mbarrier_wait` gating TMA completion already ensures SMEM visibility.

#### 4.2.6. The ACCUMULATE Flag

```python
# Before the very first GEMM tile — zero-initialize the accumulators
tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)  # C = A*B

# After the first tile — accumulate into existing registers
tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)   # C = A*B + C
```

This maps to the `scale_D` parameter in the PTX instruction. Setting False before the first tile avoids the overhead of a separate `fill(0)` on the accumulator fragment.

### 4.3. Blackwell tcgen05 UMMA

Blackwell's tcgen05 (the fifth-generation tensor core instruction, PTX mnemonic `tcgen05.mma.sync.aligned.*`) is the counterpart to Hopper's WGMMA on SM100 hardware. The most fundamental departure from WGMMA is where the accumulator lives.

On Hopper, WGMMA accumulates into the per-thread register file of the 128-thread warpgroup. On Blackwell, tcgen05 writes its accumulator into a dedicated on-chip memory called Tensor Memory (TMEM). TMEM is completely separate from shared memory and the register file — it is a 128-lane by 512-column scratchpad per SM that is specialized to hold MMA accumulators. After the MMA loop, an explicit `tcgen05.ld` instruction must move results from TMEM into per-thread registers before anything can be transformed or written to global memory. The upside is substantial: because the accumulator no longer occupies registers during the compute loop, tcgen05 tiles can be dramatically larger than equivalent WGMMA tiles without register pressure concerns.

#### 4.3.1 TMEM: Tensor Memory

TMEM is organized as a 2D array of 128 lanes (rows) and 512 columns, each cell a 32-bit value, for 128 × 512 × 4 bytes = 256 KB per SM. Allocation always consumes all 128 lanes — there is no way to use only a subset of rows. The unit of allocation is columns only, and must be a power of two between 32 and 512; the minimum allocation is 32 columns (16 KB). Even if your accumulator only requires 64×64 fp32 values, you still must allocate 64 columns across all 128 rows. Script [`z2_tmem_lower.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/z2_tmem_lower.py) demonstrates the TMEM lifecycle in the simplest possible form: load data from GMEM into registers, write to TMEM via `tcgen05.St32x32bOp`, read back via `tcgen05.Ld32x32bOp`, scale, and store to GMEM — with no GEMM at all.

The lifecycle in every Blackwell kernel follows a six-step protocol. First, a warp (in this example, warp 0) calls `cute.arch.alloc_tmem(num_cols, tmem_holding_buf)`, which writes the TMEM base address into a shared memory slot. Second, a `cute.arch.barrier()` ensures all threads see that write before continuing. Third, every thread calls `cute.arch.retrieve_tmem_ptr()` to read the base address and construct a TMEM pointer. Fourth, the MMA loop accumulates results into TMEM. Fifth, `cute.arch.relinquish_tmem_alloc_permit()` signals to hardware that this CTA will not request further TMEM allocations. Sixth, after the epilogue finishes reading TMEM, that same warp that allocated TMEM in the first place will call `cute.arch.dealloc_tmem()` to free the columns.

```python
# From d1_tcgen05_tma_umma.py
if warp_idx == 0:
    cute.arch.alloc_tmem(cutlass.Int32(num_cols), storage.tmem_holding_buf)
cute.arch.barrier(barrier_id=1, number_of_threads=128)   # all threads wait for alloc write

tmem_ptr = cute.arch.retrieve_tmem_ptr(
    cutlass.Float32, alignment=16,
    ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)
# ... MMA loop accumulates into TMEM ...
if warp_idx == 0:
    cute.arch.relinquish_tmem_alloc_permit()
cute.arch.barrier(barrier_id=1, number_of_threads=128)   # all epilogue reads done
if warp_idx == 0:
    cute.arch.dealloc_tmem(tmem_ptr, cutlass.Int32(num_cols))
```

The `utils.TmemAllocator` wrapper (used in d2, d3) encapsulates this protocol so you do not need to manage raw barriers manually.

#### 4.3.2 TMEM Load Instructions and Subtiling

Moving data out of TMEM into per-thread registers requires the `tcgen05.Ld` instruction. There are three memory movement instructions available: `tcgen05.ld` (TMEM → registers, used in every epilogue), `tcgen05.st` (registers → TMEM, for initializing or updating accumulators), and `tcgen05.cp` (SMEM → TMEM, for specialized cases like FlashAttention softmax corrections). Since `tcgen05.ld` is by far the most commonly used, understanding its shape encoding is essential.

The instruction shape is encoded as `Ld{DP}x{BITS}bOp`, where **DP** is the number of data-path lanes (how many TMEM lanes participate per warp) and **BITS** is how many bits are read from each lane per repetition. A **Repetition** count specifies how many times the instruction repeats across TMEM columns. The two most common shapes are:

**`Ld32x32bOp`** uses 32 data-path lanes and reads 32 bits (one fp32) per lane per repetition. Since there are 32 threads in a warp and 32 DP lanes, each thread maps 1:1 to a TMEM lane. With 4 warps covering 4 × 32 = 128 lanes, this shape requires MMA_M=128. Each repetition loads 1 fp32 column, so `Repetition(R)` means each thread holds R registers and R TMEM columns are consumed.

**`Ld16x256bOp`** uses 16 data-path lanes and reads 256 bits (eight fp32 values) per lane per repetition. The 16 DP lanes are distributed across 32 threads in a warp — two threads collaborate to carry the 256 bits from each lane. With 4 warps covering 4 × 16 = 64 lanes, this shape requires MMA_M=64. Each repetition loads 8 fp32 columns (256 bits / 32 bits), so `Repetition(R)` consumes R × 8 columns.

After the MMA loop, the full accumulator sits in TMEM. For a tile of (128, 256), that is 128 × 256 = 32,768 fp32 values. With 128 threads, loading the entire thing at once would require loading in 2 iterations called `subtiles`, as each thread can hold only a maximum of 128 registers for this instruction. The key relationship between repetition count and register pressure is: `subtile_n = tmem_ld_rep × fp32_cols_per_rep` columns per subtile, `subtile_cnt = N_acc / subtile_n` loop iterations, and `regs_per_thread = tmem_ld_rep` register pressure per iteration. Choosing the repetition count is a trade-off: larger repetitions reduce loop iterations (less instruction overhead) but increase register pressure. Benchmarks on B200 with a (128, 256) tile and `Ld32x32bOp` show diminishing returns past `Repetition(16)` — going from x1 (421 TFLOPS) to x8 (709 TFLOPS) to x16 (714 TFLOPS), after which the performance plateaus. In pipelined kernels with complex epilogues — like FlashAttention's softmax correction and rescaling — keeping register pressure low with smaller repetitions becomes critical to avoid register spilling. A practical choice is `Repetition(8)` to `Repetition(16)`. Or just let CuTeDSL handles this param automatically for us. Script `d1_tcgen05_tma_umma.py` let CuTeDSL chooses the `tcgen05.ld` instruction, while script `d1_tcgen05_tma_umma_ld.py` do this manually.

#### 4.3.3. API Overview

In [`d1_tcgen05_tma_umma.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d1_tcgen05_tma_umma.py), the tiled_mma for a single-CTA kernel is constructed on the host:

```python
# d1_tcgen05_tma_umma.py L78-L86
op = tcgen05.MmaF16BF16Op(
    self.a_dtype,
    self.acc_dtype,
    self.mma_inst_shape_mnk,           # e.g. (128, 256, 16)
    tcgen05.CtaGroup.ONE,
    tcgen05.OperandSource.SMEM,
    tcgen05.OperandMajorMode.K,
    tcgen05.OperandMajorMode.K,
)
self.tiled_mma = cute.make_tiled_mma(op)
```

Unlike WGMMA where `thr_mma.get_slice(tid)` gives a per-thread view, tcgen05 operates at CTA granularity: `tiled_mma.get_slice(thr_idx=0)` returns one slice for the entire CTA. The fragments `tCrA` and `tCrB` are SMEM matrix descriptors (not register tensors), exactly like WGMMA. The accumulator `tCtAcc` is backed by TMEM rather than registers, and must be populated via `cute.arch.retrieve_tmem_ptr()` before use.

#### 4.3.4. Synchronization: tcgen05_fence, commit, and mbarrier

tcgen05 uses a sequence that parallels WGMMA's `fence` / `commit_group` / `wait_group`, but adapted for the TMEM accumulator. After TMA finishes loading data into SMEM (gated by `mbarrier_wait`), the kernel must issue a `tcgen05.fence::after_thread_sync` before the first MMA instruction. This fence ensures that any prior thread-synchronous operations (in particular, the mbarrier wait that confirms SMEM data is ready) are ordered before the MMA reads SMEM. In [`d1_tcgen05_tma_umma.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d1_tcgen05_tma_umma.py#L34), this is implemented as a custom inline PTX wrapper because I couldn't find a equivalent api in CuTeDSL:

```python
# d1_tcgen05_tma_umma.py L34-L41
@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
```

The full synchronization sequence in the K-tile loop is: `mbarrier_wait` (SMEM data is ready) → `tcgen05_fence()` (order prior sync before MMA) → `cute.gemm` (issue UMMA, returns immediately) → `tcgen05.commit(mma_mbar)` (one thread signals when TMEM writes complete) → `mbarrier_wait(mma_mbar)` (epilogue threads wait for TMEM). This is visible in the [main loop at d1_tcgen05_tma_umma.py L274-L335](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d1_tcgen05_tma_umma.py#L274):

```python
# d1_tcgen05_tma_umma.py L274-L278 (before K-loop)
tiled_mma.set(tcgen05.Field.ACCUMULATE, False)   # first tile: C = A*B (zeros accumulator)
tma_phase = 0
mma_phase = 0

# d1_tcgen05_tma_umma.py L280-L335 (K-tile loop)
for kidx in range(num_k_tiles):
    # --- TMA load (same warp-level gating as Section 3.2) ---
    if warp_idx == 0:
        cute.copy(tma_atom_a, tAgA[None, kidx], tAsA[None, 0], tma_bar_ptr=tma_mbar)
        cute.copy(tma_atom_b, tBgB[None, kidx], tBsB[None, 0], tma_bar_ptr=tma_mbar)
        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_transaction_bytes)

    cute.arch.mbarrier_wait(tma_mbar, tma_phase)    # SMEM data landed
    tma_phase ^= 1

    tcgen05_fence()                                  # order barrier before MMA reads SMEM

    # --- UMMA issue (must be gated to a full warp, not a single thread) ---
    if warp_idx == 0:
        for k_block_idx in range(num_k_blocks):
            cute.gemm(tiled_mma, tCtAcc,
                      tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        if tidx == 0:
            tcgen05.commit(mma_mbar)                 # signal when TMEM writes finish

    cute.arch.mbarrier_wait(mma_mbar, mma_phase)    # epilogue: TMEM ready to read
    mma_phase ^= 1
```

Both `tma_phase` and `mma_phase` follow the same pattern: initialized to 0 before the loop, flipped with `^= 1` after each `mbarrier_wait`. The TMA mbarrier tracks when SMEM data has landed; the MMA mbarrier tracks when TMEM results are ready. Each alternating phase tells the hardware which completion event to wait for, avoiding confusion when the same barrier object is reused across iterations.

The `cute.gemm` and `tcgen05.commit` calls are gated to `if warp_idx == 0:` for the same reason as TMA loads (Section 3.1.1): CuTeDSL internally selects which thread within the calling warp issues the underlying PTX instruction. If you gate these calls to a single thread (`if tidx == 0:`), the other 31 threads never enter the function, and the warp deadlocks on its internal convergence check. The `tcgen05.commit` is a single-thread operation (only thread 0 needs to arrive at the mbarrier), which is why it sits inside the additional `if tidx == 0:` nested within the warp gate — exactly the same two-level gating pattern used for TMA's `mbarrier_arrive_and_expect_tx`.

The accumulate field works identically to WGMMA: set `ACCUMULATE=False` before the first K-tile (equivalent to zeroing the accumulator), then `ACCUMULATE=True` for all subsequent tiles so results are summed across K iterations.

#### 4.3.5. 2-CTA Cooperative MMA

Script [`d3_tcgen05_tma_umma_2cta_specialized_pipeline.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d3_tcgen05_tma_umma_2cta_specialized_pipeline.py) demonstrates the Blackwell 2-CTA cooperative instruction (`tcgen05.CtaGroup.TWO`). With a (256, 256, 16) MMA atom and `CtaGroup.TWO`, two CTAs in a 2x1 cluster cooperate: each CTA holds 128 rows of A in its own SMEM, the leader CTA (cluster ctaid=0) issues the MMA instruction, and the hardware writes the results into both CTAs' TMEM simultaneously.

The 2-CTA approach delivers two concrete benefits. First, it enables the largest MMA tile shape available on Blackwell — 256x256x16 — which doubles the M dimension from 128 to 256 compared to single-CTA operation. This maximizes arithmetic intensity per instruction dispatch. Second, the TMA side benefits from multicast: since both CTAs compute with the same B tile (they differ only in which M rows they handle), a single TMA issue with `CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)` broadcasts B from GMEM to both CTAs' SMEM in one DMA operation, saving half the B-tile bandwidth per CTA. The halved per-CTA B SMEM footprint (128 rows instead of 256) is what enables seven pipeline stages instead of four — the SMEM budget per stage drops from 48 KB (16 KB for A + 32 KB for B) to 32 KB (16 KB for A + 16 KB for B), fitting more stages within the SM's 228 KB SMEM capacity. Why deeper pipelines are faster than 2 or 3 stages will be explained in 5.3. PipelineTmaAsync.

TMEM allocation and deallocation in the 2-CTA case remain per-CTA. Each CTA allocates its own TMEM columns independently, and the hardware coordinates writing into both CTAs' TMEM internally when the leader CTA issues the cooperative MMA. Deallocation requires a cross-CTA barrier so both CTAs finish their epilogue reads before either frees its TMEM. The `TmemAllocator(is_two_cta=True)` constructor handles this automatically using an mbarrier in cluster-visible shared memory.

</details>

<details>
<summary>5. Asynchronous Pipeline: PipelineAsync, PipelineTmaAsync, and PipelineTmaUmma</summary>

The single biggest bottleneck in a non-pipelined GEMM is dead time: while TMA loads the next K-tile into SMEM, the tensor cores sit idle; while the tensor cores compute, the TMA unit has nothing to fetch. Pipelining eliminates this by running multiple K-tiles in flight simultaneously — the producer fills stage N+1 while the consumer computes on stage N. All three pipeline classes in CuTeDSL — PipelineAsync, PipelineTmaAsync, and PipelineTmaUmma — require SM90 or later because they depend on Hopper's mbarrier.`arrive.expect_tx` and `mbarrier.try_wait.parity` instructions for fine-grained, byte-counted synchronization between producers and consumers. Under the hood, these pipelines also dynamically rebalance the register file: `setmaxregister_increase` for the MMA warp (which needs hundreds of accumulator registers) and `setmaxregister_decrease` for the TMA warp (which does virtually no computation), so both can coexist on the same SM without the compute warp spilling to slower scratch memory.

---

### 5.1. Warp Specialization and Circular buffer

Producer and consumer warps run entirely different code paths inside the same CTA, which is possible in CUDA because the warp scheduler dispatches warps independently. Producer warps handle memory (GMEM to SMEM), consumer warps handle computation (MMA or WGMMA). This lets the hardware overlap the two, hiding memory latency behind compute throughput within a single CTA. The mbarrier objects in shared memory act as the communication channel between these two warp classes: the producer signals when data is ready, and the consumer signals when a stage is free to be refilled.

Each pipeline stage gets its own SMEM buffer and its own mbarrier, arranged as a **circular buffer** (ring buffer). With `num_stages = S`, the pipeline allocates S SMEM slots and S mbarriers, indexed 0 through S-1. The producer writes to stage `index`, then calls `.advance()` which increments the index modulo S — so after stage S-1 comes stage 0 again, wrapping around the ring. The consumer follows the same circular pattern, always trailing behind the producer. This means the kernel only ever needs S tiles worth of SMEM, no matter how many K-iterations the loop runs. When the producer wraps back to stage 0, it must first wait for the consumer to finish reading that stage — that is exactly what `acquire` (producer side) and `release` (consumer side) enforce via the per-stage mbarrier. The phase bit on each mbarrier flips every time the ring wraps past that slot, so the hardware can distinguish "stage 0, first lap" from "stage 0, second lap" without any reset.

```
Circular buffer with S = 3 stages:

        ┌─────────┐
   ┌───►│ Stage 0 │───┐
   │    │ mbar[0] │   │
   │    └─────────┘   │
   │                  ▼
┌─────────┐     ┌─────────┐
│ Stage 2 │◄────│ Stage 1 │
│ mbar[2] │     │ mbar[1] │
└─────────┘     └─────────┘

Producer index:  0 → 1 → 2 → 0 → 1 → 2 →   ...  (.advance() wraps mod S)
Consumer index:      0 → 1 → 2 → 0 → 1 →   ...  (trails behind producer)
Phase bit:       0   0   0   1   1   1   0 ...  (flips each lap)
```

Warp specialization introduces a subtle hazard with CTA-wide synchronization. The CUDA primitive `sync_threads()` (PTX `bar.sync 0`) requires every thread in the CTA to arrive before anyone is released. If the TMA warp and the MMA warps are in different `if/else` branches, a `sync_threads()` call inside the MMA branch will spin forever waiting for the TMA warp — which is executing an entirely different branch and never reaches the barrier. This is a deadlock, and it is the reason why `sync_threads()` cannot be used freely inside warp-specialized kernels even for something as simple as staging accumulator results through SMEM before a TMA store.

The fix is `cute.arch.barrier(barrier_id, number_of_threads)`, which maps to PTX `bar.sync n, cnt`. There are 16 independent named barriers (IDs 0–15). Each barrier releases all its participants as soon as exactly `number_of_threads` threads have arrived — threads outside that count never need to participate. Setting `number_of_threads` to cover only the MMA or epilogue warps lets those warps synchronize among themselves without any involvement from the TMA warp. In [`d1_tcgen05_tma_umma.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d1_tcgen05_tma_umma.py), there is no warp specialization and all 128 threads participate in the same code paths, so `cute.arch.barrier(barrier_id=1, number_of_threads=128)` is functionally equivalent to `sync_threads()`. Calling `cute.arch.barrier(barrier_id=1)` without the count argument produces the same full-CTA synchronization. In [`d2_tcgen05_tma_umma_specialized_pipeline.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d2_tcgen05_tma_umma_specialized_pipeline.py), the `TmemAllocator` wrapper handles the TMEM lifecycle barriers automatically, so you do not need to manage the counts manually.


### 5.2. PipelineAsync (Synchronous Writes + Synchronous MMA)

PipelineAsync is a generic pipeline class where both the producer and consumer are AsyncThreads. Usage is quite simple and intuitive.
Script `a2_smem_pipeline.py` use this apis to overlap vanilla gmem -> smem copy and naive MMA.

```python
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

### 5.3. PipelineTmaAsync (Async TMA Loads + Async WGMMA)

Used on Hopper for TMA producers and AsyncThread consumers. TMA completion is tracked via transaction byte counting: the `tx_count` parameter tells the pipeline how many bytes to expect per stage and TMA hardware decrements the barrier counter as bytes land in SMEM, making `producer_commit()` effectively a NOP. WGMMA completion is tracked via `commit_group()`/`wait_group()` pairs. Because WGMMA continues running after `cute.gemm` returns, the consumer needs two separate state trackers: `consumer_read_state` (advances when data is ready to consume) and `consumer_release_state` (advances only after `wait_group` confirms WGMMA has finished reading that SMEM stage). The producer uses a prefetch phase to fill all S stages before the steady-state loop, maximizing latency hiding.

```python
pipeline = PipelineTmaAsync.create(
    num_stages=S,
    producer_group=CooperativeGroup(Agent.Thread, num_tma_warps),
    consumer_group=CooperativeGroup(Agent.Thread, num_mma_warps),
    barrier_storage=mbar_ptr,
    tx_count=tma_transaction_bytes,
    cta_layout_vmnk=cute.make_layout((1, *cluster_shape)),
)
producer_state         = make_pipeline_state(PipelineUserType.Producer, S)
consumer_read_state    = make_pipeline_state(PipelineUserType.Consumer, S)
consumer_release_state = make_pipeline_state(PipelineUserType.Consumer, S)

# ── TMA warp (producer) ────────────────────────────────────────────────────
if is_tma_warp:
    for kidx in range(S):                                # prefetch first S stages
        pipeline.producer_acquire(producer_state)
        bar = pipeline.producer_get_barrier(producer_state)
        cute.copy(tma_atom_a, tAgA[None, producer_state.count],
                  tAsA[None, producer_state.index], tma_bar_ptr=bar)
        pipeline.producer_commit(producer_state)         # NOP — TMA hw signals via tx_count
        producer_state.advance()
    for kidx in range(S, K // BK):                      # steady-state
        pipeline.producer_acquire(producer_state)
        bar = pipeline.producer_get_barrier(producer_state)
        cute.copy(tma_atom_a, tAgA[None, producer_state.count],
                  tAsA[None, producer_state.index], tma_bar_ptr=bar)
        pipeline.producer_commit(producer_state)
        producer_state.advance()

# ── MMA warps (consumer) ───────────────────────────────────────────────────
if is_mma_warp:
    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

    for kidx in range(K // BK):
        pipeline.consumer_wait(consumer_read_state)   # wait for tx_count → 0

        cute.nvgpu.warpgroup.fence()

        for k_block_idx in range(num_k_blocks):
            cute.gemm(
                tiled_mma,
                accumulators,
                tCrA[None, None, k_block_idx, consumer_read_state.index],
                tCrB[None, None, k_block_idx, consumer_read_state.index],
                accumulators,
            )
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        cute.nvgpu.warpgroup.commit_group()           # seal all WGMMA in one group
        cute.nvgpu.warpgroup.wait_group(0)            # wait until group finishes

        pipeline.consumer_release(consumer_release_state)
        consumer_read_state.advance()
        consumer_release_state.advance()
```

```
Stage lifecycle (S stages, prefetch fully loaded before steady-state)
──────────────────────────────────────────────────────────────────────
TMA warp (producer)                   MMA warps (consumer)
      │                                      │
      ├─ producer_acquire()                  │
      ├─ TMA copy → sA/sB[stage]             │
      │    hw decrements tx_count            │
      ├─ producer_commit()  [NOP]            │
      │    tx_count hits 0 ─────────────►    ├─ consumer_wait() unblocks
      │                                      ├─ warpgroup.fence()
      │                                      ├─ WGMMA reads sA/sB (async)
      │                                      ├─ commit_group() → seals group
      │                                      ├─ wait_group(0)  → MMA done
      │  ◄───────────────────────────────────├─ consumer_release()
```

#### Why Deeper Pipelines?

With the lifecycle diagram above in mind, the natural question is: why does a 7-stage pipeline outperform a 3-stage pipeline? The answer comes down to a fundamental asymmetry between the producer and the consumer. In a typical GEMM tile, the time TMA spends loading sA and sB from DRAM is longer than the time the tensor cores spend consuming that data — this is ever more true with the fast WGMMA and UMMA instructions, this asymmetry is shown clearly in the figures in section 7-Profiling. Our GEMM kernels are strongly memoy-bound (GMEM latency >> compute per tile), more stages let the copy engine keep feeeding data while the tensor cores are still crunching pre-fetched stages in smem. More stages will allow more TMEM in-flight to saturate the HBM bandwidth longer which keeps the TMEM engines busy, and the 7 pre-fetched stages keep the 4 tensor cores per SM busy. A 7-stage pipeline can have up to 7 tiles already in-flight, so the stall between successive loads is almost eliminated, almost like using higher throughput to hide latency given large problem size (K >> BK). On contrary, if the kernel doesn't run long enough due to small K dimension, the cost of prefilling the 7 stages will outweight the benefit, .i.e. K//BK = 8, WGMMA waits until 7 stages is finished until it can starts which is essentially serializing TMA and WGMMA and there's no memory/computation overlap happening. So it's important to only use deep pipeline when K is large enough to amortize the prefilling latency.

### 5.4. PipelineTmaUmma (Async TMA Loads + Async tcgen05 UMMA)

On Blackwell, we use PipelineTmaUmma for TMA producers and UMMA consumers. Compared with Hopper pipeline, the TMA producer side is identical, but the consumer side is a single MMA warp issuing tcgen05 UMMA instructions via `cute.gemm`. The key architectural difference is that the accumulator lives in TMEM rather than registers. Because UMMA writes to TMEM and not to SMEM, the consumer can call `ab_full.release()` immediately after issuing all UMMA instructions for a stage — the hardware guarantees SMEM is no longer being read even while TMEM is still being written. A separate mbarrier (`mma_mbar`) tracks UMMA completion independently: `tcgen05.commit(mma_mbar)` causes the hardware to arrive at the barrier once all pending TMEM writes have landed, and the epilogue warps call `mbarrier_wait(mma_mbar, 0)` before touching TMEM.

PipelineTmaUmma also exposes the simpler `acquire_and_advance` / `wait_and_advance` API instead of the explicit `producer_acquire` / `producer_get_barrier` / `producer_commit` triplet used in PipelineTmaAsync. The structure is similar to `PipelineAsync`.

```python
producer, consumer = pipeline.PipelineTmaUmma.create(
    num_stages=num_stages,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
    tx_count=tma_transaction_bytes,
    barrier_storage=storage.mbar_ptr.data_ptr(),
    cta_layout_vmnk=cta_layout_vmnk,
).make_participants()

# TMA warp (producer)
if warp_idx == tma_warp_id:
    for kidx in range(num_k_tiles):
        ab_empty = producer.acquire_and_advance()   # wait until stage is free
        cute.copy(tma_atom_a, tAgA[None, ab_empty.count],
                  tAsA[None, ab_empty.index], tma_bar_ptr=ab_empty.barrier)
        cute.copy(tma_atom_b, tBgB[None, ab_empty.count],
                  tBsB[None, ab_empty.index], tma_bar_ptr=ab_empty.barrier)
        # No commit() call — TMA hw signals completion via tx_count

# MMA warp (UMMA consumer)
if warp_idx == mma_warp_id:
    for kidx in range(num_k_tiles):
        ab_full = consumer.wait_and_advance()       # wait for TMA data in SMEM
        for k_block_idx in range(num_k_blocks):
            cute.gemm(tiled_mma, tCtAcc,
                      tCrA[None, None, k_block_idx, ab_full.index],
                      tCrB[None, None, k_block_idx, ab_full.index],
                      tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        ab_full.release()                           # SMEM stage safe to refill

    # Signal UMMA completion via hardware-tracked mbarrier
    if tidx == mma_warp_id * threads_per_warp:
        tcgen05.commit(mma_mbar)   # arrives at mma_mbar when all TMEM writes land

# Epilogue: all warps wait for UMMA to finish writing TMEM
cute.arch.mbarrier_wait(mma_mbar, 0)
```

[`d2_tcgen05_tma_umma_specialized_pipeline.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d2_tcgen05_tma_umma_specialized_pipeline.py) provides the complete 6-warp implementation: 4 epilogue warps (TMEM → RMEM → GMEM), 1 MMA warp, and 1 TMA warp. [`d3_tcgen05_tma_umma_2cta_specialized_pipeline.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d3_tcgen05_tma_umma_2cta_specialized_pipeline.py) extends this to a 2-CTA cluster with TMA multicast.

</details>

<details>
<summary>6. TV Layout for Thread-Register to MN Coordinate Mapping and Layer Fusion</summary>

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
Printing the atom and tiled MMA for the warp-level `F16/BF16 → F32` instruction (script [`b2_wmma_smem.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py)) gives:

```
TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
```

- `(4,8)` → 32 threads (one warp)
- `(2,2)` → 4 registers per thread for the accumulator fragment

After tiling this atom across multiple warps, the tiled TV layout becomes much larger. For example, with a 4x4 warp arrangement (512 threads) as demonstrated in [`z1_tv2mn.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/z1_tv2mn.py):

```
tv_layout_C_tiled:
((4,8,4,4),((2,2),(1,2))):((128,1,16,512),((64,8),(0,2048)))
```

To loop over each accumulator register and recover its logical `(m, n)` coordinate:

```python
# z1_tv2mn.py / b2_wmma_smem.py L239 — accessing tv_layout_C_tiled
for reg_idx in range(cute.size(tCrC_out)):
    coord = cute.idx2crd((tid, reg_idx), tv_layout_C_tiled.shape)
    mn_flat = cute.crd2idx(coord, tv_layout_C_tiled)
    m, n = cute.idx2crd(mn_flat, fragC_layout.shape)
```

For example, with `tid=11` and `reg_idx=3`, `coord` is `((3,2,0,0),((1,1),(0,0)))`, `mn_flat` is `1547`, and the resulting logical coordinate is `m=11, n=24`. See [`z1_tv2mn.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/z1_tv2mn.py) for a runnable demo.

This same do-not-materialize pattern underpins the Implicit GEMM algorithm, where the `im2col` matrix is computed on the fly tile-by-tile and loaded into SMEM, allowing convolution to use tensor cores efficiently and minimizing round-trip global-memory reads and writes.

The TV Layout abstraction serves a second purpose beyond register coordinate mapping: it also describes how threads collectively load data from GMEM to SMEM in vectorized patterns. In [`b2_wmma_smem.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py), rather than relying on CuTeDSL's default copy assignment, the kernel defines an explicit TV layout for GMEM to SMEM loading. The thread layout `tA` assigns each thread to a strip of the K dimension, and the value layout `vA` specifies that each thread loads `num_vectorized=4` consecutive fp16 values as one 64-bit transaction. Passing both to `cute.make_tiled_copy_tv(atom_copy_A, tA, vA)` produces a TiledCopy that issues 64-bit wide loads from GMEM rather than 4 16-bit ones, cutting the number of memory transactions by 4x.

```python
# From b2_wmma_smem.py L85-L93 — K-major vectorized GMEM → SMEM copy layout
num_vectorized = 4
major_mode_size = self._bK // num_vectorized  # 64 // 4 = 16
tA = cute.make_layout(
    shape=(self._num_threads // major_mode_size, major_mode_size),
    stride=(major_mode_size, 1)
)
vA = cute.make_layout(shape=(1, num_vectorized), stride=(0, 1))
tiled_copy_A = cute.make_tiled_copy_tv(atom_copy_A, tA, vA)
```

</details>

<details>
<summary>7. Profiling Async Pipelines with Inline PTX Probing</summary>

To verify that our async pipeline is actually overlapping memory and compute, we need timing measurements inside the kernel. Taking inspiration from [gau-nernst's blog post](https://gau-nernst.github.io/tcgen05/), the probing technique uses **inline PTX** to read the `globaltimer` register, which returns the current GPU clock in nanoseconds. CuTeDSL exposes this via LLVM IR, which lets you embed raw PTX instructions directly in Python kernel code:

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

**Important caveats about the profiling kernels ([`a2_smem_pipeline_profile.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_pipeline_profile.py), [`c2_profile.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c2_profile.py)):**
- Only **2 warps per block** are launched — very low occupancy by design so the timeline is readable.
- The SMEM loading is the main bottleneck; it can be accelerated with vectorized loads or TMA.
- The MMA completes very quickly (low Arithmetic Intensity = FLOPS / bytes transferred), so the pipeline profile shows a memory-bound workload.

**Profile of PipelineAsync (a2):**

The figure below shows producer (SMEM load) and consumer (MMA) timelines per warp. With only 2 warps there is limited overlap, but the barrier handoff between producer and consumer is clearly visible.

![PipelineAsync profile](./assets/a2_pipeline_profile.png)

**Profile of PipelineTmaAsync (c2, 64x128x64 tile):**

Running [`c2_profile.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c2_profile.py) and visualizing on Perfetto shows the first 3 prefetched TMA loads (filling the pipeline), followed by well-overlapped WGMMA compute and TMA loads in the steady state. Launch overhead is still visible at the start.

![PipelineTmaAsync profile](./assets/c3_pipeline_profile_64x128x64.png)

</details>

---
TFLOPS progression (M=N=K=4096, dtype=float16):
| Techniques | Script | Architecture | SM90 | SM100 | SM120 |
|---|---|---|---|---|---|
| Naïve | [`a1`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a1_naive_cute.py) | Any | 0.58 | similar | similar |
| Shared memory | [`a2`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/a2_smem_cuda_like.py) | Any | 7 | similar | similar |
| WMMA CUDA C++ | [`b2.cuh`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cuda/kernels/b2_wmma_smem_vec_2D.cuh) | SM80+ | 94 | 197 | 257 |
| WMMA CuTeDSL | [`b2.py`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b2_wmma_smem.py) | SM80+ | 203 | 241 | 295 |
| WMMA + TMA | [`b5`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b5_wmma_tma_load_store.py) | SM90+ | 355 | 324 | 335 |
| WMMA + TMA warp-specialized pipeline | [`b7`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/b7_wmma_tma_specialized_pipeline.py) | SM90+ | 392 | 424 | 343 |
| WGMMA + TMA | [`c1`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c1_wgmma_tma_load_store.py) | SM90 | 532 | - | - |
| WGMMA + TMA warp-specialized pipeline | [`c2`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/c2_wgmma_tma_specialized_pipeline.py) | SM90 | 685 | - | - |
| tcgen05 MMA + TMA | [`d1`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d1_tcgen05_tma_umma.py) | SM100 | - | 717 | - |
| tcgen05 MMA + TMA warp-specialized pipeline | [`d2`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d2_tcgen05_tma_umma_specialized_pipeline.py) | SM100 | - | 1279 | - |
| tcgen05 MMA 2CTA + TMA warp-specialized pipeline | [`d3`](https://github.com/luongthecong123/learn-cutedsl/blob/main/cutedsl/d3_tcgen05_tma_umma_2cta_specialized_pipeline.py) | SM100 | - | 1366 | - |

To close the remaining gap to cuBLAS (~720 TFLOPS on H100, ~1500 TFLOPS on B200), the next step is persistent kernels — a technique that keeps the CTA alive across multiple output tiles, overlapping the epilogue of the current tile with the computation of the next to hide GMEM store latency and eliminate kernel launch overhead. On Hopper, kernel c2 would gain from TMA multicast and CTA clusters that broadcast shared tiles across cooperating CTAs, reducing L2 and HBM traffic. On Blackwell SM100, kernel d3 can exploit TMEM staging: split the 512 TMEM columns into two halves so the epilogue reads from the first 256 columns while the next tile's MMA accumulates into the second 256, achieving true compute-epilogue overlap within a single SM. On SM120 (Blackwell RTX), which lacks WGMMA and UMMA, kernel b7 can benefit from register double-buffering — producer threads fill next-stage fragment registers via LdMatrix while consumer threads execute MMA on the current-stage registers, with a partial-CTA barrier (number_of_threads=num_producer_threads) avoiding the deadlock that a full sync_threads() would cause.

## Job Submission
All files in this repo can be run directly and independantly as each file is self contained (`cuda/gemm_cuda.py` or all files in `cutedsl/*.py`) if you have the hardware at your disposable. Or you can submit these jobs to the cloud to have access to beefier server class GPUs.

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
11. https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
