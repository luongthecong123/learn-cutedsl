# learn-cutedsl
Adding hardware features and optimization techniques brick by brick and measure the FLOPS speed up, making it easier to add CuTeDSL to your codebase according to your needs.

Apart from educational purpose, this repo can be treated as commonly used cutedsl api examples. Where readers can pick out a feature/api and add it to their code or inject them as context to LLMs for coding assistance, thanks to LLMs being few-shot learners.

For example: Feed script a1 + a2 to LLM to spit out script a2_profile...

Disclaimer: Most of this documentation were written by human. LLMs helped proof-reading and formatting.

Below are the pros and cons that I observed while learning cutedsl:

Pros:
- Provides quality of life apis to help write efficient cuda kernels
- Easier to learn compared to CUTLASS CuTe C++ counterpart which can be daunting to start learning with its highly templated code
- Expose low level features for speed of light (SoL) optimization, you can write it the cute way or the cuda/ptx way
- Seamless integration with Pytorch with JIT compilation
- Blazing fast compilation
- Faster development cycle thanks to Python 
- Latest hardware features on new Nvidia GPUs are supported
- Great SoL examples by Junkai Wu and team

Cons:
- Too many apis to do the same thing, can be confusing and hard to master
- Lack detailed documentation on lots of apis
- Examples are too complicated due to SoL requirements.

Job submission using Ray
```bash
pip install ray
# Assume ray cluster is already created

ray job submit \
    --address 'http://localhost:8265' \
    --working-dir . \
	--runtime-env-json='{"pip":"./requirements.txt"}'\
	-- python submit_ray.py
```

Job submission using Modal:
```bash
pip install modal
python3 -m modal setup

modal submit_modal.py
```

```bash
git config --global user.name "luongthecong123"
git config --global user.email "luongthecong123@gmail.com"
```

# Learning curve
- CuTeDSL and CUDA in general, have a very steep but rewarding learning curve, so don't get frustrated the first time you do it. Try to look at example and write kernels by yourself and observe the performance speed up and what makes it speed up. Once you can wrap your head around the concept of massively parallel programming with CUDA, the next kernels will be easier to digest.
- If reader is new to CUDA, it is suggested to be familar with the first gateway example: vector addition, there are plenty of explanations online for this. Here, I provide an example code with cutedsl, CUDA has a broader example ecosystem with youtube videos and blog post explanations than cutedsl, but cutedsl is essentially a wrapper of CUDA/PTX.
- After familiar with simple vector addition, it's time to get to understand how to perform General Matrix Multiplication (GEMM) in a parallel fashion (example a1 [tidx, tidy, _ = cute.arch.thread_idx()](https://github.com/luongthecong123/learn-cutedsl/blob/ec53c071e588d166af25f1f3e6f679f798da42b0/cutedsl/a1_naive_cuda_like.py#L31)), then using shared memory to speed it up (example a2).
- Next, usage of warp matrix multiplication addition (WMMA) with tensor core, there are great blogs from Lei Mao that provides example as well as great explanation https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/ . In CUDA, b1, wmma apis are used to simplify tensor core programming, but in cutedsl or cutlass cute, the consensus is to provide a wrapper on ptx instructions, hence lower level than the CUDA C++ counterpart when it comes to tensor core programming.
- Then we can move from Ampere to Hopper GPU with Tensor Memory Accelerator (TMA) and Warp Group Matrix Multiply Addition (WGMMA) tensor core, and new barrier instructions for Asynchronous Pipelining introduced in Hopper (a true async chip).
- Then Blackwell is the next step. (And potentially Vera Rubin...)

# CuTeDSL fundamentals

## Linear indexing

When performing matrix multiplication, we can do something like A[i, j] * B[j, k] = C[i, k]. Which perform 2-D indexing, but the physical memory pointer offset is 1-D, what C++ does under the hood is to convert this to linear offset. When we write CUDA kernel, we pass just the pointer to the first element of the memory block in VRAM to the kernel (pass as reference) so we C++ doesn't create a copy of that parameter when we call the function.

## Host code and device code

API:

@cute.jit is the decorator of the host code function, it will call our device code

@cute.kernel is the decorator for the kernel code

Compilation: you can call the function in just in time (JIT) fashion or you can compiled it and store to a cache folder to be reused the next time the function is called with ahead of time (AOT) compilation

## Interfacing with Pytorch

Pytorch calls to under the hood CUDA kernels, and its tensors on GPU are just pieces of memory/data block that is sliced with pytorch's pointers. So we can use dlpack to grant access to these memory block to run custom kernels on it.
If you write code in CUDA C++, you need glue code (point to the gluecode.cu) to connect it with pytorch, here this code used separation of cuda source and pytorch source for faster compilation through removing duplicated compilation, based on the benchmarking done by Lei Wang https://github.com/LeiWang1999/CPPTorchExecutable 

cutedsl and other DSLs in general, removed this boilerplate with MLIR and NVVM, so that the compilation time is faster

## Layout
Arguebly the most important concept in CUTLASS CuTe/CuTeDSL.

API: 

Layout

TV Layout

Swizzle composed Layout

Indexing from layout

## Shared memory

API:

Shared memory allocation

## Atom

API:

Copy atom

Universal copy atom

TMA copy atom

Tiled copy atom

MMA atom

Universal MMA atom

WMMA atom

WGMMA atom

Tcgen05 atom (TODO)

Tiled mma atom

Thr_mma

# Thread-register to mn coordinate mapping and layer fusion optimization
In CUDA optimization, one can choose to optimize a specific operation like GEMM to speed of light, which is more tedious and time-consuming. Or one can choose to optimize ...

# Tensor Memory Accelerator (TMA)

Hopper and above provides a memory accelerator called TMA. 
Normally, without TMA, the flow to store data to shared memory is: GMEM -> register -> SMEM, which requires register allocation. TMA bypass the register step and store data straight to SMEM asynchronously, hence leaves us with more register to program with.
Example c1 provides a way to use TMA with WMMA, which makes this code runnable on Hopper SM90, Blackwell SM100 and SM120. Using TMA as a drop in replacement for SMEM loading provides a nice speedup.

# Warp group matrix multiplication (WGMMA)
Hopper also provides a faster tensor core instruction called WGMMA, which requires a warpgroup (4 warps) to issue. This instruction is also async, combining TMA and WGMMA with the new barriers described in the next section provides neat pipeline to overlap computation with data copy.

# Compare PipelineAsync and PipelineTmaAsync

Hopper architecture introduces new barrier primitives that help us overlap memory transactions and computation efficiently. CuTeDSL/CUTLASS provides a convenient way to do this through their `PipelineAsync` and `PipelineTmaAsync` APIs.

## Warp Specialization

We call computation the **"consumer"** and memory transaction the **"producer"**, and assign these roles to different warps. On NVIDIA GPUs, a **warp** is a group of 32 threads issued together in lockstep. Individual threads within a warp taking different branches cause **warp divergence** (the branches serialize instead of executing in parallel). However, different warps within a thread block can take entirely different code paths without any performance penalty — they are independently scheduled. This is why we split work by warp, and the technique is called **warp specialization**.

```python
if warp_group_idx == 0:    # Producer warps — handle memory
    ...
if warp_group_idx == 1:    # Consumer warps — handle computation
    ...
# Different warps, different paths — no divergence penalty
```

## Pipeline Communication via Barriers in Shared Memory

Since producer and consumer warps operate independently, we need a mechanism for them to communicate:
- The producer needs to signal: *"data is ready to be consumed"*
- The consumer needs to signal: *"I'm done reading, this slot is free to overwrite"*

To achieve this, we store **`mbarrier`** (barrier) objects in **shared memory**, which is accessible by all threads within a thread block. Each pipeline stage gets its own barrier, and the stages are organized as a **circular buffer** — when we reach the last stage, we wrap back to stage 0.

Each barrier tracks a **phase** that flips between even and odd. The producer and consumer agree on which phase means "full" (data ready) and which means "empty" (slot free). A barrier completes and advances its phase when the expected number of arrivals is reached. This is what prevents the two fundamental race conditions:

- **Producer overwriting data the consumer hasn't read yet** → producer `acquire` waits on the barrier until the consumer has released that stage
- **Consumer reading a stage the producer hasn't filled yet** → consumer `wait` blocks on the barrier until the producer has committed that stage

With multiple stages, the producer can run ahead by up to `S` iterations before it must stall, allowing memory latency to be hidden behind computation.

## PipelineAsync

```python
# Setup
pipeline = PipelineAsync.create(
    num_stages=S,
    producer_group=CooperativeGroup(Agent.Thread, 128),   # 1 warpgroup
    consumer_group=CooperativeGroup(Agent.Thread, 128),   # 1 warpgroup
    barrier_storage=mbar_ptr,
)
producer, consumer = pipeline.make_participants()
```

```python
# Mainloop
if warp_group_idx == 0:  # Producer
    for k in range(K):
        handle = producer.acquire_and_advance()   # wait for stage to be free

        smem[handle.index] = data[k]              # threads write to smem

        handle.commit()                           # threads signal "data ready"

    producer.tail()                               # ensure all used buffers are properly synchronized before producer exit.

if warp_group_idx == 1:  # Consumer
    for k in range(K):
        handle = consumer.wait_and_advance()      # wait for "data ready"

        result += smem[handle.index]              # threads read from smem

        handle.release()                          # signal "stage free"
```

This is quite trivial, as these are synchronous operations (threads write to smem on the producer side, threads perform MMA using registers on the consumer side). A thread is blocked until it has finished writing data to shared memory, and the same logic applies for MMA on registers — when the instruction returns, the work is done.

The only concern is **race conditions** — one warpgroup could overwrite a shared memory stage before the other warpgroup has finished reading from it, or a consumer could read from a stage before the producer has finished filling it. The pipeline API handles this for us: `acquire` blocks the producer until the consumer has released a stage, and `wait` blocks the consumer until the producer has committed to a stage. The circular buffer of `S` stages allows the producer to run up to `S` iterations ahead of the consumer, hiding memory latency while preventing any data hazards.

## PipelineTmaAsync

To take advantage of new async hardware accelerators like TMA and WGMMA, new barrier primitives are introduced. Even though these operations are asynchronous (the issuing thread returns immediately), we need a way to know when the hardware has actually finished — without spinning or blocking the thread unnecessarily.

**For TMA**, the mechanism is **transaction byte counting** on barriers. When creating the pipeline, we specify `tx_count` — the number of bytes we expect TMA to write into each stage. Each `cute.copy` with a `tma_bar_ptr` tells the TMA hardware: *"when you finish writing these bytes to smem, decrement this barrier's transaction count."* The barrier only completes its phase when both the expected thread arrivals reach zero **and** the transaction byte count reaches zero. This means the producer thread doesn't need to signal anything — the TMA hardware does it automatically, which is why `producer_commit()` is a NOP.

**For WGMMA**, the mechanism is **commit groups and group counting**. WGMMA is also asynchronous — `cute.gemm` returns immediately while the Tensor Cores work in the background, reading from smem. We need to know when WGMMA has finished reading from a given smem stage before we can release it back to the producer. The three primitives for this are:

| Primitive                    | Purpose                                                           |
|------------------------------|-------------------------------------------------------------------|
| `warpgroup.fence()`          | Ensures prior memory operations are ordered before WGMMA          |
| `warpgroup.commit_group()`   | Seals all WGMMA instructions issued since the last commit into one group |
| `warpgroup.wait_group(N)`    | Blocks until at most `N` committed groups are still in flight     |

`commit_group()` help group a bunch of WGMMA calls into 1 call. For example, we want WGMMA to process 16 elements in K dimension (WGMMA_K = 16), but the tile shape in the K dimension is 64 (BK = 64), meaning we have to iterate 4 times through this BK dimension. And finally, `commit_group()` help us group these 4 consecutive WGMMA instructions into one group.

Calling `wait_group(1)` means: *"the group I just committed can still be running, but the group from the previous iteration must be done."* This allows one WGMMA group to overlap with pipeline housekeeping (release, wait, acquire) for the next stage — which is critical for hiding Tensor Core latency.

Because the consumer releases a stage **one iteration behind** where it reads, it needs **two separate state trackers**: `consumer_read_state` (which stage to read next) and `consumer_release_state` (which stage to release next, lagging behind).

```python
# Setup
mainloop_pipeline = PipelineTmaAsync.create(
    num_stages=S,
    producer_group=CooperativeGroup(Agent.Thread, 128),
    consumer_group=CooperativeGroup(Agent.Thread, 128),
    barrier_storage=mbar_ptr,
    tx_count=tma_copy_bytes,          # expected bytes per stage — barrier tracks this
    cta_layout_vmnk=cta_layout_vmnk,
)
producer_state = make_pipeline_state(PipelineUserType.Producer, S)
consumer_read_state = make_pipeline_state(PipelineUserType.Consumer, S)
consumer_release_state = make_pipeline_state(PipelineUserType.Consumer, S)
```

```python
# Mainloop

# ── Warpgroup 0: Producer (TMA) ───────────────────────
if warp_group_idx == 0:
    for k in range(K):
        mainloop_pipeline.producer_acquire(producer_state)  # wait for stage free

        cute.copy(tma_atom_a, gA[k], sA[producer_state.index],
                  tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                  #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  #           TMA hardware will decrement this barrier's
                  #           tx_count when bytes land in smem
                  )
        cute.copy(tma_atom_b, gB[k], sB[producer_state.index],
                  tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                  )

        mainloop_pipeline.producer_commit(producer_state)   # NOP — TMA signals via
        producer_state.advance()                            #        tx_count, not threads

# ── Warpgroup 1: Consumer (WGMMA) ─────────────────────
if warp_group_idx == 1:
    for k in range(K):
        mainloop_pipeline.consumer_wait(consumer_read_state)  # wait for tx_count == 0
                                                              # (TMA bytes all landed)

        cute.nvgpu.warpgroup.fence()                          # order memory before wgmma

        for kb in range(num_k_blocks):                        # issue wgmma — async,
            cute.gemm(tiled_mma, acc, sA[...], sB[...], acc)  # returns immediately

        cute.nvgpu.warpgroup.commit_group()                   # seal as group G_k

        cute.nvgpu.warpgroup.wait_group(1)                    # wait until G_{k-1} done
                                                              # (G_k can still be running)

        mainloop_pipeline.consumer_release(                   # release stage from G_{k-1}
            consumer_release_state)                           # safe: wgmma done reading it

        consumer_read_state.advance()
        consumer_release_state.advance()

    cute.nvgpu.warpgroup.wait_group(0)                        # drain last group
```

The timing of each stage's lifecycle looks like this:

```
                    Stage S lifecycle
                    ─────────────────

Producer (Warpgroup 0)               Consumer (Warpgroup 1)
      │                                     │
      ├─ producer_acquire(S)                │
      │   (wait for consumer_release)       │
      │                                     │
      ├─ TMA copy A → smem[S]               │
      ├─ TMA copy B → smem[S]               │
      │   (tied to barrier via tma_bar_ptr) │
      │                                     │
      ├─ producer_commit(S) [NOP]           │
      │                                     │
      │    ┌──── TMA hardware ────┐         │
      │    │ writes bytes to smem │         │
      │    │ decrements tx_count  │         │
      │    └──────────────────────┘         │
      │                                     │
      │         tx_count hits 0 ──────►     ├─ consumer_wait(S) unblocks
      │                                     │
      │                                     ├─ warpgroup.fence()
      │                                     ├─ WGMMA reads from smem[S] (async)
      │                                     ├─ warpgroup.commit_group()  → G_k
      │                                     │
      │                                     ├─ warpgroup.wait_group(1)
      │                                     │   (confirms G_{k-1} done reading
      │                                     │    smem[S-1])
      │                                     │
      │  ◄──────────────────────────────────├─ consumer_release(S-1)
      │                                     │
```

## What About SM120 (Blackwell RTX)?

SM120 has TMA but **does not have WGMMA**. It uses the standard Tensor Core MMA instructions (via `ldmatrix` + `mma`), where operands must live in **registers**, not shared memory. The producer code is identical — TMA is TMA. The consumer side is where things change.

Since MMA reads from registers, the consumer must explicitly copy data from smem to register fragments (`tCrA`, `tCrB`) via `ldmatrix` before issuing the MMA. Both `ldmatrix` and MMA are **synchronous** — when the instruction returns, the data has been fully read from smem (for `ldmatrix`) or the computation is done (for MMA). This means:

- **No `fence` / `commit_group` / `wait_group` needed** — there is no async compute to track.
- **Only one consumer state tracker** — the stage can be released as soon as the last `ldmatrix` from that stage completes, because MMA never touches smem. No need to delay release by one iteration.
- **`consumer_release` happens inside the k_block loop**, right at the boundary when the last k_block of a stage finishes its `ldmatrix` copy.

Another pattern in the SM120 code is **double-buffering the smem → register copy with MMA computation**: the `ldmatrix` for the **next** k_block is issued **before** the MMA for the **current** k_block. This overlaps the `ldmatrix` latency with useful MMA compute — classic software pipelining that WGMMA doesn't need because it reads directly from smem.

For the full SM120 implementation, readers can refer to [Junkai Wu's dense_gemm example for SM120](https://github.com/NVIDIA/cutlass/blob/main/python/examples/blackwell_rtx/dense_gemm.py).

# Profiling asynchronous warp specialization with probing

To know if our async pipeline is actually doing overlapping, we need to be able to measure them. Take inspiration from gau-nernst's blog post, I re-implemented his probing code in cutedsl, which requires the usage of inline ptx to call a globaltimer instruction with returns the current clock time.

<p align="center">
  <img src="./assets/a2_pipeline_profile.png" width="600"><br>
  <em>Figure 1: Naive CUDA-like kernel execution flow.</em>
</p>

# Blackwell tcgen05 matrix multiplication

TODO

# Reference:
1. https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL
2. https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
3. https://research.colfax-intl.com/tutorial-hopper-tma/
4. https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
5. https://gau-nernst.github.io/tcgen05/
6. https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
7. https://github.com/LeiWang1999/CPPTorchExecutable
8. https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html
