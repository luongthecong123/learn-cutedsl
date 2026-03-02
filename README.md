# learn-cutedsl
Adding hardware features and optimization techniques brick by brick and measure the FLOPS speed up, making it easier to add CuTeDSL to your codebase according to your needs.

Apart from educational purpose, this repo can be treated as commonly used cutedsl api examples. Where readers can pick out a feature/api and add it to their code or inject them as context to LLMs for coding assistance, thanks to LLMs being few-shot learners.

For example: Feed script a1 + a2 to LLM to spit out script a2_profile...

Disclaimer: Most of this documentation were written by human. LLMs helped proof-reading and formatting.

Below are the pros and cons that I observed while learning cutedsl:

Pros:
- Provides quality of life apis to help write efficient cuda kernels
- Easier to learn compared to CUTLASS CuTe C++ counterpart which can be daunting to start learning with its highly templated code
- Expose low level features for speed of light (SoL) optimization, you can write it the cute way or the cuda/ptx way (very versitile)
- Seamless integration with Pytorch with JIT compilation
- Blazing fast compilation
- Faster development cycle thanks to Python 
- Latest hardware features on new Nvidia GPUs are supported
- Great SoL examples by Junkai Wu and team

Cons:
- Too many apis to do the same thing, can be confusing and hard to master (a trade-off of versibility)
- Lack detailed documentation on lots of apis
- Examples are too complicated due to SoL requirements.

In overview, the first section provides beginners with a learning roadmap for becoming familiar with CUDA/CuTeDSL. The second section CuTeDSL Fundametal will explain important/commonly-used APIs, namely: Layout, .... Next ...

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
Reference to the exact line a2 wizzled and c3 swizzle

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

# TV Layout for Thread-register to mn coordinate mapping and layer fusion optimization
In CUDA optimization, one can choose to optimize a specific operation like GEMM to speed of light, which is more tedious and time-consuming. Or one can choose to optimize ...

For example, in a kernel that I wrote to optimize RNN, that managed to speed it up 90-110x over Pytorch, the key is to not materialize the resulted large matC, then perform a GEMV (matrix vector multiplication) afterwards, and save the smaller GEMV result back to global memory, this requires knowing how to map Thread Register of accumulator fragment/register of tensor core op to logical m, n coordinate. [Usage in custom RNN kernel](https://github.com/chongxi/rnn_train_ring_attractor/blob/main/cpp/kernels/fwd_1loop_tc_idx.cuh#L5). Knowing this technique allows aggressive layer fusion that gives more speedup, since we're bypassing round trip global memory reads and writes which are the really slow.

By reading PTX documentation, one can come up with something like this, that uses modulo and integer division to do this mapping
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
Cutlass cute provides the formula to calculate these already in the form of atom.
For example, if we print out the atom and tiled of warp MMA instruction (reference b2 script), we get:

```bash
mma_op: warp-level F16/BF16 MMA Operation
  A/B data type         = Float16
  Accumulator data type = Float32
  Instruction shape MNK = (16, 8, 16)
tiled_mma: Tiled MMA
  Thr Layout VMNK: (32,4,4,1):(1,32,128,0)
  Permutation MNK: (64:1,64:1,16:1)
MMA Atom
  ThrID:           32:1
  Shape MNK:       (16,8,16)
  TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
  TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
  TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
```

In `TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))`, (4,8) has 4*8 = 32 threads, which is the number of threads in a warp, and (2,2) shows that there are 4 32-bit registers per thread are required to store this accumulation fragment. So this is for a single warp, I later tile this atom to 4x4 tiled layout, meaning I will launch 4x4x32 = 512 threads to calculate larger tensor core instruction. The tiled tv layout will be:

```bash
tv_layout_C_tiled:
((4,8,4,4),((2,2),(1,2))):((128,1,16,512),((64,8),(0,2048)))
```
`(4,8,4,4)` means I have tiled this warp layout to 4x4 shape. And `((2,2),(1,2))` means that now each thread will hold double the number of registers in this particular layout.

`tv_layout_C_tiled` is what we are looking for, it is exactly the function that we used above. And we just have to tap into this function to get the m, n that we need like in the script (reference b5 or b6). For each thread with thread idex (tid), we loop through the its allocated fragC register. 

```python
for reg_idx in range(cute.size(tCrC_out)):
    coord = cute.idx2crd((tid, reg_idx), tv_layout_C_tiled.shape)
    mn_flat = cute.crd2idx(coord, tv_layout_C_tiled)
    m, n = cute.idx2crd(mn_flat, fragC_layout.shape)
```
You can also loop through tv layout C, but the m,n received from that will be local to that warp, and needs to be converted to global m, n.

The first line, we will loop through each registers allocated for this thread for the accumulator. Then we pass the thread index to `(4,8,4,4)` and register index to `((2,2),(1,2))` to unflatten the index back to these shape, for example (reference z1_tv2mn.py), if thread id = 11, and register index = 3, then coord will be `((3, 2, 0, 0), ((1, 1), (0, 0)))`. Next step is to get the value that encodes m, n, printing mn_flat would give us `1547`, and unflatten this will give us the logical m, n coordinate of the value at the 11-th thread and the 3-th register in the accumulator, which is `m=11, n=24`.

Not materialize is also used in Implicit GEMM algorithm that calculate im2col matrix on the fly and load tiles of it to smem, resulted in convolution algorithm that can take advantage of tensor core and smem efficiently.


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

Calling `wait_group(0)` means: *"wait until all committed groups are done."* This ensures the stage can be safely released back to the producer since WGMMA has finished reading from shared memory. The wait happens **after** committing the current group but **before** releasing the stage.

Since the consumer waits for all groups to complete before releasing, it uses **two separate state trackers**: `consumer_read_state` (which stage to read next) and `consumer_release_state` (which stage to release). Both advance together after the wait completes.

```python
# Setup
num_consumer_warps = mma_warp_groups * (num_threads_per_warp_group // warp_size)
mainloop_pipeline = PipelineTmaAsync.create(
    num_stages=num_stages,
    producer_group=CooperativeGroup(Agent.Thread),
    consumer_group=CooperativeGroup(Agent.Thread, num_consumer_warps),
    barrier_storage=mbar_ptr,
    tx_count=tma_transaction_bytes,   # expected bytes per stage — barrier tracks this
    cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape))
)
producer_state = make_pipeline_state(PipelineUserType.Producer, num_stages)
consumer_read_state = make_pipeline_state(PipelineUserType.Consumer, num_stages)
consumer_release_state = make_pipeline_state(PipelineUserType.Consumer, num_stages)
```

```python
# Mainloop

# ── TMA WARP (producer) ───────────────────────────────
if is_tma_warp:
    # Prefetch: fill all pipeline stages
    for kidx in range(num_stages):
        mainloop_pipeline.producer_acquire(producer_state)

        cute.copy(
            tma_atom_a,
            tAgA[None, producer_state.count],
            tAsA[None, producer_state.index],
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state)
            #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #           TMA hardware will decrement this barrier's
            #           tx_count when bytes land in smem
        )
        cute.copy(
            tma_atom_b,
            tBgB[None, producer_state.count],
            tBsB[None, producer_state.index],
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state)
        )

        mainloop_pipeline.producer_commit(producer_state)   # NOP — TMA signals via tx_count
        producer_state.advance()

    # Continue producing for remaining K tiles
    for kidx in range(num_stages, k_tile_cnt):
        mainloop_pipeline.producer_acquire(producer_state)

        cute.copy(
            tma_atom_a,
            tAgA[None, producer_state.count],
            tAsA[None, producer_state.index],
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state)
        )
        cute.copy(
            tma_atom_b,
            tBgB[None, producer_state.count],
            tBsB[None, producer_state.index],
            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state)
        )

        mainloop_pipeline.producer_commit(producer_state)
        producer_state.advance()

# ── MMA WARPs (consumer) ──────────────────────────────
if is_mma_warp:
    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

    for kidx in range(k_tile_cnt):
        # Wait until TMA bytes for this stage have landed (tx_count == 0)
        mainloop_pipeline.consumer_wait(consumer_read_state)

        # Ensure memory ordering before issuing async WGMMA instructions
        cute.nvgpu.warpgroup.fence()

        # Issue WGMMA blocks (these return immediately)
        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
            # Build the per-k-block fragment coordinates using the consumer state index
            k_block_coord = (None, None, k_block_idx, consumer_read_state.index)
            tCrA_k = tCrA[k_block_coord]
            tCrB_k = tCrB[k_block_coord]

            cute.gemm(
                tiled_mma,
                accumulators,
                tCrA_k,
                tCrB_k,
                accumulators,
            )
            # Flip accumulate flag after first MMA
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        # Seal the issued WGMMA instructions as one commit group
        cute.nvgpu.warpgroup.commit_group()

        # Wait until all committed groups are done reading smem
        cute.nvgpu.warpgroup.wait_group(0)

        # Release the stage now that WGMMA is done reading from it
        mainloop_pipeline.consumer_release(consumer_release_state)
        consumer_read_state.advance()
        consumer_release_state.advance()
```

The timing of each stage's lifecycle looks like this:

```
                    Stage S lifecycle
                    ─────────────────

Producer (TMA warp)                  Consumer (MMA warps)
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
      │                                     ├─ warpgroup.wait_group(0)
      │                                     │   (confirms all groups done reading)
      │                                     │
      │  ◄──────────────────────────────────├─ consumer_release(S)
      │                                     │
```

**Note**: The producer uses a two-phase approach:
1. **Prefetch phase**: Fill all `num_stages` pipeline stages (0 to `num_stages-1`)
2. **Steady state**: Continue producing remaining K tiles while consumer processes

This allows the pipeline to stay full and maximize overlapping of TMA loads with WGMMA compute.

## What About SM120 (Blackwell RTX)?

SM120 has TMA but **does not have WGMMA**. It uses the standard Tensor Core MMA instructions (via `ldmatrix` + `mma`), where operands must live in **registers**, not shared memory. The producer code is identical — TMA is TMA. The consumer side is where things change.

Since MMA reads from registers, the consumer must explicitly copy data from smem to register fragments (`tCrA`, `tCrB`) via `ldmatrix` before issuing the MMA. Both `ldmatrix` and MMA are **synchronous** — when the instruction returns, the data has been fully read from smem (for `ldmatrix`) or the computation is done (for MMA). This means:

- **No `fence` / `commit_group` / `wait_group` needed** — there is no async compute to track.
- **Only one consumer state tracker** — the stage can be released as soon as the last `ldmatrix` from that stage completes, because MMA never touches smem. No need to delay release by one iteration.
- **`consumer_release` happens inside the k_block loop**, right at the boundary when the last k_block of a stage finishes its `ldmatrix` copy.

Another pattern in the SM120 code is **double-buffering the smem → register copy with MMA computation**: the `ldmatrix` for the **next** k_block is issued **before** the MMA for the **current** k_block. This overlaps the `ldmatrix` latency with useful MMA compute — classic software pipelining that WGMMA doesn't need because it reads directly from smem.

For the full SM120 implementation, readers can refer to [Junkai Wu's dense_gemm example for SM120](https://github.com/NVIDIA/cutlass/blob/main/python/examples/blackwell_rtx/dense_gemm.py).

# Profiling asynchronous warp specialization with probing

To know if our async pipeline is actually doing overlapping, we need to be able to measure them. Take inspiration from gau-nernst's blog post, I re-implemented his probing code in cutedsl, which requires the usage of inline ptx to call a globaltimer instruction with returns the current clock time. This is just a template example where we launch just 2 warps per block, which is very low in occupancy, and the smem loading is the main bottle neck here, we can speed up gmem -> smem with vectorized load or TMA. And the MMA happens too fast, meaning it's not computing a whole lot, leading to a really low Arithmetic Intensity (computation/bytes-transferred).

<p align="center">
  <img src="./assets/a2_pipeline_profile.png" width="600"><br>
  <em>Figure 1: Profile PipelineAsync.</em>
</p>

Below is the result of running script c3_wgmma_tma_specialized_pipeline_profile.py and visualize on [perfecto](https://ui.perfetto.dev), we can see the first 3 prefetched TMAs, then the WGMMA are overlapped quite nicely, although there are still launch overheads.

<p align="center">
  <img src="./assets/c3_pipeline_profile_64x128x64.png" width="600"><br>
  <em>Figure 2: Profile PipelineTmaAsync.</em>
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
