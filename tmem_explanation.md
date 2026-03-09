# Understanding Tensor Memory (TMEM) on Blackwell

## What is TMEM?

On NVIDIA's Blackwell architecture (sm_100), the MMA accumulator no longer lives in the register file. Instead, Blackwell introduces a dedicated on-chip memory called **Tensor Memory (TMEM)**. Every SM has its own TMEM, and its sole purpose is to hold the results of matrix multiply-accumulate operations. When you issue a `cute.gemm` with a tcgen05 MMA, the accumulator values are written directly into TMEM — not into per-thread registers like on Hopper.

This has a profound implication for the epilogue: you can no longer just read the accumulator from registers after the MMA loop. Instead, you need an explicit data-movement step to copy the results out of TMEM into registers before you can cast, transform, or store them to global memory. Understanding TMEM's structure and the APIs for moving data in and out of it is essential for writing any Blackwell kernel.

TMEM is organized as a 2D array of **128 lanes (rows)** and **512 columns**. Each cell holds a 32-bit value. That gives 128 × 512 × 4 bytes = **256 KB per SM**. You have to allocate all 128 lanes — there is no way to allocate a subset of rows. The allocation unit is **columns only**, and the number of columns must be a power of two: 32, 64, 128, 256, or 512. The minimum is 32 columns (16 KB). Even if your accumulator only requires 64×64 fp32 values, you still must allocate 64 columns across all 128 rows — you cannot allocate fewer than 128 rows.


## TMEM Lifecycle: Allocation, Barriers, and Deallocation

Before TMEM can be used, it must be allocated. After the kernel is done with it, it must be deallocated. Between these two steps, a careful barrier protocol ensures all threads agree on the TMEM base address before anyone tries to read or write it.

The full lifecycle, extracted from `d1_tcgen05_tma_umma.py`, looks like this:

```python
# 1. Allocate — must be called from an entire warp (all 32 threads)
if warp_idx == 0:
    cute.arch.alloc_tmem(cutlass.Int32(num_cols), storage.tmem_holding_buf)

# 2. Barrier — ensure the SMEM write from alloc is visible to all threads
cute.arch.barrier(barrier_id=1, number_of_threads=128)

# 3. Retrieve — every thread reads the TMEM base address from SMEM
tmem_ptr = cute.arch.retrieve_tmem_ptr(
    cutlass.Float32, alignment=16,
    ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

# ... tcgen05 MMA loop accumulates results into TMEM ...
# ... Epilogue: tcgen05.ld reads TMEM → registers, cast, store to GMEM ...

# 4. Relinquish — promise to hardware: no more allocations from this CTA
if warp_idx == 0:
    cute.arch.relinquish_tmem_alloc_permit()

# 5. Barrier — ensure all epilogue reads from TMEM are complete
cute.arch.barrier(barrier_id=1)

# 6. Deallocate — must be the same warp that allocated
if warp_idx == 0:
    cute.arch.dealloc_tmem(tmem_ptr, cutlass.Int32(num_cols))
```

Both `alloc_tmem` and `dealloc_tmem` must be called from the same warp. The `alloc_tmem` call writes the TMEM base address into a shared memory slot (`storage.tmem_holding_buf`). The barrier at step 2 ensures this write is visible before other threads call `retrieve_tmem_ptr` to read it. After the epilogue, `relinquish_tmem_alloc_permit` tells the hardware this CTA is done allocating, and the second barrier at step 5 ensures all epilogue TMEM reads are finished before `dealloc_tmem` frees the columns for other CTAs.


## TMEM Memory Movement

Typically, data gets into TMEM via UMMA operations, and is explicitly moved out to registers using `tcgen05.ld` for post-processing. It's also possible for threads to manually load data into TMEM, either from SMEM through `tcgen05.cp` or from registers through `tcgen05.st`. However, TMEM access patterns for explicit load and store are very restricted.

There are three memory movement instructions under tcgen05:

**tcgen05.ld** copies data from TMEM into per-thread registers. This is the most common explicit TMEM instruction — every Blackwell GEMM epilogue uses it to pull the accumulator out of TMEM so it can be cast, scaled, or otherwise transformed in registers before writing to global memory.

**tcgen05.st** copies data from registers into TMEM — the reverse direction. It is used when you need to write computed values back into TMEM, for example to initialize or update the accumulator from register values.

**tcgen05.cp** copies data directly from shared memory into TMEM without going through registers. It appears in specialized scenarios like FlashAttention-style kernels where softmax correction values need to be loaded into TMEM for the next MMA round.

Since `tcgen05.ld` is by far the most commonly used — appearing in every kernel epilogue — the rest of this section focuses on how it works.

### Understanding the load instruction shape

The `tcgen05.ld` instruction has a shape encoded in its name: `Ld{DP}x{BITS}bOp`. The first number, **DP** (data-path lanes), tells you how many TMEM lanes participate per warp. The second number, **BITS**, tells you how many bits are read from each lane per repetition.

In CuTe DSL, you also specify a **Repetition** count — how many times the instruction repeats across TMEM columns. Each repetition loads `BITS / 32` fp32 columns from each participating lane. The repetition count directly determines how many fp32 registers each thread consumes.

The two most common load shapes are:

**`Ld32x32bOp`** uses 32 data-path lanes and reads 32 bits (one fp32) per lane per repetition. Since there are 32 threads in a warp and 32 DP lanes, each thread maps 1:1 to a TMEM lane. With 4 warps covering 4 × 32 = 128 lanes, this shape requires `MMA_M=128`. Each repetition loads 1 fp32 column, so `Repetition(R)` means each thread holds R registers and R TMEM columns are consumed.

**`Ld16x256bOp`** uses 16 data-path lanes and reads 256 bits (eight fp32 values) per lane per repetition. The 16 DP lanes are distributed across 32 threads in a warp — two threads collaborate to carry the 256 bits from each lane. With 4 warps covering 4 × 16 = 64 lanes, this shape requires `MMA_M=64`. Each repetition loads 8 fp32 columns (256 bits / 32 bits), so `Repetition(R)` consumes R × 8 columns.

The choice of instruction is determined by `num_dp`, the number of data-path lanes per warp. For `cta_group::1` with 4 warps along M (`tmem_warp_shape = (4,1)`), this is simply `num_dp = MMA_M / 4`:

```python
num_dp = M_acc // 4
if num_dp == 32:   # MMA_M=128 → Ld32x32bOp,  1 fp32 col per rep
    ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition(tmem_ld_rep))
    fp32_cols_per_rep = 1
elif num_dp == 16:  # MMA_M=64  → Ld16x256bOp, 8 fp32 cols per rep
    ld_op = tcgen05.Ld16x256bOp(tcgen05.Repetition(tmem_ld_rep))
    fp32_cols_per_rep = 8
```

Getting this wrong — for example using `Ld32x32bOp` with M=64 — will cause a layout mismatch error because the instruction expects 32 DP lanes but only 16 are available.

### Subtiling and register pressure

After the MMA loop, the full accumulator sits in TMEM. For a tile of (128, 256), that's 128 × 256 = 32,768 fp32 values. With 128 threads, loading the entire thing at once would require 256 registers per thread — right at the hardware limit. In practice, you split the accumulator into **subtiles** and loop over them, loading one subtile per iteration.

The key relationship is:

```
subtile_n = tmem_ld_rep × fp32_cols_per_rep     (columns per subtile)
subtile_cnt = N_acc / subtile_n                  (number of loop iterations)
regs_per_thread = tmem_ld_rep                    (register pressure per iteration)
```

For `Ld32x32bOp` with `tmem_ld_rep=8` on a (128, 256) tile: each subtile covers 8 columns, the loop runs 32 times, and each thread uses 8 registers per iteration. For `Ld16x256bOp` with `tmem_ld_rep=4` on a (64, 256) tile: each subtile covers 32 columns, the loop runs 8 times, and the register pressure accounting is different because 256 bits are split across two threads per lane.

Choosing the repetition count is a trade-off: larger repetitions reduce loop iterations (less instruction overhead) but increase register pressure. Our benchmarks on B200 with a (128, 256) tile and `Ld32x32bOp` show diminishing returns past `Repetition(16)`:

| Repetition | Regs/thread | Subtile count | TFLOPS |
|:---:|:---:|:---:|:---:|
| x1 | 1 | 256 | 421.1 |
| x2 | 2 | 128 | 619.4 |
| x4 | 4 | 64 | 678.5 |
| x8 | 8 | 32 | 708.7 |
| x16 | 16 | 16 | 713.8 |
| x32 | 32 | 8 | 701.3 |
| x64 | 64 | 4 | 711.9 |
| x128 | 128 | 2 | 704.9 |

In this no-overlap kernel, the epilogue is a small fraction of total time, so the differences plateau quickly. In pipelined kernels with complex epilogues — like FlashAttention's softmax correction and rescaling — keeping register pressure low with smaller repetitions becomes critical to avoid register spilling.


## TMEM in the 2CTA Case

When two CTAs cooperate on a single MMA instruction (`CtaGroup.TWO`), each CTA still allocates its own TMEM independently — the allocation API is the same. What changes is how the MMA writes results: the leader CTA (V=0) issues the MMA instruction, and the hardware internally coordinates writing into **both CTAs' TMEM**. Each CTA ends up with its portion of the result in its own local TMEM, even though only one CTA issued the instruction.

The epilogue is also per-CTA — each CTA reads from its own TMEM and stores its own portion of the output to global memory. The only difference is in deallocation: since two CTAs share the cooperative MMA, they need a cross-CTA barrier (an mbarrier) to ensure both CTAs have finished their epilogue reads before either deallocates. In `d3_tcgen05_tma_umma_2cta_specialized_pipeline.py`, the `TmemAllocator` handles this when constructed with `is_two_cta=True`:

```python
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf,
    barrier_for_retrieve=tmem_alloc_barrier,
    is_two_cta=True,
    two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
)
```

The `tmem_dealloc_mbar_ptr` is an mbarrier shared between the two CTAs in the cluster. When `tmem.free(tmem_ptr)` is called, it waits on this mbarrier before deallocating, ensuring safe cross-CTA coordination.

The load instruction selection also changes slightly when `use_2cta_instrs=True` is passed to `sm100_utils.get_tmem_load_op`. With a per-CTA M of 128 but 2CTA cooperative mapping, the utility may select different warp-to-lane mappings internally. But from the perspective of writing the epilogue, the code structure remains the same — you still build a copy atom, subtile the accumulator, loop over subtiles, and store to GMEM.
