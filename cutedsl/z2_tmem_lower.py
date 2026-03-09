import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05

"""
z3_tmem_lower.py — Minimal TMEM read/write (no GEMM, no TMA, no MMA)

Data flow: GMEM → RMEM → TMEM (St) → TMEM → RMEM (Ld) → scale ×2 → GMEM

Normally, UMMA loads data to TMEM straight from SMEM. 
It can also be moved from rmem to tmem using threads like we're doing here with tcgen05.st
We can also move data from smem directly to tmem using threads with tcgen05.cp
To get data from tmem to rmem, we have to use threads for both case with tcgen05.ld

Two TMEM allocation methods:
  1. TmemAllocator wrapper (pipeline.NamedBarrier)
  2. Manual cute.arch.alloc_tmem + cute.arch.barrier

TMEM Ld/St ops used:
  - tcgen05.Ld32x32bOp: TMEM → RMEM  (PTX: tcgen05.ld.32x32b)
  - tcgen05.St32x32bOp: RMEM → TMEM  (PTX: tcgen05.st.32x32b)

TMEM hardware: 128 lanes × 512 cols per SM
TMEM addressing: lane in bits 31-16, col in bits 15-0 of 32-bit address
Physical layout: (128, N) : (65536, 1)  — lane stride = 1<<16, col stride = 1

This script allocates 32 columns in the TMEM. We can't allocate less than 32 columns.
We can only partition the TMEM tile in the column dimension.
All 128 rows (lanes) of TMEM have to be allocated as well, even if we don't actually use all the rows.
"""

ROWS = 128     # Fixed by TMEM hardware
COLS = 32      # Minimum TMEM allocation
THREADS = 128  # 4 warps


# ============================================================
# Example 1: TmemAllocator wrapper
# ============================================================
@cute.jit
def host_wrapper(c: cute.Tensor):
    kernel_wrapper(c).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


@cute.kernel
def kernel_wrapper(mC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    @cute.struct
    class SS:
        buf: cutlass.Int32

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SS)

    # --- TmemAllocator: alloc + barrier + retrieve in one abstraction ---
    barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.buf, barrier_for_retrieve=barrier)
    tmem.allocate(COLS)
    tmem.wait_for_alloc()       # barrier sync (bar.sync 1, 128)
    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

    # TMEM tensor: 128 lanes × 32 cols, physical TMEM addressing
    tT = cute.make_tensor(tmem_ptr, cute.make_layout((ROWS, COLS), stride=(65536, 1)))

    # Coordinate tensor (same shape, for partition_D to derive per-thread shape)
    cT = cute.make_identity_tensor((ROWS, COLS))

    # Ld/St copy atoms
    rep = tcgen05.Repetition(COLS)
    ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(rep), cutlass.Float32)
    st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(rep), cutlass.Float32)

    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tT)
    tiled_st = tcgen05.make_tmem_copy(st_atom, tT)

    thr_ld = tiled_ld.get_slice(tidx)
    thr_st = tiled_st.get_slice(tidx)

    # Partitions
    t_src = thr_ld.partition_S(tT)       # TMEM source (for ld)
    t_coord = thr_ld.partition_D(cT)     # coordinate dest (gives shape)
    t_dst = thr_st.partition_D(tT)       # TMEM dest (for st)
    g_part = thr_ld.partition_D(mC)      # GMEM partitioned same way

    # RMEM buffer
    r = cute.make_rmem_tensor(t_coord.shape, cutlass.Float32)

    simt = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)

    # GMEM → RMEM → TMEM → RMEM → scale → GMEM
    cute.copy(simt, g_part, r)          # GMEM → RMEM
    cute.copy(tiled_st, r, t_dst)       # RMEM → TMEM
    cute.copy(tiled_ld, t_src, r)       # TMEM → RMEM
    for i in cutlass.range(cute.size(r), vectorize=True):
        r[i] = r[i] * cutlass.Float32(2.0)
    cute.copy(simt, r, g_part)          # RMEM → GMEM

    # Cleanup
    tmem.relinquish_alloc_permit()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


# ============================================================
# Example 2: Manual cute.arch (no pipeline)
# ============================================================
@cute.jit
def host_manual(c: cute.Tensor):
    kernel_manual(c).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


@cute.kernel
def kernel_manual(mC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    @cute.struct
    class SS:
        buf: cutlass.Int32

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SS)
    tmem_alloc_cols = cutlass.Int32(COLS)

    # --- Manual TMEM lifecycle ---
    if warp_idx == 0:                     # all 32 threads in warp must participate
        # Allocate COLS columns of TMEM, and store its address in shared memory so threads in a cta can access
        cute.arch.alloc_tmem(tmem_alloc_cols, storage.buf)
    cute.arch.barrier(barrier_id=1, number_of_threads=THREADS)
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=storage.buf)

    tT = cute.make_tensor(tmem_ptr, cute.make_layout((ROWS, COLS), stride=(65536, 1)))
    cT = cute.make_identity_tensor((ROWS, COLS))

    rep = tcgen05.Repetition(COLS)
    ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(rep), cutlass.Float32)
    st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(rep), cutlass.Float32)

    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tT)
    tiled_st = tcgen05.make_tmem_copy(st_atom, tT)

    thr_ld = tiled_ld.get_slice(tidx)
    thr_st = tiled_st.get_slice(tidx)

    t_src = thr_ld.partition_S(tT)
    t_coord = thr_ld.partition_D(cT)
    t_dst = thr_st.partition_D(tT)
    g_part = thr_ld.partition_D(mC)

    r = cute.make_rmem_tensor(t_coord.shape, cutlass.Float32)
    simt = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)

    cute.copy(simt, g_part, r)
    cute.copy(tiled_st, r, t_dst)
    cute.copy(tiled_ld, t_src, r)
    for i in cutlass.range(cute.size(r), vectorize=True):
        r[i] = r[i] * cutlass.Float32(2.0)
    cute.copy(simt, r, g_part)

    # Cleanup
    if warp_idx == 0:
        cute.arch.relinquish_tmem_alloc_permit()
    cute.arch.barrier(barrier_id=1)
    if warp_idx == 0:
        cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

def main():
    C_orig = torch.randn((ROWS, COLS), device="cuda", dtype=torch.float32)
    ref = C_orig * 2.0

    for name, host_fn in [("TmemAllocator", host_wrapper), ("cute.arch", host_manual)]:
        C = C_orig.clone()
        C_ = from_dlpack(C, assumed_align=16)

        compiled = cute.compile(host_fn, C_)
        compiled(C_)

        ok = torch.allclose(C, ref, atol=1e-5, rtol=1e-5)
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        assert ok, f"{name} FAILED"


if __name__ == "__main__":
    main()
