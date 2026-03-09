import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.testing import benchmark, JitArguments

"""
Blackwell tcgen05 GEMM with TMA load, single-stage, manual mbarriers.
This script is used to understand tcgen05.ld instructions

The simplest Blackwell kernel using tcgen05 UMMA (Unified MMA).
tcgen05 differs from Hopper's WGMMA in key ways:
  - Accumulator lives in TMEM (Tensor Memory) instead of registers
  - MMA is issued by a single thread like TMA
  - Epilogue requires TMEM → RMEM before storing

ALGORITHM (no warp specialization, no pipeline, no TMA/UMMA overlap):
  - TMA -> sync -> UMMA -> sync -> Epilogue (TMEM → RMEM → GMEM)

SMEM budget (per CTA, single stage):
  A: 128 x 256 x 2B = 64 KB
  B: 256 x 256 x 2B = 128 KB
  Total: 192 KB (1 stage)
"""


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


class Gemm_TC:
    def __init__(
        self,
        cta_tile_shape_mnk: Tuple[int, int, int] = (128, 256, 256), # or faster with (128, 512, 128)
    ):
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.BM, self.BN, self.BK = cta_tile_shape_mnk
        self.mma_inst_shape_mnk = (self.BM,  min(self.BN, 256), 16)

        self.threads_per_cta = 128
        self.num_stages = 1

        # How many columns each thread loads from TMEM per tcgen05.ld instruction.
        # Valid: 1,2,4,8,16,32,64,128.  Larger = fewer instructions, more reg pressure.
        self.tmem_ld_rep = 4

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32

        op = tcgen05.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        print("tiled_mma: ", self.tiled_mma)
        
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.cta_tile_shape_mnk, a.element_type, self.num_stages,
        )
        print("a_smem_layout: ", self.a_smem_layout)
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.cta_tile_shape_mnk, b.element_type, self.num_stages,
        )
        print("b_smem_layout: ", self.b_smem_layout)

        a_smem_layout_one_stage = cute.select(self.a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])

        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, a, a_smem_layout_one_stage, self.cta_tile_shape_mnk, self.tiled_mma,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, b, b_smem_layout_one_stage, self.cta_tile_shape_mnk, self.tiled_mma,
        )

        self.c_layout = utils.LayoutEnum.from_tensor(c)

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        grid_dim = *cute.ceil_div(c.shape, (self.BM, self.BN)), 1
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=grid_dim,
            block=(self.threads_per_cta, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_tma_tensor: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_tma_tensor: cute.Tensor,
        mC: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()
        mma_coord_mnk = (bidx, bidy, None)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        # ====== SMEM allocation ======
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ====== Partition tensors for MMA ======
        gA = cute.local_tile(mA_tma_tensor, self.cta_tile_shape_mnk, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, self.cta_tile_shape_mnk, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.cta_tile_shape_mnk, mma_coord_mnk, proj=(1, 1, None))

        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(self.cta_tile_shape_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc)
        print(f"num_tmem_cols: {num_tmem_cols}")
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        # ====== TMA partition ======
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # ====== Barrier setup ======
        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(
            self.b_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
        )

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(tma_mbar, cnt=1)
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        tmem_barrier_id = 1
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=self.threads_per_cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # ====== Epilogue setup: TMEM → RMEM → GMEM ======
        # See findings.md §5 for full explanation of subtiling and TMEM load shapes.

        M_acc = cute.size(tCtAcc, mode=[0, 0])
        N_acc = cute.size(tCtAcc, mode=[0, 1])

        # Select load op based on num_dp (data-path lanes per warp).
        # tmem_warp_shape_mn = (4, 1) for cta_group::1, so num_dp = M / 4.
        #   M=128 → num_dp=32 → Ld32x32bOp  (32 lanes × 32b  = 1 fp32 col/rep)
        #   M=64  → num_dp=16 → Ld16x256bOp (16 lanes × 256b = 8 fp32 cols/rep)
        num_dp = M_acc // 4
        if cutlass.const_expr(num_dp == 32):
            ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
            fp32_cols_per_rep = 1     # 32b / 32b
        elif cutlass.const_expr(num_dp == 16):
            ld_op = tcgen05.Ld16x256bOp(tcgen05.Repetition(self.tmem_ld_rep))
            fp32_cols_per_rep = 8     # 256b / 32b

        # Each rep loads fp32_cols_per_rep columns, so subtile width =
        subtile_n = self.tmem_ld_rep * fp32_cols_per_rep
        epi_tiler = ((M_acc, subtile_n),)

        # zipped_divide reshapes (M,N) into ((subtile), num_subtiles) for iteration.
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
        gC_epi = cute.zipped_divide(tCgC, epi_tiler)

        copy_atom_t2r = cute.make_copy_atom(ld_op, self.acc_dtype)

        # Tile the atom over one subtile, then per-thread slice
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

        # Partition TMEM source and GMEM destination across all subtiles
        tTR_tAcc = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_gC   = tmem_thr_copy.partition_D(gC_epi)

        # Register fragments — sized to hold ONE subtile's worth of data per thread.
        # We index [None, None, 0] to get the shape of a single subtile.
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[None, None, 0].shape, self.acc_dtype)
        tTR_rC   = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

        # CopyUniversalOp for the register → GMEM store (scalar SIMT copy)
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)

        # Derive subtile_cnt from the partition shape:
        #   For zipped_divide, the last mode = number of subtiles = N_acc / subtile_n
        subtile_cnt = cute.size(tTR_tAcc, mode=[2])

        print(f"Epilogue: rep=x{self.tmem_ld_rep}, subtile_cnt:  {subtile_cnt}, regs / thread / subtile:  {cute.size(tTR_rAcc)}")

        # ====== Main loop (only warp 0 does TMA + MMA) ======
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        tma_phase = 0
        mma_phase = 0

        for kidx in range(mA_tma_tensor.shape[1] // self.BK):
            if warp_idx == 0:
                cute.copy(
                    tma_atom_a,
                    tAgA[None, kidx],
                    tAsA[None, 0],
                    tma_bar_ptr=tma_mbar
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[None, kidx],
                    tBsB[None, 0],
                    tma_bar_ptr=tma_mbar
                )
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        tma_mbar,
                        tma_transaction_bytes,
                    )

            cute.arch.mbarrier_wait(tma_mbar, tma_phase)
            tma_phase ^= 1

            tcgen05_fence()

            num_k_blocks = cute.size(tCrA, mode=[2])

            if warp_idx == 0:
                for k_block_idx in range(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, 0)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[k_block_coord],
                        tCrB[k_block_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                if tidx == 0:
                    tcgen05.commit(mma_mbar)

            cute.arch.mbarrier_wait(mma_mbar, mma_phase)
            mma_phase ^= 1

        # ====== Epilogue: TMEM → RMEM → GMEM ======

        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        for subtile_idx in range(subtile_cnt):
            # TMEM → RMEM: issues tcgen05.ld.sync.aligned.32x32b.xN.b32
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, subtile_idx], tTR_rAcc)
            # fp32 → fp16 in registers
            tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
            # RMEM → GMEM
            cute.copy(simt_atom, tTR_rC, tTR_gC[None, None, subtile_idx])

        # Sync all threads before TMEM deallocation
        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


def main():
    M, N, K = 4096, 4096, 4096

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    gemm = Gemm_TC()
    compiled = cute.compile(gemm, A_, B_, C_)
    compiled(A_, B_, C_)

    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()
