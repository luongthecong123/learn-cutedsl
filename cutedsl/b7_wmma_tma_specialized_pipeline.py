import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
from cutlass.pipeline import PipelineTmaAsync, CooperativeGroup, Agent, make_pipeline_state, PipelineUserType
from typing import Tuple

import torch


class Gemm_TC:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 64),
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.BK = self.tile_shape_mnk
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (4, 2, 1)
        self.warp_size = cute.arch.WARP_SIZE

        # 16 MMA warps (warps 0-15) + 1 TMA warp (warp 16)
        self.num_mma_warps = self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self.num_tma_warps = 1
        self.threads_per_cta = self.warp_size * (self.num_mma_warps + self.num_tma_warps)

        assert self.BM % 16 == 0, "bM must be divisible by 16"
        assert self.BN % 16 == 0, "bN must be divisible by 16"

        cc = torch.cuda.get_device_capability()
        if cc >= (9, 0) and cc < (12, 0):
            self.num_stages = 4  # sm90, sm100
        else:
            self.num_stages = 2  # sm120+
        self.buffer_align_bytes = 1024

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
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Swizzled smem layouts with staging dimension for pipeline
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype,
            num_stages=self.num_stages
        )

        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype,
            num_stages=self.num_stages
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.BM, self.BK),
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.BN, self.BK),
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # MMA Layout
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)

        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )

        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)

        print("tiled_mma: ", tiled_mma)

        grid_dim = *cute.ceil_div(c.shape, (self.BM, self.BN)), 1
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tiled_mma,
            c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
        ).launch(
            grid=grid_dim,
            block=(self.threads_per_cta, 1, 1)
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):
        # ====== Thread, Block setup =======
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # Threads 0-511: MMA warps (warps 0-15)
        # Threads 512-543: TMA warp (warp 16)
        num_mma_threads = self.num_mma_warps * self.warp_size
        is_mma_warp = tidx < num_mma_threads
        is_tma_warp = warp_idx == self.num_mma_warps

        if is_tma_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ===== Smem allocation and copy setup ======
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        gA = cute.local_tile(
            input=mA_mk,
            tiler=self.tile_shape_mnk,
            coord=(bidx, bidy, None),
            proj=(1, None, 1))

        gB = cute.local_tile(
            input=mB_nk,
            tiler=self.tile_shape_mnk,
            coord=(bidx, bidy, None),
            proj=(None, 1, 1))

        gC = cute.local_tile(
            input=mC,
            tiler=self.tile_shape_mnk,
            coord=(bidx, bidy, None),
            proj=(1, 1, None))

        # ===== TMA partition setup =====
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA, 0, 2)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB, 0, 2)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            sB_for_tma_partition,
            gB_for_tma_partition,
        )

        # ===== MMA thread partitioning =====
        thr_mma = tiled_mma.get_slice(tidx)

        tCgC = thr_mma.partition_C(gC)

        # Use stage 0 as template for register fragment allocation
        sA_0 = cute.slice_(sA, (None, None, 0))
        sB_0 = cute.slice_(sB, (None, None, 0))

        tCsA_0 = thr_mma.partition_A(sA_0)
        tCsB_0 = thr_mma.partition_B(sB_0)

        tCrA = tiled_mma.make_fragment_A(tCsA_0)
        tCrB = tiled_mma.make_fragment_B(tCsB_0)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        # ====== Shared memory to register copy (LdMatrix) ======
        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.a_dtype,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.b_dtype,
        )

        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)

        # Register copy views (same shape for all stages)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

        # ===== Pipeline setup =====
        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline = PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, self.num_tma_warps),
            consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_warps),
            barrier_storage=mbar_ptr,
            tx_count=tma_transaction_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1))
        )

        producer_state = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        consumer_state = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)

        num_k_tiles = mA_mk.shape[1] // self.BK

        # ====== TMA warp (producer) ======
        if is_tma_warp:
            # Prefetch: fill all pipeline stages
            for kidx in range(self.num_stages):
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

            # Continue producing for remaining K tiles
            for kidx in range(self.num_stages, num_k_tiles):
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

        # ====== MMA warps (consumer) ======
        if is_mma_warp:
            tCrC.fill(0.0)

            for kidx in range(num_k_tiles):
                mainloop_pipeline.consumer_wait(consumer_state)

                # Get smem view for current pipeline stage
                sA_stage = cute.slice_(sA, (None, None, consumer_state.index))
                sB_stage = cute.slice_(sB, (None, None, consumer_state.index))

                # Partition smem source for LdMatrix copy
                tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA_stage)
                tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB_stage)

                # Load smem -> registers via LdMatrix
                cute.copy(
                    atom=tiled_copy_s2r_A,
                    src=tCsA_copy_view,
                    dst=tCrA_copy_view
                )

                cute.copy(
                    atom=tiled_copy_s2r_B,
                    src=tCsB_copy_view,
                    dst=tCrB_copy_view
                )

                # GEMM on register fragments
                cute.gemm(
                    atom=tiled_mma,
                    d=tCrC,
                    a=tCrA,
                    b=tCrB,
                    c=tCrC
                )

                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # ====== Store results: register -> gmem ======
            atom_universal = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mC.element_type
            )

            tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)

            for reg_idx in range(cute.size(tCrC_out)):
                tCrC_out[reg_idx] = cutlass.Float16(tCrC[reg_idx])

            cute.copy(
                atom=atom_universal,
                src=tCrC_out,
                dst=tCgC
            )

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
        )
        return tma_atom, tma_tensor


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

    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")


if __name__ == "__main__":
    main()
