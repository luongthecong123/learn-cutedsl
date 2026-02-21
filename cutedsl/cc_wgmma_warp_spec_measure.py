# trace_gemm_tc_globaltimer.py

import json
import torch
import math
from typing import Tuple
import sys
sys.path.insert(0, 'cutedsl')

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm


TAGS = [
    "Setup",        # 0
    "IssueTMA",     # 1
    "IssueMMA",     # 2
    "WaitTMA",      # 3
    "WaitMMA",      # 4
    "WaitMainloop", # 5
    "WaitEpilogue", # 6
    "Epilogue",     # 7
]

Setup        = 0
IssueTMA     = 1
IssueMMA     = 2
WaitTMA      = 3
WaitMMA      = 4
WaitMainloop = 5
WaitEpilogue = 6
Epilogue     = 7

PROFILER_HEADER = 1
PROFILER_ENTRY  = 4


@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(
        T.i64(), [],
        "mov.u64 $0, %globaltimer;", "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(
        T.i32(), [],
        "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


def profiler_start(profiler, slot, sm, tag, cnt):
    profiler[slot, PROFILER_HEADER + cnt * PROFILER_ENTRY + 0] = cutlass.Int64(sm)
    profiler[slot, PROFILER_HEADER + cnt * PROFILER_ENTRY + 1] = cutlass.Int64(tag)
    profiler[slot, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2] = globaltimer_u64()


def profiler_stop(profiler, slot, cnt):
    profiler[slot, PROFILER_HEADER + cnt * PROFILER_ENTRY + 3] = (
        globaltimer_u64() - profiler[slot, PROFILER_HEADER + cnt * PROFILER_ENTRY + 2]
    )
    return cnt + cutlass.Int32(1)


class Gemm_TC_Profiled:
    def __init__(self, cta_tiler: Tuple[int, int, int] = (128, 128, 64)):
        self.tile_shape_mnk = cta_tiler
        self._bM, self._bN, self._bK = self.tile_shape_mnk

        self.atom_layout_mnk = (
            (2, 1, 1) if self._bM > 64 and self._bN > 64 else (1, 1, 1)
        )

        self.warp_size = cute.arch.WARP_SIZE
        self.warp_group_size = 128

        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_cta = self.mma_warp_groups * self.warp_group_size + self.warp_size

        assert self._bM % 64 == 0
        assert self._bN % 64 == 0

        self.num_stages = 1
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, profiler: cute.Tensor):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype, num_stages=self.num_stages,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype, num_stages=self.num_stages,
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged, (self.tile_shape_mnk[0], self.tile_shape_mnk[2]), 1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b, self.b_smem_layout_staged, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), 1,
        )

        c_smem_layout = cute.make_layout(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            stride=(self.tile_shape_mnk[1], 1),
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c, c_smem_layout,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, self.tile_shape_mnk[0] * self.tile_shape_mnk[1]],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype, self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )

        grid_dim = *cute.ceil_div(c.shape, (self._bM, self._bN)), 1
        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            self.tiled_mma, cta_layout_mnk, c,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            profiler,
        ).launch(grid=grid_dim, block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_tma_tensor: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_tma_tensor: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_tma_tensor: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        profiler: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        is_mma_warp = tidx < self.mma_warp_groups * self.warp_group_size
        is_tma_warp = warp_idx // 4 == self.mma_warp_groups

        grid_x, grid_y, _ = cute.arch.grid_dim()
        bid    = bidy * grid_x + bidx
        # TMA warp slot: bid * 2 + 0
        # MMA warp slot: bid * 2 + 1
        tma_slot = bid * 2 + 0
        mma_slot = bid * 2 + 1

        sm = smid_u32()
        tma_cnt = cutlass.Int32(0)
        mma_cnt = cutlass.Int32(0)

        # --- Setup ---
        if is_tma_warp:
            if tidx == self.mma_warp_groups * self.warp_group_size:
                profiler_start(profiler, tma_slot, sm, Setup, tma_cnt)

        if is_tma_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        c_smem_layout = cute.make_layout((self._bM, self._bN), stride=(self._bN, 1))
        sC = storage.sC.get_tensor(c_smem_layout)

        tile_coord_mnk = (bidx, bidy, None)

        gA = cute.local_tile(mA_tma_tensor, self.tile_shape_mnk, tile_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, self.tile_shape_mnk, tile_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.tile_shape_mnk, tile_coord_mnk, proj=(1, 1, None))

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, cluster_coord_mnk[1], a_cta_layout,
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, cluster_coord_mnk[0], b_cta_layout,
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2),
        )

        tma_transaction_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        )

        mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        k_tile_cnt = mA_tma_tensor.shape[1] // self._bK
        num_k_blocks = self._bK // 16

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.warp_group_size)
        warp_group_thread_layout = cute.make_layout(self.mma_warp_groups, stride=self.warp_group_size)
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        acc_shape = tCgC.shape
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        if is_mma_warp:
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        if is_tma_warp:
            if tidx == self.mma_warp_groups * self.warp_group_size:
                tma_cnt = profiler_stop(profiler, tma_slot, tma_cnt)  # Setup end

        for kidx in range(k_tile_cnt):

            # --- IssueTMA ---
            if is_tma_warp:
                if tidx == self.mma_warp_groups * self.warp_group_size:
                    profiler_start(profiler, tma_slot, sm, IssueTMA, tma_cnt)

            if tidx == self.mma_warp_groups * self.warp_group_size:
                cute.arch.mbarrier_init(mbar_ptr, cnt=1)
                cute.arch.mbarrier_init_fence()

            cute.arch.sync_threads()

            if is_tma_warp:
                if tidx == self.mma_warp_groups * self.warp_group_size:
                    cute.arch.mbarrier_expect_tx(mbar_ptr, tma_transaction_bytes)
                    cute.arch.mbarrier_arrive(mbar_ptr)

                cute.copy(tma_atom_a, tAgA[None, kidx], tAsA[None, 0], tma_bar_ptr=mbar_ptr)
                cute.copy(tma_atom_b, tBgB[None, kidx], tBsB[None, 0], tma_bar_ptr=mbar_ptr)

                if tidx == self.mma_warp_groups * self.warp_group_size:
                    tma_cnt = profiler_stop(profiler, tma_slot, tma_cnt)  # IssueTMA end

            # --- WaitTMA ---
            if is_mma_warp:
                if tidx == 0:
                    profiler_start(profiler, mma_slot, sm, WaitTMA, mma_cnt)

            cute.arch.mbarrier_wait(mbar_ptr, 0)

            if is_mma_warp:
                if tidx == 0:
                    mma_cnt = profiler_stop(profiler, mma_slot, mma_cnt)  # WaitTMA end

            # --- IssueMMA ---
            if is_mma_warp:
                if tidx == 0:
                    profiler_start(profiler, mma_slot, sm, IssueMMA, mma_cnt)

                cute.nvgpu.warpgroup.fence()

                for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    k_block_coord = (None, None, k_block_idx, 0)
                    cute.gemm(tiled_mma, accumulators, tCrA[k_block_coord], tCrB[k_block_coord], accumulators)
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                if tidx == 0:
                    mma_cnt = profiler_stop(profiler, mma_slot, mma_cnt)  # IssueMMA end

        # --- Epilogue ---
        if is_mma_warp:
            if tidx == 0:
                profiler_start(profiler, mma_slot, sm, Epilogue, mma_cnt)

            tv_layout_C_tiled = tiled_mma.tv_layout_C_tiled
            for reg_idx in range(cute.size(accumulators)):
                coord = cute.idx2crd((tidx, reg_idx), tv_layout_C_tiled.shape)
                mn_local_tile_flat = cute.crd2idx(coord, tv_layout_C_tiled)
                m_local, n_local = cute.idx2crd(mn_local_tile_flat, c_smem_layout.shape)
                sC[m_local, n_local] = cutlass.Float16(accumulators[reg_idx])

            if tidx == 0:
                mma_cnt = profiler_stop(profiler, mma_slot, mma_cnt)  # Epilogue end

        cute.arch.sync_threads()
        cute.arch.fence_proxy("async.shared", space="cta")

        gC_tma = cute.local_tile(mC_tma_tensor, self.tile_shape_mnk, tile_coord_mnk, proj=(1, 1, None))
        tCsC, tCgC_store = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2), cute.group_modes(gC_tma, 0, 2),
        )

        if is_tma_warp:
            if tidx == self.mma_warp_groups * self.warp_group_size:
                profiler_start(profiler, tma_slot, sm, Epilogue, tma_cnt)

            cute.copy(tma_atom_c, tCsC, tCgC_store)

            if tidx == self.mma_warp_groups * self.warp_group_size:
                tma_cnt = profiler_stop(profiler, tma_slot, tma_cnt)  # Epilogue end
                profiler[tma_slot, 0] = cutlass.Int64(tma_cnt)

        if is_mma_warp:
            if tidx == 0:
                profiler[mma_slot, 0] = cutlass.Int64(mma_cnt)

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile, mcast_dim):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor


def write_chrome_trace_json(profiler_cpu, out_path: str, block_id: int = 0):
    if isinstance(profiler_cpu, torch.Tensor):
        profiler_cpu = profiler_cpu.contiguous().cpu().tolist()

    events = []
    # tma_slot = block_id * 2 + 0, mma_slot = block_id * 2 + 1
    for warp_label, slot, tid in [("TMA", block_id * 2, 0), ("MMA", block_id * 2 + 1, 1)]:
        data = profiler_cpu[slot]
        cnt  = data[0]
        base = None
        for i in range(cnt):
            sm_id = data[PROFILER_HEADER + i * PROFILER_ENTRY + 0]
            tag   = data[PROFILER_HEADER + i * PROFILER_ENTRY + 1]
            start = data[PROFILER_HEADER + i * PROFILER_ENTRY + 2]
            dur   = data[PROFILER_HEADER + i * PROFILER_ENTRY + 3]
            if base is None:
                base = start
            events.append(dict(
                name=f"{warp_label}:{TAGS[tag]}", ph="X",
                ts=(start - base) / 1000.0,
                dur=dur / 1000.0,
                pid=sm_id, tid=tid,
            ))

    # re-base across both warps
    if events:
        base = min(e["ts"] for e in events)
        for e in events:
            e["ts"] -= base

    with open(out_path, "w") as f:
        json.dump({"traceEvents": events}, f)


def main():
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    bM, bN = 128, 128
    grid_x = N // bN
    grid_y = M // bM
    num_blocks  = grid_x * grid_y
    # slots: 2 per block (TMA + MMA), each with enough entries
    # TMA: 1 Setup + k_tiles * IssueTMA + 1 Epilogue
    # MMA: k_tiles * (WaitTMA + IssueMMA) + 1 Epilogue
    k_tiles     = K // 64
    num_entries = k_tiles * 2 + 4
    num_slots   = num_blocks * 2

    profiler = torch.zeros(
        (num_slots, PROFILER_HEADER + num_entries * PROFILER_ENTRY),
        device="cuda", dtype=torch.int64,
    )

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)
    profiler_ = from_dlpack(profiler, assumed_align=16)

    gemm = Gemm_TC_Profiled(cta_tiler=(bM, bN, 64))
    compiled = cute.compile(gemm, A_, B_, C_, profiler_)

    # warmup
    compiled(A_, B_, C_, profiler_)
    torch.cuda.synchronize()

    # clean capture
    profiler.zero_()
    compiled(A_, B_, C_, profiler_)
    torch.cuda.synchronize()

    C_ref = torch.matmul(A, B.T)
    assert torch.allclose(C, C_ref, atol=1e-2, rtol=1e-2), \
        f"Correctness check failed\nmax abs diff: {(C - C_ref).abs().max().item():.6f}"
    print("Correctness check PASSED")

    out_path = "gemm_tc_trace.json"
    write_chrome_trace_json(profiler, out_path, block_id=0)
    print(f"Wrote trace JSON: {out_path}")
    print("Open with https://ui.perfetto.dev")


if __name__ == "__main__":
    main()