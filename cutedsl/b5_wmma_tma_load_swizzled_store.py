import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from typing import Tuple

import torch


class Gemm_TC:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (64, 64, 64),
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.BK = self.tile_shape_mnk
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (4, 4, 1)
        self.warp_size = cute.arch.WARP_SIZE
        self.threads_per_cta = self.warp_size * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        assert self.BM % 16 == 0, "bM must be divisible by 16"
        assert self.BN % 16 == 0, "bN must be divisible by 16"

        self.num_stages = 1
        assert self.num_stages == 1, "Only single-stage TMA is supported in this implementation"
        
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)
        
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
        
        cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))        

        # Create swizzled layout: S<3,4,3> o 0 o ((8,8),(64,1),(1,1)):((64,512),(1,0),(0,0))
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype,
            num_stages=self.num_stages
        )
        print("self.a_smem_layout_staged: ", self.a_smem_layout_staged)

        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype,
            num_stages=self.num_stages
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        self.c_smem_layout = cute.make_layout(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            stride=(self.tile_shape_mnk[1], 1)
        )

        self.c_smem_layout_swizzled = cute.make_composed_layout(
            inner=cute.make_swizzle(3, 4, 3),
            offset=0,
            outer=self.c_smem_layout
        )
        
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c,
            self.c_smem_layout_swizzled,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
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
            # sC: cute.struct.Align[
            #     cute.struct.MemRange[
            #         self.c_dtype, self.tile_shape_mnk[0] * self.tile_shape_mnk[1]
            #     ],
            #     self.buffer_align_bytes,
            # ]            

        self.shared_storage = SharedStorage

        #===============================================
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
            tma_atom_c,
            tma_tensor_c,
            tiled_mma,
            cta_layout_mnk,
            c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_swizzled,
            ).launch(
            grid=grid_dim, 
            block=(self.threads_per_cta,1,1)
        )
    
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_tma_tensor: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_swizzled: cute.ComposedLayout,
    ):

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)  

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()    
        thr_mma = tiled_mma.get_slice(tid)

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

        c_smem_layout = cute.make_layout(
            (self.BM, self.BN),
            stride=(self.BN, 1)
        )
        sC = storage.sA.get_tensor(
            c_smem_layout_swizzled.outer, swizzle=c_smem_layout_swizzled.inner)

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
        
        tCgC = thr_mma.partition_C(gC)
        
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        
        sA_mma = cute.slice_(sA, (None, None, 0))
        sB_mma = cute.slice_(sB, (None, None, 0))
        
        tCsA = thr_mma.partition_A(sA_mma)
        tCsB = thr_mma.partition_B(sB_mma)
        
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        tCrC.fill(0.0)
        
        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False,num_matrices=4),
            self.a_dtype,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False,num_matrices=4),
            self.b_dtype,
        )

        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tid)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tid)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA_mma)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB_mma)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
        
        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        for kidx in range(mA_mk.shape[1] // self.BK):
            if tid == 0:
                cute.arch.mbarrier_init(mbar_ptr, cnt=1)
                cute.arch.mbarrier_init_fence()
            
            cute.arch.sync_threads()
            
            if warp_idx == 0:
                if tid == 0:
                    cute.arch.mbarrier_expect_tx(mbar_ptr, tma_transaction_bytes)
                    cute.arch.mbarrier_arrive(mbar_ptr)

                cute.copy(
                    tma_atom_a,
                    tAgA[None, kidx],
                    tAsA[None, 0],
                    tma_bar_ptr=mbar_ptr
                )
                
                cute.copy(
                    tma_atom_b,
                    tBgB[None, kidx],
                    tBsB[None, 0],
                    tma_bar_ptr=mbar_ptr
                )

            cute.arch.mbarrier_wait(mbar_ptr, 0)

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
            
            cute.gemm(
                atom=tiled_mma,
                d=tCrC,
                a=tCrA,
                b=tCrB,
                c=tCrC
            )
            
            cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=cutlass.Float32,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.c_dtype,
        )

        tiled_copy_r2s_C = cute.make_tiled_copy_C(copy_atom_C, tiled_mma)

        thr_copy_stmatrix_C = tiled_copy_r2s_C.get_slice(tid)
        tCsC_copy_view = thr_copy_stmatrix_C.retile(tCrA)
        
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB_mma)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

        # tv_layout_C_tiled = tiled_mma.tv_layout_C_tiled

        # for reg_idx in range(cute.size(tCrC)):
        #     coord = cute.idx2crd((tid, reg_idx), tv_layout_C_tiled.shape)
        #     mn_local_tile_flat = cute.crd2idx(coord, tv_layout_C_tiled)
        #     m_local, n_local = cute.idx2crd(mn_local_tile_flat, c_smem_layout.shape)
        #     sC[m_local, n_local] = cutlass.Float16(tCrC[reg_idx])
        
        cute.arch.sync_threads()
        cute.arch.fence_proxy("async.shared", space="cta")
        
        gC_tma = cute.local_tile(
            input=mC_tma_tensor,
            tiler=self.tile_shape_mnk,
            coord=(bidx, bidy, None),
            proj=(1, 1, None)
        )
        
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_tma, 0, 2)
        
        tCsC, tCgC_store = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        
        if warp_idx == 0:
            cute.copy(tma_atom_c, tCsC, tCgC_store)

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
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

    gemm = Gemm_TC(cta_tiler=(64, 64, 64))
    compiled = cute.compile(gemm, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")


if __name__ == "__main__":
    main()