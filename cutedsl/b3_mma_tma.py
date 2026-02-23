import torch
import ray
from typing import Tuple


import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils

@ray.remote(num_gpus=1)
class Gemm_TC:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (64, 64, 64),
    ):
        self.tile_shape_mnk = cta_tiler
        self._bM, self._bN, self._bK = self.tile_shape_mnk
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (4, 4, 1)
        self.warp_size = cute.arch.WARP_SIZE
        self.threads_per_cta = self.warp_size * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        assert self._bM % 16 == 0, "bM must be divisible by 16"
        assert self._bN % 16 == 0, "bN must be divisible by 16"

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
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
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

        #===============================================
        # MMA Layout
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0], # 4 * 16 = 64
            # if atom layout's N-mode is 1, to leverage the largest coalesced
            # shared memory -> register copy, set the tiled mma's N mode to 16
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2, # 4 * 8 * 2 = 64
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )     
        
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)

        print("tiled_mma: ", tiled_mma)
        # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, 1)
        grid_dim = *cute.ceil_div(c.shape, (self._bM, self._bN)), 1
        self.kernel(
            tma_atom_a, 
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,  # output tensor
            tiled_mma,
            cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged
            ).launch(
            grid=grid_dim, 
            block=(self.threads_per_cta,1,1)
        )

        cutlass.pipeline.arrive_and_wait
    
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        mC_mn: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
    ):

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptor
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

        # Shared memory
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        print("a smem layout staged: ", a_smem_layout_staged)
        print("a smem layout: ", a_smem_layout)

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
            input=mC_mn, 
            tiler=self.tile_shape_mnk,
            coord=(bidx, bidy, None),
            proj=(1, 1, None))
        

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        print("tAgA: ", tAgA)
        print("tAsA: ", tAsA)

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        
        # ====================== Registers allocation for  MmaF16BF16Op
        # Slice sA/sB to remove stage dimension for MMA partitioning (stage 0)
        sA_mma = cute.slice_(sA, (None, None, 0))
        sB_mma = cute.slice_(sB, (None, None, 0))
        
        tCsA = thr_mma.partition_A(sA_mma)
        tCsB = thr_mma.partition_B(sB_mma)
        
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        tCrC.fill(0.0)
        
        # Creates the tiled copy so that it matches the thread-value layout
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
        

        # Calculate TMA transaction bytes for synchronization
        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # Get pointer to mbarrier in shared memory
        mbar_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        for kidx in range(mA_mk.shape[1] // self._bK):
            # Reinitialize mbarrier each iteration (simpler for single-stage)
            if tid == 0:
                cute.arch.mbarrier_init(mbar_ptr, cnt=1)
                cute.arch.mbarrier_init_fence()
            
            cute.arch.sync_threads()
            
            # Set expected transaction bytes and arrive (single thread)
            if warp_idx == 0:
                if tid == 0:
                    cute.arch.mbarrier_expect_tx(mbar_ptr, tma_transaction_bytes)
                    cute.arch.mbarrier_arrive(mbar_ptr)

                # Issue TMA copies - whole warp calls, internal elect_one handles single-thread
                cute.copy(
                    tma_atom_a,
                    tAgA[None, kidx],
                    tAsA[None, 0],  # 0 for single stage
                    tma_bar_ptr=mbar_ptr
                )
                
                cute.copy(
                    tma_atom_b,
                    tBgB[None, kidx],
                    tBsB[None, 0],  # 0 for single stage
                    tma_bar_ptr=mbar_ptr
                )

            # Wait for TMA to complete
            cute.arch.mbarrier_wait(mbar_ptr, 0)

            # Load from SMEM to registers
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
            
            # GEMM
            cute.gemm(
                atom=tiled_mma,
                d=tCrC,
                a=tCrA,
                b=tCrB,
                c=tCrC
            )
            
            cute.arch.sync_threads()
        
        atom_universal = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC_mn.element_type 
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
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
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

    def run_gemm(self, M: int = 64, N: int = 64, K: int = 64, 
                 iterations: int = 10, abs_tol: float = 1e-5, rel_tol: float = 1e-5):
        import time
        
        device = torch.device("cuda")
        print(f"Running GEMM on device: {device}")
        print(f"M: {M}, N: {N}, K: {K}")
        
        torch.manual_seed(42)
        
        # Create test matrices
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(N, K, device=device, dtype=torch.float16)
        c = torch.zeros(M, N, device=device, dtype=torch.float16)
        
        a_ = from_dlpack(a, assumed_align=16)
        b_ = from_dlpack(b, assumed_align=16)
        c_ = from_dlpack(c, assumed_align=16)

        output = torch.zeros_like(c)
        output_ = from_dlpack(output, assumed_align=16)       

        # Ref
        c_test = a @ b.T
        
        # Impl
        compiled_code = cute.compile(self, a_, b_, output_)

        # Warmup
        compiled_code(a_, b_, output_)
        torch.cuda.synchronize()

        abs_diff = torch.abs(c_test - output)
        rel_diff = abs_diff / (torch.abs(c_test) + 1e-8)
        
        if abs_diff.mean().item() > abs_tol or rel_diff.mean().item() > rel_tol:
            return f"ERROR: Tolerance exceeded - abs_diff: {abs_diff.mean().item():.6f} (tol: {abs_tol}), rel_diff: {rel_diff.mean().item():.6f} (tol: {rel_tol})"

        start_time = time.time()
        for i in range(iterations):
            output = torch.zeros_like(c)
            output_ = from_dlpack(output, assumed_align=16)
            compiled_code(a_, b_, output_)
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / iterations) * 1000
        
        ops_per_gemm = 2 * M * N * K
        tflops = ops_per_gemm / (avg_time_ms / 1000) / 1e12
        
        result_str = (
            f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n"
            f"Matrix Shape (M, N, K): {(M, N, K)}\n"
            f"Iterations: {iterations}\n"
            f"Avg Time per GEMM: {avg_time_ms:.4f}ms\n"
            f"Performance: {tflops:.2f} TFLOPS\n"
            f"Mean Absolute Diff: {abs_diff.mean().item():.6f}\n"
            f"Mean Relative Diff: {rel_diff.mean().item():.6f}"
        )
        
        return result_str


def main():
    if not ray.is_initialized():
        ray.init()
    num_gpus = int(ray.available_resources().get("GPU", 0))
    num_workers = 1
    print(f"Detected {num_gpus} GPUs. Creating {num_workers} worker.")
    workers = [Gemm_TC.remote(cta_tiler=(64, 64, 64)) for i in range(num_workers)]
    M, N, K = 8192, 8192, 8192
    iterations = 10
    futures = [worker.run_gemm.remote(M=M, N=N, K=K, iterations=iterations) for worker in workers]
    results = ray.get(futures)
    for idx, result in enumerate(results):
        print(f"\nWorker {idx}:")
        print(result)
    
    ray.shutdown()


if __name__ == "__main__":
    main()
    