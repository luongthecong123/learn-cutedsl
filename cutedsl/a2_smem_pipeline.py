import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.pipeline import PipelineAsync, CooperativeGroup, Agent
import cutlass.utils.hopper_helpers as sm90_utils

import torch

class GemmPipeAsync:
    def __init__(self):
        self.tile_shape_mnk = (8, 4, 16)
        self.BM, self.BN, self.BK = self.tile_shape_mnk
        self.padding = 8
        self.block_size = self.BM * self.BN
        
        # Pipeline params
        self.num_stages = 4
        self.buffer_align_bytes = 1024
        self.shared_storage = None
        self.num_producer_threads = 32
        self.num_consumer_threads = 32
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor
    ):
        sA_layout_staged = cute.make_layout(
            shape=(self.num_stages, self.BM, self.BK),
            stride=(self.BM * (self.BK + self.padding), self.BK + self.padding, 1)
        )

        sB_layout_staged = cute.make_layout(
            shape=(self.num_stages, self.BN, self.BK),
            stride=(self.BN * (self.BK + self.padding), self.BK + self.padding, 1)
        )

        @cute.struct
        class SharedStorage:
            pipeline_mbarrier_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout_staged)],
                self.buffer_align_bytes,
            ] 

        self.shared_storage = SharedStorage

        M, N = mC.shape
        
        self.kernel(
            mA, mB, mC,
            sA_layout_staged, sB_layout_staged
            ).launch(
            grid=[N//self.BN, M//self.BM, 1],
            block=[self.num_producer_threads + self.num_consumer_threads, 1, 1]
        )
    
    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
        sA_layout_staged: cute.Layout, sB_layout_staged: cute.Layout
    ):
        BM, BN, BK = self.tile_shape_mnk

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sA = storage.sA.get_tensor(layout=sA_layout_staged)
        sB = storage.sB.get_tensor(layout=sB_layout_staged)

        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        mainloop_pipeline = PipelineAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, self.num_producer_threads),
            consumer_group=CooperativeGroup(Agent.Thread, self.num_consumer_threads),
            barrier_storage=storage.pipeline_mbarrier_ptr.data_ptr()
        )

        producer, consumer = mainloop_pipeline.make_participants()

        # Producer warp
        if warp_idx == 0:
            tid, _, _ = cute.arch.thread_idx()
            
            for ctak in range(0, gA.shape[1], BK):
                handle = producer.acquire_and_advance()

                num_loads_A = BM * BK
                for i in range(tid, num_loads_A, self.num_producer_threads):
                    row = i // BK
                    col = i % BK
                    sA[handle.index, row, col] = gA[bidy * BM + row, ctak + col]

                num_loads_B = BN * BK
                for i in range(tid, num_loads_B, self.num_producer_threads):
                    row = i // BK
                    col = i % BK
                    sB[handle.index, row, col] = gB[bidx * BN + row, ctak + col]
                
                handle.commit()
            
            producer.tail()

        # Consumer warp    
        if warp_idx == 1:
            tid, _, _ = cute.arch.thread_idx()
            tid = tid - self.num_producer_threads
            tidx = tid % BM
            tidy = tid // BM

            acc = cute.Float32(0)

            for ctak in range(0, gA.shape[1], BK):
                handle = consumer.wait_and_advance()

                for mmak in range(BK):
                    acc += cute.Float32(sA[handle.index, tidx, mmak]) * cute.Float32(sB[handle.index, tidy, mmak])
                    
                handle.release()

            gC[bidy * BM + tidx, bidx * BN + tidy] = cute.Float16(acc)          

def main():
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    gemm = GemmPipeAsync()
    compiled = cute.compile(gemm, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.matmul(A, B.T), atol=1e-1, rtol=1e-1), f"CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()