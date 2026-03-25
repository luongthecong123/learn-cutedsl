import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import math

"""
Scaled dot-product attention (SDPA) in a single kernel, naive implementation with global memory mostly, using vanilla softmax (not numerically stable).

Each thread block will calculate a single output element

A thread block will calculate a tile of (1, BN) of the score matrix. Loop through N dimension in step size of BN, perform exponential and partial reduction with BN elements in V, and in the meantime, accumulate the denominator for softmax.

Finally we will perform a final reduction across the block (intra-warp reduction using warp shuffle -> inter-warp reduction on smem), then divide the accumulated output by the denominator to get the final output.
"""

def spda_explicit_ref(
    Q: torch.Tensor, # (M, dk)
    K: torch.Tensor, # (N, dk)
    V: torch.Tensor, # (N, dv)
) -> torch.Tensor:
    M, dk = Q.shape
    
    score = torch.matmul(Q, K.T) / (dk ** 0.5) # (M, N)
    score_sm = torch.softmax(score, dim=-1) # (M, N)
    output = torch.matmul(score_sm, V) # (M, dv)
    return output

def spda_with_kv_cache(
    Q_new: torch.Tensor,       # (1, dk)   — only the new query token
    K_new: torch.Tensor,       # (1, dk)   — new key to append
    V_new: torch.Tensor,       # (1, dv)   — new value to append
    K_cache: torch.Tensor,     # (T, dk)   — cached keys from previous steps
    V_cache: torch.Tensor,     # (T, dv)   — cached values from previous steps
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Append new K/V to cache: (T, d) -> (T+1, d)
    K = torch.cat([K_cache, K_new], dim=0)  # (T+1, dk)
    V = torch.cat([V_cache, V_new], dim=0)  # (T+1, dv)

    score = torch.matmul(Q_new, K.T) / (K.shape[1] ** 0.5)  # (1, T+1)
    score_sm = torch.softmax(score, dim=-1)                   # (1, T+1)
    output = torch.matmul(score_sm, V)                        # (1, dv)

    return output, K, V  # return updated cache for next step

def spda_pytorch_ref(
    Q: torch.Tensor, # (M, dk)
    K: torch.Tensor, # (N, dk)
    V: torch.Tensor, # (N, dv)
) -> torch.Tensor:
    
    output = torch.nn.functional.scaled_dot_product_attention(
        Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
    ).squeeze(0)
    
    return output

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        # cute.arch.shuffle_sync_bfly will read from another thread's registers
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

class SPDA_kernel:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int, int] = (1, 256, 1, 1)
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.Bdk, self.Bdv = self.tile_shape_mnk
        
    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor, # (M, dk)
        K: cute.Tensor, # (N, dk)
        V: cute.Tensor, # (N, dv)
        output: cute.Tensor # (M, dv)
    ):  
        self.kernel(Q, K, V, output).launch(
            grid=(output.shape[0] // self.BM, output.shape[1] // self.Bdv, 1),   
            block=(self.BN, 1, 1)
        )
        
    @cute.kernel
    def kernel(
        self, 
        Q: cute.Tensor, # (M, dk)
        K: cute.Tensor, # (N, dk)
        V: cute.Tensor, # (N, dv)
        output: cute.Tensor, # (M, dv)
    ):
        # ====== Thread, Block setup =======
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        
        smem = utils.SmemAllocator()
        NUM_WARPS = self.BN // 32
        smem_partials_denom = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NUM_WARPS,), stride=(1,)), 16, None)
        smem_partials_out = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NUM_WARPS,), stride=(1,)), 16, None)
     
        gK_ = cute.zipped_divide(K, (self.BN, 1)) # ((BN, 1), (N//BN, d))
        gV_ = cute.zipped_divide(V, (self.BN, 1)) # ((BN, 1), (N//BN, d))
        
        # ====== Main loop ======
        partial_sm_denominator = cutlass.Float32(0)
        partial_output_acc = cutlass.Float32(0)
        
        for nidx in range(K.shape[0] // self.BN):
            gK = gK_[(None, None), (nidx, None)] # (BN, 1, d)
            gV = gV_[(None, None), (nidx, None)] # (BN, 1, d)
            
            score = cutlass.Float32(0)
            
            for kidx in range(Q.shape[1]):
                score += Q[bidx, kidx] * gK[tidx, 0, kidx]
            
            exp_score = cute.math.exp(score / (Q.shape[1]** 0.5))
            partial_sm_denominator += exp_score
        
            partial_output_acc += exp_score * gV[tidx, 0, bidy]
        
        # intra-warp reduction
        partial_reduced_sm_de = warp_reduce(partial_sm_denominator, lambda a, b: a + b, width=32)
        partial_reduced_out = warp_reduce(partial_output_acc, lambda a, b: a + b, width=32)
        
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        
        if lane_idx == 0:
            smem_partials_denom[warp_idx] = partial_reduced_sm_de
            smem_partials_out[warp_idx] = partial_reduced_out
        
        cute.arch.sync_threads()
        
        # Level 2: inter-warp reduction
        if warp_idx == 0:
            val_denom = cutlass.Float32(0)
            val_out = cutlass.Float32(0)
            if lane_idx < NUM_WARPS:
                val_denom = smem_partials_denom[lane_idx]
                val_out = smem_partials_out[lane_idx]
            
            final_denom = warp_reduce(val_denom, lambda a, b: a + b, width=NUM_WARPS)
            final_out = warp_reduce(val_out, lambda a, b: a + b, width=NUM_WARPS)
            
            if lane_idx == 0:
                output[bidx, bidy] = cutlass.Float32(final_out / final_denom)

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_k = 64
    d_v = 128
    M = 256
    N = 512
    
    Q = torch.randn(M, d_k, device=device)
    K = torch.randn(N, d_k, device=device)
    V = torch.randn(N, d_v, device=device)
    output_impl = torch.empty((M, d_v), device=device)

    # Our reference
    out_ref = spda_explicit_ref(Q, K, V)
    out_pt = spda_pytorch_ref(Q, K, V)

    assert torch.allclose(out_ref, out_pt, atol=1e-4, rtol=1e-4), "TORCH CORRECTNESS FAILED"
    print("TORCH CORRECTNESS PASS")
    
    
    my_kernel = SPDA_kernel()
    Q_ = from_dlpack(Q, assumed_align=16)
    K_ = from_dlpack(K, assumed_align=16)
    V_ = from_dlpack(V, assumed_align=16)
    output_ = from_dlpack(output_impl, assumed_align=16)
    
    compiled = cute.compile(my_kernel, Q_, K_, V_, output_)
    compiled(Q_, K_, V_, output_)
    
    assert torch.allclose(output_impl, out_ref, atol=1e-4, rtol=1e-4), "CUTEDSL CORRECTNESS FAILED"
    print("CUTEDSL CORRECTNESS PASS")
   
    