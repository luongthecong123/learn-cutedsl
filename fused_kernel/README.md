Solution explanation for Deepseed Sparse Attention and Indxer

1. Scaled dot product attention

2. Deepseek Sparse Attention

Moving up a notch, we will try to use B200 bf16 tensor core and the new FFMA2 instruction, which 2x faster than FFMA. Analyzing the SASS of a CUBLASS SGEMM kernel, FFMA2 was used in SGEMM (single precision GEMM) by CUBLAS, it achieved 95% peak FP32 FLOPS (shows evidence). We will try to do warp specialization to overlap tcgen05.mma and FFMA2 computation in 2 ways. Due to the competition's precision requirement, we can't use tensor core for both the score computation and the output computation.

3. Deepseek Sparse Attention Indexer