# Project Context
Adding hardware features and optimization techniques brick by brick and measure the FLOPS speed up, making it easier to add CuTeDSL to readers' codebase according to their needs. Educational purpose. Also stores quality-of-life codes for the author to reuse later.

# Main parts
                                    
  .                                                                                                                                                           
  ├── .claude/                                          
  │   ├── context.md               
  │   ├── rules.md
  │   └── settings.json
  ├── CLAUDE.md
  ├── README.md
  ├── check_correctness.py
  ├── cuda/
  │   ├── cuda_common.cuh
  │   ├── gemm_cuda.py
  │   ├── glue_code.cpp
  │   ├── glue_code.cu
  │   └── kernels/*.cuh
  ├── cutedsl/*.py
  ├── cutedsl_vs_cuda.py
  ├── submit_modal.py
  └── submit_ray.py

  