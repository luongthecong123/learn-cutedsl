# learn-cutedsl
Adding hardware features and optimization techniques brick by brick and measure the FLOPS speed up, making it easier to add CuTeDSL to your codebase according to your needs.

Apart from educational purpose, this repo can be treated as cutedsl api examples. Where readers can pick out a feature/api and add it to their code or inject them as context to LLMs for coding assistance.

Job submission using Ray
```bash
pip install ray
# Assume ray cluster is already created

ray job submit \
    --address 'http://localhost:8265' \
    --working-dir . \
	--runtime-env-json='{"pip":"./requirements.txt"}'\
	-- python submit_ray.py
```

Job submission using Modal:
```bash
pip install modal
python3 -m modal setup

modal submit_modal.py
```