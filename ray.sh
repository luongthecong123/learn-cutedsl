#!/bin/bash

ray job submit \
    --address 'http://localhost:8265' \
    --working-dir . \
	--runtime-env-json='{"pip":"./requirements.txt"}'\
	-- python submit_ray.py