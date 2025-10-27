#!/bin/bash
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
source /scratch/ssd004/scratch/jacobtian/uv-venvs/vllm-serving/bin/activate
cd ~/vllm-serving
export UV_PROJECT_ENVIRONMENT=/scratch/ssd004/scratch/jacobtian/uv-venvs/vllm-serving
$@