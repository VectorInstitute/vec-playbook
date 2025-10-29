#!/bin/bash
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
source $SCRATCH/uv-venvs/vllm-serving/bin/activate
export UV_PROJECT_ENVIRONMENT=/scratch/ssd004/scratch/jacobtian/uv-venvs/vllm-serving
$@