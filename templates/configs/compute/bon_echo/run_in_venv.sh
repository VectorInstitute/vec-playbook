#!/bin/bash

# Script for running vLLM on Bon Echo
# Example:
# bash run_in_container.sh uv run vllm serve /model-weights/Qwen3-8B
source ~/.bashrc

unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
if [[ -n "$UV_PROJECT_ENVIRONMENT" ]]; then
    source ${UV_PROJECT_ENVIRONMENT}/bin/activate
fi

nvidia-smi

$@
