#!/bin/bash
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
source $SCRATCH/uv-venvs/vllm-serving/bin/activate
$@