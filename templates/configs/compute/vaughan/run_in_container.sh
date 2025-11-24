#!/bin/bash

# Script for running vLLM on Bon Echo
# Example:
# bash run_in_container.sh uv run vllm serve /model-weights/Qwen3-8B  
source ~/.bashrc
source /opt/lmod/lmod/init/bash
export MODULEPATH=/opt/modulefiles:/pkgs/modulefiles:/pkgs/environment-modules

module load singularity-ce
export SINGULARITYENV_SLURM_CONF=/opt/slurm/etc/slurm.conf
export SINGULARITYENV_PATH="/opt/slurm/bin:$PATH"
export SINGULARITYENV_LD_LIBRARY_PATH="/opt/slurm/lib:/opt/slurm/lib64:/opt/munge/lib:/opt/munge/lib64:${LD_LIBRARY_PATH:-}"

unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT
if [[ -n "$UV_PROJECT_ENVIRONMENT" ]]; then
    source ${UV_PROJECT_ENVIRONMENT}/bin/activate
fi

singularity exec \
--nv \
--bind /model-weights:/model-weights \
--bind /projects/llm:/projects/llm \
--bind $HOME:$HOME \
--bind $SCRATCH:$SCRATCH \
/projects/llm/unsloth-vllm-trl-latest.sif \
$@