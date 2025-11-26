# RL with Verifiable Reward (RLVR) Reference Implementations

This folder contains scripts for running RLVR algorithms on LLMs on the Vector cluster.

Supported algorithms:

- GRPO

Features:

- Compatibility with Chat Completion models.
- LLM-as-a-judge for more involved reward verifications.
- Optimized for heterogenous compute environments- run backpropagation on H100/A100, and use L40S/A40/RTX8000 for rollout and LLM judge via dedicated SLURM jobs (see [submitit_vllm.py](submitit_vllm.py) for details.)

Current limitations and TODO items:

- Single-GPU finetuning only.
- Backprop GPU does not participate in rollouts.
- Rollout GPUs might sit idle when all rollouts are done and only eval is pending.
- Verify support for function calling via Chat Completion.
- Integrate LangFuse datasets to track eval traces across steps.

## Setup

Basic: you will need uv and a working PyTorch installation. Running from within a container is possible, but you need to make sure SLURM commands are available from within the container.

### Option A- running vLLM in uv venv

Make sure vLLM runs on your environment.

- Create vLLM uv venv following [instructions from vllm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html#set-up-using-python)
- Make a copy of [run_in_venv.sh](/run_in_venv.sh) and point the uv venv to your newly-created vllm venv.
- Remember to `chmod a+x <your_script.sh>`
- Make sure that in a new GPU job, `<your_script.sh> uv run vllm serve <EXAMPLE_MODEL_NAME>` launches the vLLM server.

Example: clusters running modern Linux distros for which pre-built vLLM wheels are available

- Vector Institute Killarney
- Mila TamIA

### Option B- running vLLM via Singularity

It might be difficult to install vLLM on some clusters- e.g., unusual Linux distribution. As long as vLLM runs through Singularity on these environments, these reference implementations would work there as well. Steps:

- Make sure you can manually spin up singularity and run `vllm serve` from within the GPU container.
- Make a copy of [run_in_container.sh](/run_in_container.sh) and point to your singularity image. Remember, all scripts will be added to `$@` and sent to singularity.
- Remember to `chmod a+x <your_script.sh>`
- Make sure that in a new GPU job, `<your_script.sh> uv run vllm serve <EXAMPLE_MODEL_NAME>` launches the vLLM server.

Examples:

- Vector Institute Bon Echo
- Compute Canada Narval

## Run Trainer

```bash
uv run python \
-m rlvr.grpo.launch \
--multirun \
compute=vaughan/a40_1x \
requeue=off \
+trainer.num_epochs=10 \
+trainer.data.train_split="train\[:100\]"
```

## Adapting to your workflow

Configurable options:

- GRPO hyperparameters
- dataset (the example uses `openai/gsm8k`)
- evaluation scheme and LLM judge setup

## Optional- Observability Integration

Set up LangFuse to track the output of your models as training proceeds.

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_HOST="https://us.cloud.langfuse.com"
```
