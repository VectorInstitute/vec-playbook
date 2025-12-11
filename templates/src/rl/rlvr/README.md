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

## Vaughan (Bon Echo) User? Extra setup steps required

(Skip this section if you are running on Killarney)

This reference implementation depends on `vllm>=0.11.0`. vLLM does not provide pre-built wheels for older Linux distros, such as the one running on Vaughan (Bon Echo). Follow the steps in [vaughan_setup.md](vaughan_setup.md) before continuing.

## Run Trainer

```bash
uv run python \
-m rl.rlvr.grpo.launch \
--multirun \
compute=killarney/l40s_1x \
requeue=off \
num_epochs=10 \
data.train_split="train\[:100\]" \
run_name="grpo_gsm8k_dry_run"
```

## Adapting to your workflow

Configurable options:

- GRPO hyperparameters
- dataset (the example uses `openai/gsm8k`)
- evaluation scheme and LLM judge setup

## Optional- Observability Integration via LangFuse

Set up LangFuse to track the output of your models as training proceeds.

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_HOST="https://us.cloud.langfuse.com"
```
