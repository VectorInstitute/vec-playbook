### LLM training templates

This directory includes templates for LLM training tasks:

- [text_classification](text_classification/): Fine-tunes a small Transformer on AG News using Hugging Face Trainer.
- [finetune_distributed](finetune_distributed/): Demonstrates DDP/FSDP finetuning and defaults to the small `karpathy/tiny_shakespeare` dataset (requires `trust_remote_code=True`) so you can validate multi-GPU plumbing quickly.

# FSDP on 1 node × 4 GPUs
uv run python -m llm.finetune_distributed.launch \
  compute=bon_echo/a40_4x \
  +trainer.dist.mode=fsdp --multirun


This template now includes a lightweight `trainer_core.py` that mirrors the VectorLM trainer structure while remaining minimal enough for quick experiments.
