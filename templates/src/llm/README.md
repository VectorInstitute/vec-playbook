# LLM Training Templates

This directory includes templates for language-model workloads:

- [text_classification](text_classification/): fine-tunes a small LLM on AG News via Hugging Face Trainer.
- [finetune_distributed](finetune_distributed/): distributed finetuning example adapted from VectorLM (https://github.com/VectorInstitute/vectorlm).

## Finetune Distributed (DDP/FSDP)

Run the distributed template:
```bash
uv run python -m llm.finetune_distributed.launch \
  compute=bon_echo/a40_4x \
  +trainer.dist.mode=fsdp --multirun
```
You can choose **DDP** or **FSDP** mode by setting the `+trainer.dist.mode` argument (`ddp` or `fsdp`).

A few points to clarify for this template:
- **`launch.py`** is the Hydra entrypoint; it merges config layers and hands the resolved config to Submitit.
- **`distributed_launcher.py`** is a Submitit helper; it shells out to `torch.distributed.run` so that torchrun controls per-rank workers without re-entering Hydra (the same pattern used in VectorLM).
- **`train.py`** is the torchrun worker; it loads the saved config, builds tokenizer, dataloaders, model, and optimizer, and then delegates to the Trainer.
- **`trainer_core.py`** is a minimal trainer (adapted from VectorLM’s `trainer.py`); it handles gradient accumulation, checkpointing, optional evaluation, and works with either DDP or FSDP.

Hydra and Submitit resolve and submit jobs once. Torchrun (DDP/FSDP) needs to own process creation per GPU. Launching `torch.distributed.run` in a subprocess is the standard Hydra + Submitit approach: it avoids nested Hydra invocations, keeps the Hydra run directory stable for requeues and checkpoints, and makes local debugging under `torchrun` straightforward.
