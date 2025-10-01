# MLP Training Templates

These examples show how to run small MLP jobs with Hydra + Submitit. Each folder can be launched as `mlp.<name>.launch` and includes its own config, launcher, and trainer.

## Templates
- `single_not_checkpointable/`: single-GPU run without resume logic; use it when you just want a quick check.
- `single/`: single-GPU run that saves checkpoints so Submitit can resume after a requeue.
- `ddp/`: multi-GPU version using PyTorch DistributedDataParallel; it expects the compute preset to request multiple GPUs.

All variants build a dummy dataset at runtime (`create_dummy_data`), so no external files are required.

## Quick Start

```bash
# Submit checkpointable single-GPU job
uv run python -m mlp.single.launch compute=bon_echo/a40_1x requeue=on --multirun

# Launch 2×A40 DDP training with a larger hidden layer
uv run python -m mlp.ddp.launch \
  compute=bon_echo/a40_2x \
  +trainer.hidden_dim=256 \
  --multirun
```
