# VLM Training Templates

These folders show how to fine-tune a vision-language model with Hydra + Submitit.

## Templates
- `image_captioning/`: fine-tunes `Salesforce/blip-image-captioning-base`. Defaults to `dataset_name=cifar10`, where captions are derived from CIFAR-10 labels.

## Quick Start

```bash
# Submit a GPU run on Bon Echo A40
uv run python -m vlm.image_captioning.launch \
  compute=bon_echo/a40_1x \
  +trainer.batch_size=16 \
  --multirun
```
