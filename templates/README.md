# Templates

Templates for training ML models on Vaughan and Killarney clusters using Hydra and Submitit.

## Layout

```
templates/
├── src/
│   ├── basic_mlp/        # simple MNIST MLP
│   ├── llm/              # language models
│   ├── vlm/              # vision–language models
│   └── rl/               # reinforcement learning
└── configs/              # Hydra + Submitit configs
```

Each template directory is self-contained: it has a `launch.py`, a `train.py`, and a `config.yaml`.
The `configs/` directory defines Slurm presets and shared Hydra + Submitit settings.

## Setup (uv)

1) Create and activate a virtual environment:
```bash
uv venv .venv
source .venv/bin/activate
```

2) Resolve and install dependencies from `pyproject.toml`:
```bash
uv lock
uv sync
```

3) Fill in your Slurm account defaults (in `templates/configs/user.yaml`):
```yaml
slurm:
  account: AIP-XXXX
  partition: gpu
  qos: normal
```

4) Pick a compute preset:
- `templates/configs/compute/vaughan/*` (A40, A100)
- `templates/configs/compute/killarney/*` (L40S, H100)

## Running templates

Hydra + Submitit handle Slurm submission automatically via `configs/_global.yaml`. Run each template by module path:

```bash
uv run python -m llm.text_classification.launch compute=vaughan/a40_1x --multirun

uv run python -m llm.text_classification.launch compute=vaughan/a40_1x requeue=on --multirun

uv run python -m llm.text_classification.launch compute=vaughan/a40_1x requeue=off --multirun
```

<!-- Override any Hydra value at the CLI, for example:
```bash
uv run python -m llm.launch \\
  compute=killarney/h100_1x \\
  trainer.num_train_epochs=3 \\
  work_dir=/scratch/$USER/myrun
``` -->

## Checkpointing & requeue

All templates can be made requeue-able on Slurm via Submitit:

1. Implement `checkpoint()` in your `train.py` and return a `submitit.helpers.DelayedSubmission`:
   ```python
   import submitit

   def checkpoint(self, *args, **kwargs):
       return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
   ```
2. Ensure your code saves state periodically (so a requeued run can resume).
3. In `configs/_global.yaml`, enable requeue:
   ```yaml
   hydra:
     launcher:
       params:
         max_num_timeout: 2
         additional_parameters:
           requeue: ""
   ```

See:
- Hydra Submitit launcher: https://hydra.cc/docs/plugins/submitit_launcher
- Submitit (DelayedSubmission): https://github.com/facebookincubator/submitit
