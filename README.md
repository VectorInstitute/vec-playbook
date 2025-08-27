# vec-playbook

Starter templates for training ML models on Vaughan and Killarney clusters using
Hydra and Submitit.

## Layout

```
vec-playbook/
├── [starters](./starters)
│   ├── [basic_mlp](./starters/basic_mlp)        # simple MNIST MLP
│   ├── [llm](./starters/llm)                    # language models
│   ├── [vlm](./starters/vlm)                    # vision–language models
│   └── [rl](./starters/rl)                      # reinforcement learning
├── [configs](./configs)                         # Hydra + Submitit configs
└── [scripts/train.py](./scripts/train.py)       # Hydra entrypoint
```

Each `starter/*/` is self-contained; it has a `runner.py` and a `config.yaml`.

## Setup

1. Install the environment:

   a. Create and activate a virtual environment:

   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

   b. Resolve and install dependencies from `pyproject.toml`:

   ```bash
   uv lock
   uv sync
   ```

2. Edit `configs/user.yaml`:
   ```yaml
   slurm:
     account: ACCOUNT
     partition: PARTITION
     qos: QOS
   ```

3. Pick a compute preset (`configs/compute/vaughan/*` or `configs/compute/killarney/*`).

4. Run jobs:

   Hydra + Submitit handle Slurm submission automatically. The `configs/_global.yaml` file contains generic settings, while each researcher must edit `configs/user.yaml` with their own account info.

   Examples:

   ```bash
   # Basic MLP on 1×A40 (Vaughan)
   python -m scripts.train starter=basic_mlp compute=vaughan/a40_1x

   # Text classification with DistilBERT on 1×L40S (Killarney)
   python -m scripts.train starter=llm_text_classification compute=killarney/l40s_1x
   ```

   You can override any configuration at the command line:

   ```bash
   python -m scripts.train starter=llm_text_classification \
     compute=killarney/h100_1x \
     trainer.num_train_epochs=3 \
     work_dir=/scratch/$USER/myrun
   ```


## Checkpoint and requeue

## Checkpointing & Requeue

All starters in this repo can be made **requeue-able** on Slurm by using
[Submitit’s checkpointing mechanism](https://hydra.cc/docs/plugins/submitit_launcher/#checkpointing)
and [Submitit DelayedSubmission](https://github.com/facebookincubator/submitit).

To enable this:

1. **Implement `checkpoint()`** in your runner class.
   The method must return a `submitit.helpers.DelayedSubmission` of the same object.
   Example pattern:
   ```python
   import submitit

   def checkpoint(self, *args, **kwargs):
       return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
   ```

2. **Ensure your code saves state periodically** (e.g. model checkpoints, training progress)
   so that when the job is requeued, the restarted run can resume from where it left off.

3. **Configure requeue in `_global.yaml`**:
   ```yaml
   hydra:
     launcher:
       params:
         max_num_timeout: 2        # how many times Submitit will requeue
         additional_parameters:
           requeue: ""             # request Slurm to allow requeue
   ```

With this setup:
- When the scheduler preempts or a time limit is reached, Submitit calls `checkpoint()`.
- A new job is automatically resubmitted with the same configuration.
- On restart, your runner should load the saved state and continue.

For more details, see:
- Hydra Submitit launcher docs: https://hydra.cc/docs/plugins/submitit_launcher
- Submitit source/docs on `DelayedSubmission`: https://github.com/facebookincubator/submitit
