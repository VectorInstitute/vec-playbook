# Templates

Templates for training ML models on Bon Echo and Killarney clusters using Hydra and Submitit.

## Layout

```
templates/
├── src/
│   ├── mlp/              # multi-layer perceptrons
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
  account: ACCOUNT
  partition: PARTITION
  qos: QOS
```

4) Pick a compute preset:
- `templates/configs/compute/bon_echo/*` (A40, A100)
- `templates/configs/compute/killarney/*` (L40S, H100)

## Running Templates

### SLURM Submission Requires `--multirun`

**All SLURM jobs must use the `--multirun` flag**, even for single runs. This is how Hydra's submitit launcher works:

- **Without `--multirun`**: Runs locally (CPU, no SLURM submission)
- **With `--multirun`**: Submits to SLURM (GPU, proper cluster resources)


### Single Runs (Single Parameter Set)
```bash
uv run python -m mlp.single_not_checkpointable.launch compute=bon_echo/a40_1x requeue=off --multirun
uv run python -m mlp.single.launch compute=bon_echo/a40_1x requeue=off --multirun
uv run python -m mlp.ddp.launch compute=bon_echo/a40_2x requeue=off --multirun  # Multi-GPU
uv run python -m llm.text_classification.launch compute=bon_echo/a40_1x requeue=off --multirun
uv run python -m vlm.image_captioning.launch compute=bon_echo/a40_1x requeue=off --multirun
```

**Note**: Hydra will wait for job completion before returning. For long training jobs, use tmux or run in background.

### Parameter Sweeps (Multiple Parameter Sets)
```bash
# Run multiple experiments with different parameters
uv run python -m mlp.single.launch \
  learning_rate=1e-2,1e-3,1e-4 \
  hidden_dim=64,128,256 \
  compute=bon_echo/a40_1x --multirun

uv run python -m llm.text_classification.launch \
  learning_rate=1e-3,1e-4,1e-5 \
  per_device_train_batch_size=16,32 \
  compute=bon_echo/a40_1x --multirun

uv run python -m vlm.image_captioning.launch \
  learning_rate=1e-4,1e-5,1e-6 \
  batch_size=8,16,32 \
  compute=bon_echo/a40_1x --multirun
```
### Practical Patterns for Long Jobs

#### Option 1: tmux (Recommended)
```bash
# Start a persistent session
tmux new-session -s my_training

# Run your job/sweep inside tmux
uv run python -m llm.text_classification.launch compute=bon_echo/a40_1x --multirun

# Detach with Ctrl+B, D (can close laptop/disconnect)
# Later reattach with: tmux attach -s my_training
```

#### Option 2: Submit Hydra as SLURM Job
```bash
# Submit the entire sweep as a single SLURM job
sbatch --job-name=my_sweep --time=24:00:00 --partition=cpu --wrap="\
cd $(pwd) && \
uv run python -m llm.text_classification.launch \
  learning_rate=1e-3,1e-4,1e-5 \
  --multirun"
```


### Monitoring Jobs

```bash
# Check SLURM job status
squeue -u $USER

# Check Hydra output directory
ls ~/vec_jobs/

# Follow logs in real-time
tail -f ~/vec_jobs/YYYYMMDD-HHMMSS/.submitit/*/std*.out
```
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
