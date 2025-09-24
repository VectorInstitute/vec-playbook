# Templates

Templates for training ML models workflows on Bon Echo and Killarney clusters using Hydra and Submitit.

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

Hydra starts from `configs/_global.yaml`, pulls in the appropriate entries from `configs/user.yaml` and `configs/compute/*`, then merges the template's own `config.yaml` before forwarding the resolved configuration to Submitit; CLI overrides (e.g. `compute=killarney/h100_1x`) are applied in that final merge, so every launch script receives a single, fully-specified config that Submitit uses to submit or run locally.

## Local Setup

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

## Cluster Setup

1) Provide your user Slurm account and any optional parameters in `templates/configs/user.yaml`.

```yaml
user:
  slurm:
    account: YOUR_ACCOUNT
    # additional_parameters:
    #   qos: m2  # example Bon Echo QoS
```

Uncomment and edit `additional_parameters` entries as needed. Use CLI overrides for alternate accounts or QoS when launching jobs, for example `... user.slurm.account=ACCOUNT_B user.slurm.additional_parameters.qos=fast`.

2) Pick a compute preset:
- `templates/configs/compute/bon_echo/*` (A40, A100)
- `templates/configs/compute/killarney/*` (L40S, H100)
- Create your own preset under `templates/configs/compute/` if you need different resources (match the YAML shape used in the existing files).

## Running Templates

All launchers follow the same pattern: use `uv run python -m <package>.launch` with Hydra overrides that select compute presets, requeue behaviour, and any template-specific hyperparameters.
### Command Pattern

```bash
uv run python -m <template_pkg>.launch \
  compute=<cluster>/<preset> \
  requeue=<on|off> \
  <other.hydra.or.template.overrides> \
  --multirun
```

- `compute=<cluster>/<preset>` chooses the Slurm resources defined under `templates/configs/compute/` (or a custom preset you add).
- `requeue=<on|off>` toggles the Submitit requeue flag described in the checkpointing section.
- Additional Hydra overrides use `key=value` syntax; nested keys follow the YAML structure (e.g., `trainer.learning_rate=5e-4`).
- Use of `--multirun` is required for the launcher to be picked up..
- Prepend `+` to introduce new keys at runtime, like `+trainer.notes=baseline_a`.

### Examples (single parameter set)

```bash
# Submit a single-GPU job on Bon Echo A40
uv run python -m mlp.single.launch compute=bon_echo/a40_1x requeue=off --multirun

# Submit a single-GPU job on Killarney L40S
uv run python -m mlp.single.launch compute=killarney/l40s_1x requeue=off --multirun

# Fine-tune a text classifier template with custom learning rate
uv run python -m llm.text_classification.launch \
  compute=killarney/l40s_1x \
  +trainer.learning_rate=5e-4 \
  --multirun
```

Hydra blocks until the job finishes (or fails). For long or interactive sessions, wrap the command in `tmux`, `screen`, or submit a wrapper script as shown below.
### Practical Patterns for Long Jobs

```bash
# Start a persistent session
tmux new-session -s my_training

# Run your job/sweep inside tmux
uv run python -m llm.text_classification.launch compute=bon_echo/a40_1x --multirun

# Detach with Ctrl+B, D (can close laptop/disconnect)
# Later reattach with: tmux attach -s my_training
```

### Parameter Sweeps with Hydra

Hydra sweeps expand comma-separated value lists into Cartesian products and schedule each configuration as a separate Submitit job. Output directories are numbered based on Hydra's sweep index.

```bash
# Sweep learning rate and hidden size for the MLP template
uv run python -m mlp.single.launch \
  +trainer.learning_rate=1e-2,1e-3,1e-4 \
  +trainer.hidden_dim=64,128,256 \
  compute=bon_echo/a40_1x \
  --multirun

# Sweep batch size and LR for the VLM captioning template
uv run python -m vlm.image_captioning.launch \
  +trainer.batch_size=8,16,32 \
  +trainer.learning_rate=1e-4,5e-5 \
  compute=killarney/h100_1x \
  --multirun
```

### Monitoring Jobs

By default, Hydra and Submitit create the working directory at `~/vec_jobs/<timestamp>` (see `configs/_global.yaml`). Override it when needed with flags such as `paths.work_root=/scratch/$USER` or `work_dir=/scratch/$USER/vec_jobs/${experiment_name}`.

```bash
# Check SLURM job status
squeue -u $USER

# Inspect the latest work directory
ls -1t ~/vec_jobs | head

# Follow Submitit stdout for the most recent job
tail -f ~/vec_jobs/YYYYMMDD-HHMMSS/.submitit/*/stdout*
```
## Checkpointing & Requeue

Checkpointing lets Submitit resubmit interrupted jobs (preemption, timeout, manual `scontrol requeue`) without restarting from scratch. The templates already subclass `submitit.helpers.Checkpointable`, so they ship with a default `checkpoint()` that returns `DelayedSubmission(self, *args, **kwargs)`. You simply need to persist enough training state to continue where you left off.

Submitit’s official [checkpointing guide](https://github.com/facebookincubator/submitit/blob/main/docs/checkpointing.md) covers how the `checkpoint()` hook works under the hood and provides additional patterns (e.g., swapping callables, partial pickling) if you need more control.

**Toggling requeue behaviour**
- Defaults live in `configs/requeue/{on,off}.yaml`. Pick the version you want via `requeue=on` or `requeue=off` on the CLI. (`off` disables the Slurm `--requeue` flag.)
- Global safeguards such as `max_num_timeout` come from `configs/_global.yaml`; adjust them if your workload needs more automatic retries.

**Implementation checklist**
1. Save checkpoints regularly inside `cfg.work_dir` (e.g., `outputs/checkpoint-epoch-*`). Capture model weights, optimizer state, and any metadata you need to resume.
2. On startup (`__call__`), look for the most recent checkpoint and restore state before training continues. The templates include helper methods (`_latest_checkpoint`) you can reuse or extend.
3. Ensure your `checkpoint()` method returns a `DelayedSubmission` that recreates the callable with the same arguments. If you need custom behaviour (changing hyperparameters, skipping corrupt steps), instantiate a new callable and pass it to `DelayedSubmission` instead of `self`.
4. Test the flow by requeueing a running job (`scancel --signal=USR1 <jobid>` or Submitit's `job._interrupt(timeout=True)`) to confirm state is restored as expected.


## Resources
- Submitit: https://github.com/facebookincubator/submitit
- Hydra Submitit launcher: https://hydra.cc/docs/plugins/submitit_launcher
