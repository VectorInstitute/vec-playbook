# Templates

Templates for training ML models workflows on Bon Echo and Killarney clusters using Hydra and Submitit.

[Hydra](https://hydra.cc/docs/intro/) is a python framework for creating configurable experiments that you can change through a config file. One of it's main uses is its ability to automatically perform hyperparameter sweeps for model training.

[submitit](https://github.com/facebookincubator/submitit) is a simple python package that lets you submit slurm jobs programmatically and automatically access and manipulate the results of those jobs once they are complete. It also handles automatic requeing of jobs should they be inturrupted for some reason.

Hydra conveniently has a submitit plugin that allows them to work together. Put simply, using these tools you can automatically queue up a large number of experiments, run dependent experiments sequentially, requeue long running experiments and more.

## Local Setup

1) Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Clone the vec-playbook repository
```bash
git clone https://github.com/VectorInstitute/vec-playbook.git
```

3) Resolve and install dependencies from `pyproject.toml` into a virtual environment:
```bash
cd path/to/vec-playbook
uv sync  # Automatically installs dependencies in vec-playbook/.venv
```

Finally, ensure you're working directory (by default your cluster scratch space) exists and that you have access to the resources you're requesting on the cluster. 

### UV Tip for Killarney

If you're on killarney you'll have to clone the repository into your scratch space. You can't run files stored in your home directory. The UV cache by default is located in your home directory which is a different filesystem. This breaks uv's default method of hardlinking packages to avoid having to redownload packages. You can either change your cache directory to be on the same filesystem or use `--link-mode=copy`. Avoid using symlink mode as this can break things.

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

Hydra starts from `configs/_global.yaml` and pulls in the appropriate entries from `configs/user.yaml` and `configs/compute/*`. The launch script within each template then merges the template's own local `config.yaml` before forwarding the resolved configuration to Submitit; CLI overrides (e.g. `compute=killarney/h100_1x`) are applied in that final merge, so every launch script receives a single, fully-specified config that Submitit uses to submit or run locally.

The `_global.yaml` config contains the bulk of the autoconfiguration. Placeholders are used to automatically fill values with values from other configuration files. `hydra.launcher` arguments largely align with the CLI arguments available for the [sbatch](https://slurm.schedmd.com/sbatch.html) command. See [this](https://hydra.cc/docs/plugins/submitit_launcher/) page for the officialy available hydra slurm launcher parameters. Note that the majority of the parameters are sourced from the selected `compute` config.


## Cluster Setup

1) Provide your Slurm user account and any optional parameters in `templates/configs/user.yaml`.

```yaml
user:
  slurm:
    account: YOUR_ACCOUNT
    # additional_parameters:
    #   qos: m2  # example Bon Echo QoS
```
**NOTE:** why is qos used as example of additional parameter here when it is an official launcher parameter that seems to be sourced from compute config?

Uncomment and edit `additional_parameters` entries as needed. This field is solely for sbatch arguments not already available in the [Hydra Submitit Slurm Launcher Plugin](https://hydra.cc/docs/plugins/submitit_launcher/). Use CLI overrides for alternate accounts or QoS when launching jobs, for example `... user.slurm.account=ACCOUNT_B user.slurm.additional_parameters.qos=fast`.

2) Pick a compute preset to use in the next section:
- `templates/configs/compute/bon_echo/*` (A40, A100)
- `templates/configs/compute/killarney/*` (L40S, H100)
- Create your own preset under `templates/configs/compute/` if you need different resources (match the YAML shape used in the existing files).

## Running Templates

All launchers follow the same pattern: use `uv run python -m <package>.launch` with Hydra overrides that select compute presets, requeue behaviour, and any template-specific hyperparameters. uv will automatically detect the virtual environment located in `.venv` of your CWD. The templates are automatically loaded as python modules by `uv`. If you add your own template you will have to sync the virtual environment using `uv sync`.

### Command Pattern

```bash
uv run python -m <template_pkg>.launch \
  compute=<cluster>/<preset> \
  requeue=<on|off> \
  <other.hydra.or.template.overrides> \
  --multirun
```

-  `<template_pkg>`: The module path to the template launch script (eg.  `mlp.single`)
- `compute=<cluster>/<preset>`: chooses the Slurm resources defined under `templates/configs/compute/` (or a custom preset you add).
- `requeue=<on|off>`: toggles the Submitit requeue flag described in the checkpointing section.
- Additional Hydra overrides use `key=value` syntax; nested keys follow the YAML structure (e.g., `trainer.learning_rate=5e-4`).
- Prepend `+` to introduce new keys (not already present in config) at runtime, like `+trainer.notes=baseline_a`.
- Use of `--multirun` is required for the launcher to be picked up.

[//]: <> (What does "picked up" mean when explaining --multirun flag?)

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

Your output should look something like this:
```
[2025-09-29 11:06:00,546][HYDRA] Submitit 'slurm' sweep output dir : /scratch/$USER/vec_jobs/20250929-110600
[2025-09-29 11:06:00,546][HYDRA]        #0 : compute=killarney/l40s_1x
```

[//]: <> (Why does learning_rate need the + prepended if its already in local config?)
[//]: <> (Perhaps a little more clarity on this)
[//]: <> (`+trainer.num_epochs=100` override did not work for mlp.single)
[//]: <> (multirun.yaml is long and confusing and still contains placeholders. Is there a way to save the final static config yaml?)

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

[//]: <> (Sweep seems to work, but checkpoints overwrite eachother i'm assuming? Hydra does not create subdirectories in outputs for sweep.l)

```bash
# Sweep learning rate and hidden size for the MLP template
uv run python -m mlp.single.launch \
  +trainer.learning_rate=1e-2,1e-3 \
  +trainer.hidden_dim=64,128 \
  compute=bon_echo/a40_1x \
  --multirun

# Sweep batch size and LR for the VLM captioning template
uv run python -m vlm.image_captioning.launch \
  +trainer.batch_size=8,16,32 \
  +trainer.learning_rate=1e-4,5e-5 \
  compute=killarney/h100_1x \
  --multirun
```

Your output for a sweep should look something like this:

```
[2025-09-29 11:06:00,546][HYDRA] Submitit 'slurm' sweep output dir : /scratch/$USER/vec_jobs/20250929-110600
[2025-09-29 11:06:00,546][HYDRA]    #0 : +trainer.learning_rate=0.01 +trainer.hidden_dim=64 compute=killarney/l40s_1x
[2025-09-29 11:06:00,546][HYDRA]    #1 : +trainer.learning_rate=0.01 +trainer.hidden_dim=128 compute=killarney/l40s_1x
[2025-09-29 11:06:00,546][HYDRA]    #2 : +trainer.learning_rate=0.001 +trainer.hidden_dim=64 compute=killarney/l40s_1x
[2025-09-29 11:06:00,546][HYDRA]    #3 : +trainer.learning_rate=0.001 +trainer.hidden_dim=128 compute=killarney/l40s_1x
```

### Monitoring Jobs

By default, Hydra and Submitit create the working directory at `~/vec_jobs/<timestamp>` (see `configs/_global.yaml`). Override it when needed with flags such as `paths.work_root=/scratch/$USER` or `work_dir=/scratch/$USER/vec_jobs/${experiment_name}`.

```bash
# Check SLURM job status
squeue --me

# Inspect the latest work directory
ls -1t ~/vec_jobs | head

# Follow Submitit stdout for the most recent job
tail -f ~/vec_jobs/YYYYMMDD-HHMMSS/.submitit/*/stdout*
```
## Checkpointing & Requeue

Checkpointing lets Submitit resubmit interrupted jobs (preemption, timeout, manual `scontrol requeue`) without restarting from scratch. The templates already subclass `submitit.helpers.Checkpointable`, so they ship with a default `checkpoint()` that returns `DelayedSubmission(self, *args, **kwargs)`. You simply need to persist enough training state to continue where you left off. See [mlp.single.train](src/mlp/single/train.py) for an example of a basic checkpointing implementation.

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
