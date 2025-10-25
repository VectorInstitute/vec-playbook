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

Each template directory contains a `launch.py`, a `train.py`, and a `config.yaml`.
The `configs/` directory defines Slurm presets and shared Hydra + Submitit settings.

The launch script contains the `hydra.main` decorator which points hydra to the templates local `config.yaml`. This `config.yaml` imports the `_global` config from the `configs/` directory, which in turn imports other preset configs.

Most templates import the `_global.yaml` config from the `configs/` directory as a base experimental setup, and are therefore dependent on it's settings. The global config in turn imports other preset configs such as the `user.yaml` config and the compute configs. Modifying the `_global.yaml` file may break some of the other templates. Therefore be careful making changes to `_global.yaml` settings, if the settings do not need to be globally applied to all templates consider including them in the local config instead. Hydra takes the starting local config, populates it with the additional fields from all its dependencies, and provides that to submitit to launch a job. Submitit executes the function decorated by `hydra.main` as a slurm job. The fully specified config that was provided to slurm is passed to that function as an argument.

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

Uncomment and edit `additional_parameters` entries as needed. This field is solely for sbatch arguments not already available in the [Hydra Submitit Slurm Launcher Plugin](https://hydra.cc/docs/plugins/submitit_launcher/). Use CLI overrides for alternate accounts or QoS when launching jobs, for example `... user.slurm.account=ACCOUNT_B user.slurm.additional_parameters.qos=fast`.

[//]: <> (Will specifying qos as an additional parameter overwrite the qos in compute setting?)

2) Pick a compute preset to use in the next section:
- `templates/configs/compute/bon_echo/*` (A40, A100)
- `templates/configs/compute/killarney/*` (L40S, H100)
- Create your own preset under `templates/configs/compute/` if you need different resources (match the YAML shape used in the existing files).

## Running Templates

All launchers follow the same pattern: use `uv run python -m <templatee>.launch` with Hydra overrides that select compute presets, requeue behaviour, and any template-specific hyperparameters. uv will automatically detect the virtual environment located in `.venv` of your CWD. The templates are automatically loaded as python modules by `uv`. If you add your own template you will have to sync the virtual environment using `uv sync`.

### Command Pattern

```bash
uv run python -m <template_pkg>.launch \
  compute=<cluster>/<preset> \
  requeue=<on|off> \
  <config.overrides> \
  <new-keys> \
  --multirun
```

- `<template_pkg>`: The module path to the template launch script (eg.  `mlp.single`)
- `compute=<cluster>/<preset>`: chooses the Slurm resources defined under `templates/configs/compute/` (or a custom preset you add).
- `requeue=<on|off>`: toggles the Submitit requeue flag described in the checkpointing section.
- Additional config overrides use `key=value` syntax; nested keys follow the YAML structure (e.g., `compute.mem_gb=32`).
- Keys not already present in the local `config.yaml` or it's dependencies (`_global.yaml`, `user.yaml`, compute yamls) must be prepended with a `+`. This denotes the key as being new rather than an override.
- Use of `--multirun` is required to use the submitit slurm launcher, even if you are only performing a single run. Otherwise the model will attempt to train locally on your login node.

### Examples (single parameter set)

```bash
# Submit a single-GPU job on Bon Echo A40
uv run python -m mlp.single.launch compute=bon_echo/a40_1x requeue=off --multirun

# Submit a single-GPU job on Killarney L40S
uv run python -m mlp.single.launch compute=killarney/l40s_1x requeue=off --multirun

# Fine-tune a text classifier template with custom learning rate
uv run python -m llm.text_classification.launch \
  compute=killarney/l40s_1x \
  trainer.learning_rate=5e-4 \
  --multirun
```

Your output should look something like this:
```
[2025-09-29 11:06:00,546][HYDRA] Submitit 'slurm' sweep output dir : /scratch/$USER/vec_jobs/20250929-110600
[2025-09-29 11:06:00,546][HYDRA]        #0 : compute=killarney/l40s_1x
```

### Practical Patterns for Long Jobs

Hydra blocks until the job finishes (or fails). For long or interactive sessions, wrap the command in `tmux`, `screen`, or submit a wrapper script as shown below.

```bash
# Start a persistent session
tmux new-session -s my_training

# Run your job/sweep inside tmux
uv run python -m llm.text_classification.launch compute=bon_echo/a40_1x --multirun

# Detach with Ctrl+B, D (can close laptop/disconnect)
# Later reattach with: tmux attach -s my_training
```

**Note:** If you `ctrl+C` the command, this does not cancel the slurm jobs if they are already running. You have to manually use scancel to cancel the running jobs. Use `squeue --me` to see running jobs under your account.

### Parameter Sweeps with Hydra

Hydra sweeps expand comma-separated value lists into Cartesian products and schedule each configuration as a separate Submitit job. Output directories are numbered based on Hydra's sweep index.

```bash
# Sweep learning rate and hidden size for the MLP template
uv run python -m mlp.single.launch \
  trainer.learning_rate=1e-2,1e-3 \
  trainer.hidden_dim=64,128 \
  compute=bon_echo/a40_1x \
  --multirun

# Sweep batch size and LR for the VLM captioning template
uv run python -m vlm.image_captioning.launch \
  trainer.batch_size=8,16,32 \
  trainer.learning_rate=1e-4,5e-5 \
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

By default, Hydra and Submitit create a `vec_jobs/<timestamp>` working directory in your scratch folder (`/scratch/$USER` on killarney, `/scratch/ssd004/scratch/u$USER` on bon-echo). Override it when needed with flags such as `paths.work_root=/scratch/$USER` or `paths.work_dir=/scratch/$USER/vec_jobs/${experiment_name}`. These are set in `configs/_global.yaml`.

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

Note that in order to prevent multirun jobs (such as parameter sweeps) from overwriting eachothers configs, the checkpoint output directory must be different for each run. We handle this in the launch script by using the `hydra.runtime.output_dir` to set a dynamic output dir in `cfg.paths.out_dir`. The runtime output dir will always be unique to each run. For multirun sweeps, subdirectories are automatically generated by hydra. You can configure hydra to customize how these subdirectoires are named but by default they're just monotonically increasing integers.

**Toggling requeue behaviour**
- Defaults live in `configs/requeue/{on,off}.yaml`. Pick the version you want via `requeue=on` or `requeue=off` on the CLI. (`off` disables the Slurm `--requeue` flag.)
- Global safeguards such as `max_num_timeout` come from `configs/_global.yaml`; adjust them if your workload needs more automatic retries.

**Implementation checklist**
1. Save checkpoints regularly inside `cfg.paths.out_dir` (e.g., `outputs/checkpoint-epoch-*`). Capture model weights, optimizer state, and any metadata you need to resume.
2. On startup (`__call__`), look for the most recent checkpoint and restore state before training continues. The templates include helper methods (`_latest_checkpoint`) you can reuse or extend.
3. Ensure your `checkpoint()` method returns a `DelayedSubmission` that recreates the callable with the same arguments. If you need custom behaviour (changing hyperparameters, skipping corrupt steps), instantiate a new callable and pass it to `DelayedSubmission` instead of `self`.
4. Test the flow by requeueing a running job (`scancel --signal=USR1 <jobid>` or Submitit's `job._interrupt(timeout=True)`) to confirm state is restored as expected.

## Resources
- Submitit: https://github.com/facebookincubator/submitit
- Hydra Submitit launcher: https://hydra.cc/docs/plugins/submitit_launcher

## Understanding Job Outputs

After running a template, the output artifacts will be saved to the work_dir specified in `_global.yaml`. By default this is `$SCRATCH_DIR/vec_jobs/<timestamp>`. The directory structure for you're outputs will look something like this:

```
vec_jobs/<timestamp>/
├── multirun.yaml  # Template config used by hydra for all runs
├── submitit_logs/
│  ├── <base-slurm-job-ID>/  # If there is only 1 run, then all the submitit logs will be in here instead
│  │   └── <slurm-job-ID>_submission.sh  # The sbatch script used to submit all the slurm jobs
│  ├── <base-slurm-job-ID>_<hydra-run-id>/  # This is the run's slurm-job-ID, one dir for each run
│  │   │   # <slurm-job-id> = <base-slurm-job-id>_<hydra-run-id>
│  │   ├── <slurm-job-ID>_<process-id>_log.err  # stderr
│  │   ├── <slurm-job-ID>_<process-id>_log.out  # stdout
│  │   ├── <slurm-job-ID>_<process-id>_result.pkl  # You can you have multiple processes running within one job
│  │   ...
│  │   └── <slurm-job-ID>_submitted.pkl
│  ...
│  └── <slurm-jon-ID>/
│      ...
│      └── ...
│── <hydra-run-id>/ # This is the actual job/run output. One for each run
│  ├── <hydra.job.name>.log  # Hydra Logs: Only contains log messages, not stdout or stderr
│  ├── outputs/  # Contains model outputs saved to cfg.paths.out_dir (eg. checkpoints)
│  └── hydra_configs
│      ├── config.yaml  # The final config passed to the function decorated by @hydra.main (for this run)
│      ├── overrides.yaml  # CLI overrides that were used for this run
│      ├── hydra.yaml  # The hydra settings that were used for this run (some placeholder values still present)
│      └── hydra_resolved.yaml  # The hydra settings that were used for this run (with all placeholder values resolved)
│
...
└── <hydra-run-id>/
    ...
    └── ...
```

**Notes:**
- print messages will not be sent to launch.log, use a logger instead (see example templates)
- `multirun.yaml` and `hydra.yaml` will contain placeholder values (eg. `${oc.select:compute.mem_gb}`). These are used to fill in the values with values from other parts of the config or other configs included in the defaults. See hydra documentation for more detail.
- When doing a hyperparameter sweep, a run is performed for each unique combination of hyperparameters. Each run is run as a separate slurm job with a unique slurm ID.
  - All the runs are submitted as separate jobs using the slurm `--array` feature. Therefore there is a base slurm job id shared by all runs. The slurm-job-id actually used by slurm for each run is a combination of the base slurm job ID and the hydra run ID (eg. `1186868_1`). For multirun jobs you might end up with log files like: `1186868_1_0`. Not sure what the second integer is as it doesn't necessarily line up with the hydra run id. Most likely a process ID.
- The hydra logs are a good place to start to see the output of your job. If information is missing, or if an error occurs, the submitit logs are the source of truth and should contain everything. Sometimes exceptions are not captured in the hydra logs.
