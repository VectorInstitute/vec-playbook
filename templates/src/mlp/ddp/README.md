# MLP Distributed Data Parallel Template

*Data Parallelism* lets you to split your data across multiple accelerators so that you can train your model faster!

Most of the time all your accelerators (GPUs) will be on the same machine (node), and that simplifies things. However if you are using a large number of GPUs that can't fit on a single machine, then you'll have to use multiple machines (nodes). For example, on the Killarney cluster, L40's have a maximum of 4 per node and H100's have a maximum of 8 per nodes. Data Parallelism across multiple nodes is referred to as *Distributed Data Parallelism* (DDP). By default DDP works for both single node and multi-node settings.

This example implements a simple MLP using DDP.

## DDP Background

**World Size:** The total number of GPU's across all nodes

**Rank:** Integer ID for a single GPU. Unique across all nodes. (from `0` to `world_size - 1`)

**Local Rank:** Integer ID for a single GPU. Unique only within a node. (from `0` to `num_gpus_per_node - 1`)

## DDP Setup

Unlike `torchrun`, Submitit is a **job scheduler integration**, not a distributed orchestrator.
It spawns one process per GPU (or per `tasks_per_node`), but it does **not automatically set** the PyTorch environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) required by `torch.distributed`.

However, Submitit automatically determines the distributed context (each task’s **global rank**, **local rank**, **world size**, and **hostnames**).
You don’t manually assign local ranks; you retrieve them from `submitit.JobEnvironment()` and use them to initialize PyTorch DDP:

```python
job_env = submitit.JobEnvironment()
rank = job_env.global_rank
local_rank = job_env.local_rank
world_size = job_env.num_tasks
```

Once you retrieve these values, export them as environment variables and call:

```python
torch.distributed.init_process_group(init_method="env://", backend="nccl")
```

This pattern is the standard way to perform DDP initialization with Submitit when not using `torchrun`
([MosaicML Docs](https://docs.mosaicml.com/projects/composer/en/stable/examples/training_with_submitit.html),
[Hydra Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher/),
[PyTorch Forum Discussion](https://discuss.pytorch.org/t/using-submitit-for-distributed-training/121881),
[Fairseq Example](https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/submitit_train.py)).

Submitit also provides an optional helper class, `submitit.helpers.TorchDistributedEnvironment`, which wraps `JobEnvironment`.
It automatically exports the standard PyTorch environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`) so that you can initialize DDP with `init_method="env://"` directly. Think of it as a convenience layer built on top of `JobEnvironment`. `JobEnvironment` also exposes extra metadata like `hostnames` and `hostname`, which can be helpful for advanced or custom multi-node configurations.

For a minimal example that uses `submitit.helpers.TorchDistributedEnvironment()` together with
`torch.distributed.init_process_group(init_method="env://")`, see the official Submitit example
[`docs/examples/torch_distributed.py`](https://github.com/facebookincubator/submitit/blob/main/docs/examples/torch_distributed.py).


### Logging in DDP (Hydra + Submitit)

To avoid duplicated lines in the global Hydra log, we log with `logger` only on **rank 0**.
For per-rank visibility, use `print()` on non-zero ranks. Those messages appear only in that rank’s stdout (Submitit/Slurm per-task output).

- `logger.info(...)` (rank 0): goes to the single, global Hydra log for the run.
- `print(...)` (ranks > 0): stays in the rank-local stdout, not in the global Hydra log.
