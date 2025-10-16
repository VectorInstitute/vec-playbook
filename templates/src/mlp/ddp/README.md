# MLP Distributed Data Parallel Template

*Data Parallelism* lets you to split your data across multiple accelerators so that you can train your model faster!

Most of the time all your accelerators (GPUs) will be on the same machine (node), and that simplifies things. However if you are using a large number of GPUs that can't fit on a single machine, then you'll have to use multiple machines (nodes). For example, on the Killarney cluster, L40's have a maximum of 4 per node and H100's have a maximum of 8 per nodes. Data Parallelism across multiple nodes is referred to as *Distributed Data Parallelism* (DDP). By default DDP works for both single node and multi-node settings.

This example implements a simple MLP using DDP.

## DDP Background

**World Size:** The total number of GPU's across all nodes

**Rank:** Integer ID for a single GPU. Unique across all nodes. (from `0` to `world_size - 1`)

**Local Rank:** Integer ID for a single GPU. Unique only within a node. (from `0` to `num_gpus_per_node - 1`)

## DDP Setup

Unlike `torchrun`, Submitit is a **job scheduler integration**, not a distributed orchestrator. It spawns one process per GPU (or per `tasks_per_node`), but it does **not automatically set** the PyTorch environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) required by `torch.distributed`.

Therefore, this project explicitly initializes the distributed environment inside the training script using `submitit.JobEnvironment()`.
This pattern is the standard way to perform DDP initialization with Submitit when not using `torchrun`
([MosaicML Docs](https://docs.mosaicml.com/projects/composer/en/stable/examples/training_with_submitit.html),
[Hydra Submitit Launcher](https://hydra.cc/docs/plugins/submitit_launcher/),
[PyTorch Forum Discussion](https://discuss.pytorch.org/t/using-submitit-for-distributed-training/121881),
[Fairseq Example](https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/submitit_train.py)).

It works for both **single-node** and **multi-node** jobs as long as the `MASTER_ADDR` points to a hostname reachable from all nodes.
