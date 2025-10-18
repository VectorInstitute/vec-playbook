# Distributed Data Parallel Example

> :warning: WIP: This template is a work in progress and does not use DDP in its current state.

*Data Parallelism* lets you to split your data across multiple accelerators so that you can train your model faster! 

Most of the time all your accelerators (gpus) will be on the same machine (node), and that simplifies things. However if you are using a large number of gpus that can't fit on a single machine, then you'll have to use multiple machines (nodes). For example, on the Killarney cluster, L40's have a maximum of 4 per node and H100's have a maximum of 8 per nodes. Data Parallelism across multiple nodes is referred to as *Distributed Data Parallelism* (DDP). By default DDP works for both single node and multi-node settings.

This example implements a simple MLP using DDP.

## DDP Background

**World Size:** The total number of gpu's across all nodes

**Rank:** Integer ID for a single gpu. Unique across all nodes. (from `0` to `world_size - 1`)

**Local Rank:** Integer ID for a single gpu. Unique only within a node. (from `0` to `num_gpus_per_node - 1`)