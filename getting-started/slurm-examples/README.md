# Slurm Examples

A collection of Slurm example scripts that show how to implement real-world use cases. These examples were all specifically designed for and tested on our **Killarney** cluster. They will not work as-is on other clusters, which might have different hardware configurations, software environments, datasets, and other such things.

This is still a work in progress. We'll be adding more examples soon.

## Contents

 - [**Hello, Killarney!**](./hello-killarney/): The most basic Hello World job, targeting Killarney-specific resources.
 - [**Hello, Killarney Multinode!**](./hello-killarney-multinode): An adaptation of the above Hello Killarney job, demonstrating how to deploy a workload on the cluster that runs across multiple nodes.
 - [**Imagenet Training**](./imagenet): A simple training job for the popular [ImageNet](https://www.image-net.org/) dataset, using an example training script provided by Pytorch.
 - [**Imagenet Multinode Training**](./imagenet-multinode): Distributed training of the ImageNet dataset across 4 different nodes.
