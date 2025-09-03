### MLP training templates

This directory includes templates for Multi-Layer Perceptron training tasks:

- [single_not_checkpointable](single_not_checkpointable/): Simple MLP trainer without checkpointing (runs from start to finish).
- [single](single/): MLP trainer with checkpointing support (can resume from interruptions).
- [ddp](ddp/): Distributed MLP trainer using PyTorch DDP for multi-GPU training.

All trainers use dummy synthetic data for quick testing and demonstration.
