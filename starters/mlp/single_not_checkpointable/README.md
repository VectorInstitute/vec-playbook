# Simple Neural Network Training (No Checkpointing)

This example demonstrates a basic neural network trainer that **does not** inherit from `Checkpointable`. Unlike other examples in this repository, this trainer runs a complete training loop without any checkpointing or resuming capabilities.

## Features

- Simple feed-forward neural network
- No checkpointing or state persistence
- Runs training from start to finish in one go
- Easy to understand and modify
- Can be run standalone or with vec-tool

## Files

- `trainer.py` - Main trainer class that doesn't inherit from Checkpointable
- `run_standalone.py` - Standalone script to run the trainer directly
- `launch.sh` - Shell script to run via vec-tool
- `__init__.py` - Module initialization

## Usage

### Option 1: Standalone (Recommended for simplicity)

```bash
cd /path/to/vec-playbook/starters/simple_nn/single_not_checkpointable
python run_standalone.py
```

### Option 2: Direct Python import

```python
from starters.simple_nn.single_not_checkpointable.trainer import SimpleTrainer

# Create trainer with custom parameters
trainer = SimpleTrainer(
    input_dim=10,
    hidden_dim=64,
    num_classes=3,
    batch_size=32,
    lr=1e-3,
    num_epochs=50
)

# Run training
trainer()
```

### Option 3: Using vec-tool

```bash
chmod +x launch.sh
./launch.sh
```

## Key Differences from Checkpointable Examples

1. **No inheritance**: The `SimpleTrainer` class is a plain Python class, not inheriting from `Checkpointable`
2. **No state persistence**: Training runs from epoch 0 to completion every time
3. **Simpler initialization**: No need to call `super().__init__()` or handle checkpointing configuration
4. **No resume capability**: If interrupted, training must start over from the beginning
5. **Reduced default epochs**: Set to 100 epochs by default (vs 10,000 in checkpointable version) since there's no resuming

## When to Use This Approach

This non-checkpointable approach is ideal for:

- **Short training runs** that complete quickly
- **Experimentation and prototyping** where you don't need to resume
- **Simple models** that don't take long to train
- **Learning and understanding** the basic training loop without checkpointing complexity
- **Scenarios where you always want to start fresh**

For longer training runs or production scenarios, consider using the checkpointable versions that can resume from interruptions.

## Parameters

- `input_dim`: Input feature dimension (default: 10)
- `hidden_dim`: Hidden layer dimension (default: 64)
- `num_classes`: Number of output classes (default: 3)
- `batch_size`: Training batch size (default: 32)
- `lr`: Learning rate (default: 1e-5)
- `num_epochs`: Number of training epochs (default: 100)
