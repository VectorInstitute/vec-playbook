# LLM Distributed Fine-tuning Template

This template fine-tunes Hugging Face models with the **HF Trainer** and scales via **DDP** (DistributedDataParallel) or **FSDP** (Fully Sharded Data Parallel) which are PyTorch methods for **distributed training**.

## Overview: DDP vs FSDP

**DDP** runs a complete copy of the model on each GPU and synchronizes gradients across devices during training. It offers high performance and is suitable for models that comfortably fit within GPU memory.
**FSDP**, by contrast, shards model parameters, gradients, and optimizer states across GPUs to reduce memory consumption, allowing the training of much larger models.

For more information, see the official PyTorch tutorials:
- [DistributedDataParallel (DDP) Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Fully Sharded Data Parallel (FSDP) Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## Memory Breakdown and Scaling

Training memory consists of several components:

1. **Model Parameters**: The weights of the neural network
   - Size: `num_params × bytes_per_param` (2 bytes for fp16/bf16, 4 bytes for fp32)

2. **Gradients**: One gradient value per parameter for backpropagation
   - Size: Same as model parameters
   - Typically stored in the same dtype as the model parameters (fp16/bf16)

3. **Optimizer States**: Adam optimizer maintains running averages
   - Size: ~2× model parameters (first moment + second moment estimates)
   - Both are usually stored in **fp32** for numerical stability
   - **Optional:** Many mixed-precision setups also keep an **fp32 master weights** copy
   - Total optimizer footprint ≈ **2×–3× model size (in fp32)** depending on implementation

4. **Activations**: Intermediate outputs stored during the forward pass for backpropagation
   - Size: `batch_size × seq_length × hidden_dim × num_layers × constant`
   - Scales linearly with batch size and sequence length
   - Can be reduced substantially with `activation_checkpointing: true` (recomputes instead of storing; savings vary by layer, typically 30–60%)

---

### DDP vs FSDP Summary

| Method | Description | Pros | Cons |
|--------|--------------|------|------|
| **DDP** | Replicates the **entire model** on each GPU | Simple, efficient for smaller models | High memory use |
| **FSDP** | Shards parameters, gradients, and optimizer states across GPUs | Allows **larger models** | More communication overhead |

---

### With FSDP `full_shard`

- Parameters, gradients, and optimizer states are **sharded (divided)** across all GPUs.
- Activations are **not sharded**; they are fully **replicated** on each GPU.
- During training, FSDP **temporarily all-gathers parameters** per layer, so brief **memory spikes** above steady-state usage are expected.

---

### Formula

```text
Memory per GPU ≈ (Parameters + Gradients + Optimizer States) / num_GPUs + Activations

              ≈ (params × bytes_per_param × factor) / num_GPUs + (batch × seq × hidden × layers × constant)
```

Where:
- `factor = 6` → fp16/bf16 (params + grads in fp16 + m,v in fp32)
- `factor = 8` → fp16/bf16 with fp32 master weights (common in AdamW)

---

### Example

**Pythia-1B on 2× NVIDIA L40 GPUs with batch=16, seq=512:**

```text
Params: 1.0B × 2 bytes = 2.00 GiB total

Model + Gradients + Optimizer States (with master weights):
   (1.0B × 2 bytes × 8) / 2 GPUs ≈ 7.45 GiB per GPU

Activations (not sharded):
   ≈ 1.0 GiB per GPU (depends on hidden_dim × layers)

Total steady-state:
   ≈ 8.45 GiB per GPU

Transient all-gather overhead:
   +10–20% (≈ 9.3 – 10.2 GiB peak)
```

---

### To Reduce Memory

- Decrease `per_device_train_batch_size` → reduces activations
- Enable `activation_checkpointing: true` → recomputes instead of storing
- Reduce `max_length` → reduces activations linearly
- Increase number of GPUs → shards model and optimizer states further
- Use **bf16** or **fp16** mixed precision → halves parameter and gradient memory
- Use **FSDP offload to CPU/NVMe** for optimizer states if GPU memory is constrained

---

## Distributed Environment Setup (Submitit + Slurm)

In `train.py`, we use Submitit’s helper:

```python
from submitit.helpers import TorchDistributedEnvironment
TorchDistributedEnvironment().export()   # sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR/PORT
```

Then the HF `Trainer` (via `TrainingArguments`) initializes distributed training; you can also explicitly call:

```python
torch.distributed.init_process_group(backend="nccl", init_method="env://")
```

if you need lower-level control. The helper provides the same environment variables you would otherwise set by hand so that PyTorch’s `env://` init works.

---

### Tasks-per-node and GPUs-per-node

One process per GPU is the common pattern for distributed training. Concretely:

```yaml
hydra.launcher.tasks_per_node = compute.gpus_per_node
```

This makes Slurm/Submitit spawn exactly one task per GPU, where each task becomes a rank in the distributed job.

---

### What Submitit Exports

`submitit.helpers.TorchDistributedEnvironment().export()` populates:

- `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`
  (and related fields) so that `init_method="env://"` works out of the box.

---

### Binding Each Task to One GPU

Slurm's GRES plugin sets `CUDA_VISIBLE_DEVICES` for each task so the task "sees" only its assigned GPU(s). You can additionally enforce explicit GPU isolation with:

```yaml
hydra.launcher.setup:
  - "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID"
```

This ensures each task is bound to a single GPU (for example, Task 0 → GPU 0, Task 1 → GPU 1), and within each process, that GPU always appears as `cuda:0`.
This isolation prevents tasks from accessing other GPUs and keeps device handling consistent across nodes.

---

## Distributed Training Modes and Configurations

### FSDP (Fully Sharded Data Parallel)

The default configuration uses FSDP for memory-efficient training of large models:

```yaml
trainer.dist:
  mode: "fsdp"
  fsdp: ["full_shard", "auto_wrap"]
  fsdp_config:
    use_orig_params: true
    activation_checkpointing: false
    limit_all_gathers: true
    forward_prefetch: true
    sync_module_states: true
    fsdp_auto_wrap_policy: "SIZE_BASED_WRAP"
    fsdp_min_num_params: 1000000
```

**Key settings:**
- `full_shard`: Shards model parameters, gradients, and optimizer states across all GPUs
- `auto_wrap`: Automatically wraps model layers for sharding
- `SIZE_BASED_WRAP`: Wraps modules that exceed `fsdp_min_num_params`

---

### To Use DDP Instead

Set `mode: "ddp"` (or remove the `dist.mode` setting).
DDP replicates the full model on each GPU and is simpler but uses more memory.

---

## Performance and Tuning Tips

- Enable `activation_checkpointing: true` to reduce memory (at ~20–30% slower training)
- Increase `fsdp_min_num_params` for finer-grained sharding

---

## Why not torchrun here?

`torchrun` is valid for distributed launches (including on Slurm), but this template uses **Hydra’s Submitit launcher** to keep **sweeps, config composition, logging, and requeue** inside Hydra, and to avoid maintaining separate bash wrappers. Submitit handles **job submission and per-task rank context**; we still initialize PyTorch distributed via the standard env-var pathway.

If you prefer `torchrun`, you can adapt the script and configs—but you’ll then manage the Slurm submission layer (or wrap `torchrun` inside an sbatch yourself) and wire up Hydra sweeps accordingly.

---

## References

- [PyTorch distributed environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization)
- [Slurm GRES guide](https://slurm.schedmd.com/gres.html)
- [Hugging Face FSDP / Trainer documentation](https://huggingface.co/docs/transformers/fsdp)
