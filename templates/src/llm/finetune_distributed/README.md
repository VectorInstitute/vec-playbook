# LLM Distributed Fine-tuning Template

This template fine-tunes Hugging Face models with the **HF Trainer** and scales via **DDP** or **FSDP**.

## How the code works

In `train.py`, we use Submitit’s helper:

```python
from submitit.helpers import TorchDistributedEnvironment
TorchDistributedEnvironment().export()   # sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR/PORT
```

Then the HF `Trainer` (via `TrainingArguments`) initializes distributed training; you can also explicitly call `torch.distributed.init_process_group(backend="nccl", init_method="env://")` if you need lower-level control. The helper provides the same environment variables you would otherwise set by hand so that PyTorch’s `env://` init works. This pattern is used in Submitit’s own distributed examples and in downstream guides.

## Distributed environment: tasks, ranks, and GPUs (with Submitit on Slurm)

### Tasks-per-node and GPUs-per-node
- One process per GPU is the common pattern. Concretely:
  `hydra.launcher.tasks_per_node = compute.gpus_per_node`
  This makes Slurm/Submitit spawn exactly one task per GPU. Each task becomes a rank in the job.

### What Submitit exports
- `submitit.helpers.TorchDistributedEnvironment().export()` populates:
  - `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` (and related fields) so that `init_method="env://"` works out of the box.

### Binding each task to one GPU
- Slurm’s GRES plugin sets `CUDA_VISIBLE_DEVICES` for each task so the task “sees” only its assigned GPU(s). You can additionally enforce a 1:1 mapping with:
  ```yaml
  hydra.launcher.setup:
    - "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID"
  ```
  This ensures rank-local GPU selection is unambiguous (task 0 -> GPU 0, task 1 -> GPU 1 on that node).

### Quick glossary
- **WORLD_SIZE**: total number of processes across all nodes.
- **RANK**: global process id `[0 .. WORLD_SIZE-1]`.
- **LOCAL_RANK**: process id per node `[0 .. tasks_per_node-1]`.

## Why not torchrun here?

`torchrun` is valid for distributed launches (including on Slurm), but this template uses **Hydra’s Submitit launcher** to keep **sweeps, config composition, logging, and requeue** inside Hydra, and to avoid maintaining separate bash wrappers. Submitit handles **job submission and per-task rank context**; we still initialize PyTorch distributed via the standard env-var pathway.

If you prefer `torchrun`, you can adapt the script and configs—but you’ll then manage the Slurm submission layer (or wrap `torchrun` inside an `sbatch` yourself) and wire up Hydra sweeps accordingly.

## References

- PyTorch distributed environment variables: https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
- Slurm GRES guide: https://slurm.schedmd.com/gres.html
- Hugging Face FSDP / Trainer documentation: https://huggingface.co/docs/transformers/fsdp
