"""Distributed MLP training using PyTorch DDP."""

import os

import submitit
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def create_dummy_data(
    num_samples: int = 1000, input_dim: int = 10, num_classes: int = 3
):
    """Create a dummy classification dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


class DDPMLPTrainer(submitit.helpers.Checkpointable):
    """Distributed MLP trainer using PyTorch DDP."""

    def __init__(self):
        """Initialize the trainer."""
        self.ckpt_dir = None

    def _latest_checkpoint(self, out_dir):
        """Find the latest checkpoint directory."""
        if not os.path.exists(out_dir):
            return None
        names = [n for n in os.listdir(out_dir) if n.startswith("checkpoint-epoch-")]
        if not names:
            return None
        epochs = sorted(
            int(n.split("-")[-1]) for n in names if n.split("-")[-1].isdigit()
        )
        return (
            os.path.join(out_dir, f"checkpoint-epoch-{epochs[-1]}") if epochs else None
        )

    def _save_checkpoint(self, model, optimizer, epoch, out_dir, loss, accuracy, rank):
        """Save model checkpoint (only on rank 0)."""
        if rank != 0:
            return

        save_dir = os.path.join(out_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        # Save the actual model (unwrap DDP)
        model_state = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        }

        torch.save(checkpoint, os.path.join(save_dir, "model.pt"))
        print(f"Checkpoint saved at epoch {epoch}")

    def _setup_distributed(self, rank, world_size):
        """Initialize distributed training."""
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size,
            )

    def _initialize_device_and_model(self, cfg, local_rank):
        """Initialize device and model."""
        input_dim = getattr(cfg, "input_dim", 10)
        hidden_dim = getattr(cfg, "hidden_dim", 64)
        num_classes = getattr(cfg, "num_classes", 3)

        # Setup device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        # Setup model
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        ).to(device)

        return device, model

    def _initialize_data_and_loader(self, cfg, world_size, rank):
        """Initialize dataset and dataloader with distributed sampler."""
        input_dim = getattr(cfg, "input_dim", 10)
        num_classes = getattr(cfg, "num_classes", 3)
        batch_size = getattr(cfg, "batch_size", 32)

        dataset = create_dummy_data(1000, input_dim, num_classes)
        sampler = (
            DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
            if world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
        )
        return loader, sampler

    def _load_checkpoint_if_exists(self, model, optimizer, device, rank):
        """Load checkpoint if it exists and return start epoch."""
        start_epoch = 0
        if self.ckpt_dir and os.path.exists(self.ckpt_dir):
            checkpoint_path = os.path.join(self.ckpt_dir, "model.pt")
            if os.path.exists(checkpoint_path):
                if rank == 0:
                    print(f"Resuming from checkpoint: {self.ckpt_dir}")
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # Load model state (handle DDP wrapper)
                if hasattr(model, "module"):
                    model.module.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint["model_state_dict"])

                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                if rank == 0:
                    print(f"Resumed from epoch {checkpoint['epoch']}")
        return start_epoch

    def _train_epoch(
        self,
        model,
        optimizer,
        criterion,
        loader,
        sampler,
        device,
        epoch,
        world_size,
        rank,
    ):
        """Train for one epoch and return metrics."""
        # Set epoch for DistributedSampler to ensure proper shuffling across epochs
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for batch_x, batch_y in loader:
            x = batch_x.to(device)
            y = batch_y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

        # Aggregate metrics across all processes
        if world_size > 1:
            metrics = torch.tensor(
                [loss_sum, correct, total], device=device, dtype=torch.float32
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            loss_sum, correct, total = metrics.tolist()

        return loss_sum, correct, total

    def __call__(self, cfg):
        """Train the MLP model with DDP."""
        out_dir = os.path.join(cfg.work_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        self.ckpt_dir = self._latest_checkpoint(out_dir)

        # Configuration
        lr = getattr(cfg, "learning_rate", 1e-3)
        num_epochs = getattr(cfg, "num_epochs", 1000)
        seed = getattr(cfg, "seed", 42)

        # Get distributed training info from environment
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if rank == 0:
            print(f"Starting DDP MLP training with seed {seed}")
            print(f"World size: {world_size}, Local rank: {local_rank}")

        # Set seed for reproducibility (same seed on all processes)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Setup distributed training
        self._setup_distributed(rank, world_size)

        # Setup device and model
        device, model = self._initialize_device_and_model(cfg, local_rank)

        if rank == 0:
            print(f"Using device: {device}")

        # Wrap model with DDP
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
            )

        # Setup data and training
        loader, sampler = self._initialize_data_and_loader(cfg, world_size, rank)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Resume from checkpoint if available
        start_epoch = self._load_checkpoint_if_exists(model, optimizer, device, rank)

        if rank == 0:
            print(f"Training from epoch {start_epoch} to {num_epochs}...")

        # Training loop with DDP
        for epoch in range(start_epoch, num_epochs):
            loss_sum, correct, total = self._train_epoch(
                model,
                optimizer,
                criterion,
                loader,
                sampler,
                device,
                epoch,
                world_size,
                rank,
            )

            # Print metrics only on rank 0
            if rank == 0:
                acc = 100.0 * correct / total
                avg_loss = loss_sum / len(loader)
                print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.2f}%")

                if epoch % 100 == 0 or epoch == num_epochs - 1:
                    if world_size > 1:
                        dist.barrier()
                    self._save_checkpoint(
                        model, optimizer, epoch, out_dir, avg_loss, acc, rank
                    )

        if rank == 0:
            print("Training completed!")

        # Clean up distributed training
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

        return 0

    def checkpoint(self, *args, **kwargs):
        """Checkpoint the trainer."""
        # Model checkpoints are already saved in _save_checkpoint.
        # Returning a DelayedSubmission will requeue the same callable.
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
