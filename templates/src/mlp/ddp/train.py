"""Distributed MLP training using PyTorch DDP."""

import logging
import os

import submitit
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


logger = logging.getLogger(__name__)


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
        logger.info(f"Checkpoint saved at epoch {epoch}")

    def _setup_distributed(self, rank, world_size):
        """Initialize distributed training."""
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size,
            )

    def _wrap_distributed(self, model, world_size, local_rank):
        if world_size > 1:
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
            )
        return model

    def _configure_training(self, cfg):
        lr = OmegaConf.select(cfg, "trainer.learning_rate", default=1e-3)
        num_epochs = OmegaConf.select(cfg, "trainer.num_epochs", default=1000)
        seed = OmegaConf.select(cfg, "trainer.seed", default=42)
        return lr, num_epochs, seed

    def _get_distributed_config(self):
        job_env = submitit.JobEnvironment()
        return job_env, job_env.global_rank, job_env.local_rank, job_env.num_tasks

    def _prepare_environment(self, job_env, rank, local_rank, world_size):
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))

        if "MASTER_ADDR" not in os.environ:
            hostnames = getattr(job_env, "hostnames", None) or [job_env.hostname]
            os.environ["MASTER_ADDR"] = str(hostnames[0])

        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

    def _log_run_configuration(self, seed, world_size, local_rank, rank):
        if rank != 0:
            return
        logger.info(f"Starting DDP MLP training with seed {seed}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        if torch.cuda.is_available():
            logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _initialize_device_and_model(self, cfg, local_rank):
        """Initialize device and model."""
        input_dim = OmegaConf.select(cfg, "trainer.input_dim", default=10)
        hidden_dim = OmegaConf.select(cfg, "trainer.hidden_dim", default=64)
        num_classes = OmegaConf.select(cfg, "trainer.num_classes", default=3)

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
        input_dim = OmegaConf.select(cfg, "trainer.input_dim", default=10)
        num_classes = OmegaConf.select(cfg, "trainer.num_classes", default=3)
        batch_size = OmegaConf.select(cfg, "trainer.batch_size", default=32)

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
                    logger.info(f"Resuming from checkpoint: {self.ckpt_dir}")
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # Load model state (handle DDP wrapper)
                if hasattr(model, "module"):
                    model.module.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint["model_state_dict"])

                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                if rank == 0:
                    logger.info(f"Resumed from epoch {checkpoint['epoch']}")
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
        cfg: DictConfig = OmegaConf.create(cfg)

        out_dir = cfg.paths.out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.ckpt_dir = self._latest_checkpoint(out_dir)

        lr, num_epochs, seed = self._configure_training(cfg)
        job_env, rank, local_rank, world_size = self._get_distributed_config()

        self._prepare_environment(job_env, rank, local_rank, world_size)
        self._set_seed(seed)
        self._log_run_configuration(seed, world_size, local_rank, rank)

        self._setup_distributed(rank, world_size)

        device, model = self._initialize_device_and_model(cfg, local_rank)
        if rank == 0:
            logger.info(f"Using device: {device}")

        model = self._wrap_distributed(model, world_size, local_rank)

        loader, sampler = self._initialize_data_and_loader(cfg, world_size, rank)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        start_epoch = self._load_checkpoint_if_exists(model, optimizer, device, rank)
        if rank == 0:
            logger.info(f"Training from epoch {start_epoch} to {num_epochs}...")

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

            avg_loss = loss_sum / len(loader)
            acc = 100.0 * correct / total
            should_checkpoint = epoch % 100 == 0 or epoch == num_epochs - 1

            # Log metrics only on rank 0
            if rank == 0:
                logger.info(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.2f}%")

            if should_checkpoint:
                if world_size > 1:
                    dist.barrier()
                if rank == 0:
                    self._save_checkpoint(
                        model, optimizer, epoch, out_dir, avg_loss, acc, rank
                    )

        if rank == 0:
            logger.info("Training completed!")

        # Clean up distributed training
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

        return 0

    def checkpoint(self, *args, **kwargs):
        """Checkpoint the trainer."""
        # Model checkpoints are already saved in _save_checkpoint.
        # Returning a DelayedSubmission will requeue the same callable.
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
