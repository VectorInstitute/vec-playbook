"""Single-GPU MLP training with checkpointing."""

import os
import logging

import submitit
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


def create_dummy_data(
    num_samples: int = 1000, input_dim: int = 10, num_classes: int = 3
):
    """Create a dummy classification dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


class CheckpointableMLPTrainer(submitit.helpers.Checkpointable):
    """MLP trainer with checkpointing support."""

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

    def _save_checkpoint(self, model, optimizer, epoch, out_dir, loss, accuracy):
        """Save model checkpoint."""
        save_dir = os.path.join(out_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        }

        torch.save(checkpoint, os.path.join(save_dir, "model.pt"))
        logger.info(f"Checkpoint saved at epoch {epoch}")

    def __call__(self, cfg):
        """Train the MLP model."""
        cfg : DictConfig = OmegaConf.create(cfg)  # Ensure cfg is a DictConfig

        # Create output directory
        out_dir = cfg.paths.out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Get ckpt dir
        self.ckpt_dir = self._latest_checkpoint(out_dir)

        # Get trainer config variables
        input_dim = OmegaConf.select(cfg, "trainer.input_dim", default=10)
        hidden_dim = OmegaConf.select(cfg, "trainer.hidden_dim", default=64)
        num_classes = OmegaConf.select(cfg, "trainer.num_classes", default=3)
        batch_size = OmegaConf.select(cfg, "trainer.batch_size", default=32)
        lr = OmegaConf.select(cfg, "trainer.learning_rate", default=1e-3)
        num_epochs = OmegaConf.select(cfg, "trainer.num_epochs", default=1000)
        seed = OmegaConf.select(cfg, "trainer.seed", default=42)

        logger.info(f"Starting checkpointable MLP training with seed {seed}")
        torch.manual_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        ).to(device)

        dataset = create_dummy_data(1000, input_dim, num_classes)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        if self.ckpt_dir and os.path.exists(self.ckpt_dir):
            checkpoint_path = os.path.join(self.ckpt_dir, "model.pt")
            if os.path.exists(checkpoint_path):
                logger.info(f"Resuming from checkpoint: {self.ckpt_dir}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                logger.info(f"Resumed from epoch {checkpoint['epoch']}")

        logger.info(f"Training from epoch {start_epoch} to {num_epochs}...")

        # Training loop with checkpointing
        for epoch in range(start_epoch, num_epochs):
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

            acc = 100.0 * correct / total
            avg_loss = loss_sum / len(loader)
            logger.info(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.2f}%")

            # Save checkpoint every 100 epochs
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                self._save_checkpoint(model, optimizer, epoch, out_dir, avg_loss, acc)

        logger.info("Training completed!")
        return 0

    def checkpoint(self, *args, **kwargs):
        """Checkpoint the trainer."""
        # Model checkpoints are already saved in _save_checkpoint.
        # Returning a DelayedSubmission will requeue the same callable.
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
