"""Simple single-GPU MLP training (no checkpointing)."""

import logging

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


logger = logging.getLogger(__name__)


def create_dummy_data(
    num_samples: int = 1000, input_dim: int = 10, num_classes: int = 3
):
    """Create a dummy classification dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


class SimpleMLPTrainer:
    """Simple MLP trainer without checkpointing."""

    def __init__(self):
        """Initialize the trainer."""
        pass

    def __call__(self, cfg):
        """Train the MLP model."""
        cfg: DictConfig = OmegaConf.create(cfg)  # Ensure cfg is a DictConfig

        # Get trainer config variables
        input_dim = OmegaConf.select(cfg, "trainer.input_dim", default=10)
        hidden_dim = OmegaConf.select(cfg, "trainer.hidden_dim", default=64)
        num_classes = OmegaConf.select(cfg, "trainer.num_classes", default=3)
        batch_size = OmegaConf.select(cfg, "trainer.batch_size", default=32)
        lr = OmegaConf.select(cfg, "trainer.learning_rate", default=1e-3)
        num_epochs = OmegaConf.select(cfg, "trainer.num_epochs", default=1000)
        seed = OmegaConf.select(cfg, "trainer.seed", default=42)

        logger.info(f"Starting simple MLP training with seed {seed}")
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

        logger.info(f"Training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
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

        logger.info("Training completed!")
        return 0
