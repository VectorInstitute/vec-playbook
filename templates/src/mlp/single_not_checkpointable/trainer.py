"""Simple single-GPU training example for vec-tool (no checkpointing)."""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def create_dummy_data(
    num_samples: int = 1000, input_dim: int = 10, num_classes: int = 3
):
    """Create a dummy classification dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


class SimpleTrainerNoCheckpoint:
    """Simple feed-forward trainer without checkpointing."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_classes: int = 3,
        batch_size: int = 32,
        lr: float = 1e-5,
        num_epochs: int = 100,  # Reduced default for non-checkpointable example
        **cfg,
    ):
        # Store configuration parameters
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.lr = float(lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        ).to(self.device)

        self.loader = DataLoader(
            create_dummy_data(1000, self.input_dim, self.num_classes),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.crit = nn.CrossEntropyLoss()

    def __call__(self) -> None:
        """Run the complete training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)
        print("Training completed!")

    def _train_epoch(self, epoch: int) -> None:
        """Train model for one epoch and print metrics."""
        self.model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for batch_x, batch_y in self.loader:
            x = batch_x.to(self.device)
            y = batch_y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            loss = self.crit(out, y)
            loss.backward()
            self.opt.step()

            loss_sum += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch {epoch}: loss={loss_sum / len(self.loader):.4f} acc={acc:.2f}%")
