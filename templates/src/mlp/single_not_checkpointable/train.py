"""Simple single-GPU MLP training (no checkpointing)."""

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


class SimpleMLPTrainer:
    """Simple MLP trainer without checkpointing."""

    def __init__(self):
        """Initialize the trainer."""
        pass

    def __call__(self, cfg):
        """Train the MLP model."""
        input_dim = getattr(cfg, "input_dim", 10)
        hidden_dim = getattr(cfg, "hidden_dim", 64)
        num_classes = getattr(cfg, "num_classes", 3)
        batch_size = getattr(cfg, "batch_size", 32)
        lr = getattr(cfg, "learning_rate", 1e-3)
        num_epochs = getattr(cfg, "num_epochs", 100)
        seed = getattr(cfg, "seed", 42)

        print(f"Starting simple MLP training with seed {seed}")
        torch.manual_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

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

        print(f"Training for {num_epochs} epochs...")

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
            print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.2f}%")

        print("Training completed!")
        return 0
