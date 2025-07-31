"""Distributed (multi-node) example trainer using PyTorch DDP."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from vec_tool.core.checkpointable import Checkpointable


def create_dummy_data(
    num_samples: int = 1000, input_dim: int = 10, num_classes: int = 3
):
    """Create a dummy classification dataset."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


class DDPTrainer(Checkpointable):
    """Distributed PyTorch trainer (multi-node DDP) with checkpointing."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        num_classes: int = 3,
        batch_size: int = 32,
        lr: float = 1e-5,
        num_epochs: int = 10000,
        **cfg: Any,
    ) -> None:
        super().__init__(**cfg)

        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.lr = float(lr)

        self.rank = self.dist_env.rank
        self.local_rank = self.dist_env.local_rank
        self.world_size = self.dist_env.world_size

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=self.rank,
                world_size=self.world_size,
            )

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        ).to(self.device)

        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            )

        dataset = create_dummy_data(1000, self.input_dim, self.num_classes)
        self.sampler = (
            DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
            if self.world_size > 1
            else None
        )
        self.loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=self.sampler is None,
        )

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.crit = nn.CrossEntropyLoss()

    def __call__(self) -> None:
        """Run the distributed training loop and resume if checkpoint exists."""
        start = self._load_step()
        for epoch in range(start, self.num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            self._train_epoch(epoch)
            if epoch % 100 == 0:
                self.save_checkpoint(
                    f"epoch_{epoch}.pt", step=epoch, model=self.model.state_dict()
                )
        self._save_step(self.num_epochs)

    def _train_epoch(self, epoch: int) -> None:
        """Train model for one epoch and print aggregated metrics."""
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

        if self.world_size > 1:
            tensor = torch.tensor(
                [loss_sum, correct, total], device=self.device, dtype=torch.float32
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            loss_sum, correct, total = tensor.tolist()

        if self.rank == 0:
            acc = 100.0 * correct / total
            print(f"Epoch {epoch}: loss={loss_sum / total:.4f} acc={acc:.2f}%")

    def __del__(self):
        """Destroy the distributed process group on object cleanup."""
        if dist.is_initialized():
            dist.destroy_process_group()
