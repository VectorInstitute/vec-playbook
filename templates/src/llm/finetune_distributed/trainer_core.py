"""Core training utilities for the distributed fine-tuning template."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def is_main_process() -> bool:
    """Return ``True`` when running on the primary rank."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def is_fsdp(model: torch.nn.Module) -> bool:
    """Return ``True`` when the model is wrapped in ``FullyShardedDataParallel``."""
    return isinstance(model, FullyShardedDataParallel)


class TrainerState:
    """Track progress across epochs and global steps."""

    def __init__(self) -> None:
        self.epoch: int = 0
        self.global_step: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialize the state so it can be checkpointed."""
        return {"epoch": self.epoch, "global_step": self.global_step}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "TrainerState":
        """Create a state object from serialized metadata."""
        state = cls()
        state.epoch = data.get("epoch", 0)
        state.global_step = data.get("global_step", 0)
        return state


class Trainer:
    """Minimal trainer inspired by VectorLM's implementation."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        work_dir: str,
        gradient_accumulation_steps: int,
        logging_steps: int,
        eval_steps: int,
        save_steps: int,
        max_epochs: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.work_dir = work_dir
        self.grad_accum = max(gradient_accumulation_steps, 1)
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_epochs = max_epochs

        self.state = TrainerState()
        self._log_cache: list[float] = []

        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _model_state_dict(self) -> dict:
        """Return a model state dict that respects FSDP wrapping."""
        if is_fsdp(self.model):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FullyShardedDataParallel.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, cfg
            ):
                return self.model.state_dict()
        return self.model.state_dict()

    def _load_model_state_dict(self, state: dict) -> None:
        """Restore parameters into an optionally sharded model."""
        if is_fsdp(self.model):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FullyShardedDataParallel.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, cfg
            ):
                self.model.load_state_dict(state)
            return
        self.model.load_state_dict(state)

    def latest_checkpoint(self) -> Optional[str]:
        """Return the newest checkpoint directory, if it exists."""
        if not os.path.exists(self.checkpoint_dir):
            return None
        ckpts = [
            name for name in os.listdir(self.checkpoint_dir) if name.startswith("step=")
        ]
        if not ckpts:
            return None
        ckpts.sort(key=lambda name: int(name.split("=")[-1]))
        return os.path.join(self.checkpoint_dir, ckpts[-1])

    def save_checkpoint(self) -> None:
        """Persist model/optimizer/scheduler state to disk."""
        if not is_main_process():
            return
        path = os.path.join(self.checkpoint_dir, f"step={self.state.global_step}")
        os.makedirs(path, exist_ok=True)
        torch.save(self._model_state_dict(), os.path.join(path, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        torch.save(self.state.to_dict(), os.path.join(path, "trainer_state.pt"))

    def load_checkpoint(self, ckpt_path: str) -> None:
        """Reload a previously saved checkpoint."""
        map_location = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        model_state = torch.load(
            os.path.join(ckpt_path, "model.pt"), map_location=map_location
        )
        self._load_model_state_dict(model_state)

        optim_state = torch.load(
            os.path.join(ckpt_path, "optimizer.pt"), map_location=map_location
        )
        self.optimizer.load_state_dict(optim_state)

        sched_path = os.path.join(ckpt_path, "scheduler.pt")
        if self.scheduler is not None and os.path.exists(sched_path):
            sched_state = torch.load(sched_path, map_location=map_location)
            self.scheduler.load_state_dict(sched_state)

        state_path = os.path.join(ckpt_path, "trainer_state.pt")
        if os.path.exists(state_path):
            self.state = TrainerState.from_dict(torch.load(state_path))

    def train(self) -> None:
        """Run the main training loop."""
        if is_main_process():
            print(f"Starting training for {self.max_epochs} epochs")

        for epoch in range(self.state.epoch, self.max_epochs):
            self.state.epoch = epoch
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            self._train_one_epoch()

            should_eval = (
                self.eval_loader is not None
                and self.eval_steps > 0
                and self.state.global_step % self.eval_steps == 0
            )
            if should_eval:
                self.evaluate(epoch)

            if self.state.global_step >= len(self.train_loader) * self.max_epochs:
                break

        if is_main_process():
            print("Training finished")

    def _train_one_epoch(self) -> None:
        """Train the model for a single epoch."""
        self.model.train()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        for step, batch in enumerate(self.train_loader, start=1):
            batch_on_device = {key: value.to(device) for key, value in batch.items()}
            outputs = self.model(**batch_on_device)
            loss = outputs.loss / self.grad_accum
            loss.backward()

            gathered_loss = loss.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(gathered_loss, op=dist.ReduceOp.SUM)
                gathered_loss /= dist.get_world_size()
            self._log_cache.append(gathered_loss.item())

            if step % self.grad_accum == 0:
                if hasattr(self.model, "clip_grad_norm_"):
                    self.model.clip_grad_norm_(max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1

                should_log = (
                    self.logging_steps
                    and self.state.global_step % self.logging_steps == 0
                    and is_main_process()
                )
                if should_log:
                    mean_loss = sum(self._log_cache) / max(len(self._log_cache), 1)
                    print(f"step={self.state.global_step} loss={mean_loss:.4f}")
                    self._log_cache.clear()

                if self.save_steps and self.state.global_step % self.save_steps == 0:
                    if dist.is_initialized():
                        dist.barrier()
                    self.save_checkpoint()
                    if dist.is_initialized():
                        dist.barrier()

    def evaluate(self, epoch: int) -> None:
        """Run an evaluation epoch and report mean loss."""
        if self.eval_loader is None:
            return
        self.model.eval()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        losses = []
        with torch.no_grad():
            for batch in self.eval_loader:
                batch_on_device = {
                    key: value.to(device) for key, value in batch.items()
                }
                outputs = self.model(**batch_on_device)
                losses.append(outputs.loss.detach())
        if not losses:
            return
        loss_tensor = torch.stack(losses)
        if dist.is_initialized():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= dist.get_world_size()
        mean_loss = loss_tensor.mean().item()
        if is_main_process():
            print(f"epoch={epoch} eval_loss={mean_loss:.4f}")
        self.model.train()


__all__ = ["Trainer", "TrainerState", "is_main_process"]
