"""Toy feed-forward trainers (single-GPU and DDP) for quick vec-tool launches."""

from .ddp.trainer import DDPTrainer
from .single.trainer import SimpleTrainer


__all__ = ["SimpleTrainer", "DDPTrainer"]
