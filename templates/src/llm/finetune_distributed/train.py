"""Distributed finetuning worker script."""

from __future__ import annotations

import argparse
import math
import os
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import set_seed

from .trainer_core import Trainer, is_main_process


@dataclass
class RuntimeConfig:
    """Resolved runtime configuration used by the trainer."""

    work_dir: str
    grad_accum: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    max_length: int
    dataset_name: str
    text_column: str
    model_name: str
    trust_remote_code: bool
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    dist_mode: str
    dist_backend: str
    seed: int


def _prepare_datasets(
    cfg: RuntimeConfig, tokenizer
) -> tuple[DataLoader, Optional[DataLoader]]:
    raw = load_dataset(cfg.dataset_name, trust_remote_code=cfg.trust_remote_code)
    train_split = raw["train"]
    eval_split = raw.get("validation") or raw.get("test")

    def tokenize_fn(batch):
        return tokenizer(
            batch[cfg.text_column],
            truncation=True,
            max_length=cfg.max_length,
            return_special_tokens_mask=True,
        )

    tokenized_train = train_split.map(
        tokenize_fn, batched=True, remove_columns=train_split.column_names
    )

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = len(concatenated)
        if total_length >= cfg.max_length:
            total_length = (total_length // cfg.max_length) * cfg.max_length
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}
        return {
            "input_ids": [
                concatenated[i : i + cfg.max_length]
                for i in range(0, total_length, cfg.max_length)
            ],
            "attention_mask": [
                [1] * cfg.max_length for _ in range(0, total_length, cfg.max_length)
            ],
        }

    tokenized_train = tokenized_train.map(group_texts, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_sampler = (
        DistributedSampler(tokenized_train, shuffle=True)
        if dist.is_initialized()
        else None
    )
    train_loader = DataLoader(
        tokenized_train,
        batch_size=cfg.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collator,
    )

    eval_loader = None
    if eval_split is not None:
        tokenized_eval = eval_split.map(
            tokenize_fn, batched=True, remove_columns=eval_split.column_names
        )
        tokenized_eval = tokenized_eval.map(group_texts, batched=True)
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_sampler = (
            DistributedSampler(tokenized_eval, shuffle=False)
            if dist.is_initialized()
            else None
        )
        eval_loader = DataLoader(
            tokenized_eval,
            batch_size=cfg.per_device_eval_batch_size,
            sampler=eval_sampler,
            shuffle=False,
            collate_fn=collator,
        )

    return train_loader, eval_loader


def _choose_dtype(cfg: RuntimeConfig) -> torch.dtype:
    if cfg.bf16 and torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if is_main_process():
            msg = "BF16 not supported on this GPU; using {} instead".format(
                "FP16" if cfg.fp16 else "FP32"
            )
            warnings.warn(msg, stacklevel=2)
    if cfg.fp16:
        return torch.float16
    return torch.float32


def _build_model(cfg: RuntimeConfig, device: torch.device) -> torch.nn.Module:
    torch_dtype = _choose_dtype(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch_dtype, trust_remote_code=cfg.trust_remote_code
    )
    model.to(device)
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _wrap_model(
    cfg: RuntimeConfig, model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    if cfg.dist_mode == "fsdp":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=1_000_000
        )
        return FullyShardedDataParallel(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=True,
            sync_module_states=True,
        )
    if cfg.dist_mode == "ddp":
        return DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    return model


def _build_optimizer(cfg: RuntimeConfig, model: torch.nn.Module) -> AdamW:
    return AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )


def _build_scheduler(
    cfg: RuntimeConfig, optimizer: AdamW, train_loader: DataLoader
) -> LambdaLR:
    total_update_steps = (
        math.ceil(len(train_loader) / cfg.grad_accum) * cfg.num_train_epochs
    )
    warmup_steps = int(cfg.warmup_ratio * total_update_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_update_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def _maybe_resume(trainer: Trainer, resume_path: Optional[str]) -> None:
    ckpt = resume_path or trainer.latest_checkpoint()
    if ckpt and os.path.exists(ckpt):
        if dist.is_initialized():
            dist.barrier()
        trainer.load_checkpoint(ckpt)
        if is_main_process():
            print(f"Resumed from checkpoint: {ckpt}")
        if dist.is_initialized():
            dist.barrier()


def run(cfg: RuntimeConfig, raw_cfg) -> None:
    """Entry point used by torchrun workers."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if (
        dist.is_available()
        and not dist.is_initialized()
        and cfg.dist_mode in {"ddp", "fsdp"}
    ):
        dist.init_process_group(backend=cfg.dist_backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    set_seed(cfg.seed + dist.get_rank() if dist.is_initialized() else cfg.seed)

    os.makedirs(cfg.work_dir, exist_ok=True)
    if is_main_process():
        OmegaConf.save(raw_cfg, os.path.join(cfg.work_dir, "resolved_config.yaml"))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, eval_loader = _prepare_datasets(cfg, tokenizer)

    model = _build_model(cfg, device)
    model = _wrap_model(cfg, model, device)

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer, train_loader)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        eval_loader=eval_loader,
        work_dir=cfg.work_dir,
        gradient_accumulation_steps=cfg.grad_accum,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        max_epochs=cfg.num_train_epochs,
    )

    resume_path = None
    if isinstance(raw_cfg, DictConfig) and "resume_from_checkpoint" in raw_cfg:
        resume_path = raw_cfg.resume_from_checkpoint
    elif isinstance(raw_cfg, dict) and "resume_from_checkpoint" in raw_cfg:
        resume_path = raw_cfg.get("resume_from_checkpoint")
    elif hasattr(raw_cfg, "resume_from_checkpoint"):
        resume_path = raw_cfg.resume_from_checkpoint
    _maybe_resume(trainer, resume_path)

    trainer.train()

    if is_main_process():
        trainer.save_checkpoint()

    if dist.is_initialized() and cfg.dist_mode in {"ddp", "fsdp"}:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments when invoked directly."""
    parser = argparse.ArgumentParser(description="Distributed finetuning trainer")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to resolved Hydra config"
    )
    return parser.parse_args()


def build_runtime_config(cfg) -> RuntimeConfig:
    """Build the dataclass used by the trainer from a resolved config."""
    return RuntimeConfig(
        work_dir=cfg.work_dir,
        grad_accum=cfg.train.gradient_accumulation_steps,
        logging_steps=cfg.train.logging_steps,
        eval_steps=cfg.train.eval_steps,
        save_steps=cfg.train.save_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=getattr(cfg.train, "warmup_ratio", 0.03),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        max_length=cfg.data.max_length,
        dataset_name=cfg.data.name,
        text_column=cfg.data.text_key,
        model_name=cfg.model.name,
        trust_remote_code=getattr(cfg.data, "trust_remote_code", False),
        bf16=getattr(cfg.dist, "bf16", False),
        fp16=getattr(cfg.dist, "fp16", False),
        gradient_checkpointing=getattr(cfg.model, "gradient_checkpointing", False),
        dist_mode=getattr(cfg.dist, "mode", "none"),
        dist_backend=getattr(cfg.dist, "backend", "nccl"),
        seed=getattr(cfg.train, "seed", 42),
    )


def main() -> None:
    """CLI entrypoint executed by torchrun workers."""
    args = parse_args()
    raw_cfg = OmegaConf.load(args.config)
    runtime_cfg = build_runtime_config(raw_cfg)
    run(runtime_cfg, raw_cfg)


if __name__ == "__main__":
    main()
