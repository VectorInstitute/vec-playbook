"""Distributed LLM fine-tuning implemented with Hydra + Submitit."""

from __future__ import annotations

import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import submitit
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging


logger = logging.getLogger(__name__)


class RankZeroLoggingCallback(TrainerCallback):
    """Log training progress to Hydra logs on rank 0 only."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Forward trainer logs to Python logger on rank 0."""
        if state.is_world_process_zero and logs:
            filtered = {
                k: v
                for k, v in logs.items()
                if k in ["loss", "learning_rate", "epoch", "grad_norm", "eval_loss"]
            }
            if filtered:
                logger.info(f"Step {state.global_step}: {filtered}")


class FinetuneDistributedTrainer(submitit.helpers.Checkpointable):
    """Trainer that fine-tunes a causal LM with HF Trainer in distributed mode."""

    def __init__(self):
        """Initialize trainer state for submitit checkpointing."""
        self.ckpt_dir: Optional[str] = None

    def _latest_checkpoint(self, out_dir: Path) -> Optional[str]:
        """Return latest checkpoint path in out_dir or None if absent."""
        if not out_dir.exists():
            return None
        candidates = sorted(
            [p for p in out_dir.iterdir() if p.name.startswith("checkpoint-")],
            key=lambda p: p.stat().st_mtime,
        )
        return str(candidates[-1]) if candidates else None

    def _resolve_dtype(self, model_cfg: Dict[str, Any]) -> torch.dtype:
        """Convert required torch_dtype string into a torch.dtype."""
        dtype = model_cfg.get("torch_dtype")
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported torch_dtype '{dtype}'")
        return dtype_map[dtype]

    def _setup_distributed_environment(self) -> Tuple[int, int]:
        """Export distributed env (ranks/world size) and set CUDA device."""
        submitit.helpers.TorchDistributedEnvironment().export()

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if rank == 0:
            logging.getLogger().setLevel(logging.INFO)
            hf_logging.set_verbosity_info()
        else:
            logging.getLogger().setLevel(logging.WARNING)
            hf_logging.set_verbosity_error()

        logging.getLogger("datasets").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", category=FutureWarning)

        if torch.cuda.is_available():
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                # Each process sees only its own GPU -> index 0 is always correct
                torch.cuda.set_device(0)
                current_device = 0
                num_visible = torch.cuda.device_count()  # typically 1 after masking
            else:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                torch.cuda.set_device(local_rank)
                current_device = local_rank
                num_visible = torch.cuda.device_count()

            if rank == 0:
                logger.info(
                    "Distributed context initialized: rank=%s/%s, local_rank=%s, pid=%s, "
                    "num_visible_gpus=%s, current_device=%s",
                    os.environ.get("RANK", "?"),
                    os.environ.get("WORLD_SIZE", "?"),
                    os.environ.get("LOCAL_RANK", "?"),
                    os.getpid(),
                    num_visible,
                    current_device,
                )
            else:
                # keep a minimal signal on non-zero ranks for debugging
                print(f"[Rank {rank}] Using CUDA device {current_device}", flush=True)
        elif rank == 0:
            logger.info("CUDA not available in this process (CPU mode).")

        return rank, world_size

    def _prepare_dataset(
        self, cfg: DictConfig, tokenizer, rank
    ) -> Tuple[Dataset, Dataset]:
        """Tokenize and chunk dataset splits for causal LM training."""
        data_cfg = cfg.trainer.data
        dataset_kwargs = OmegaConf.to_container(data_cfg.load_kwargs) or {}
        dataset = load_dataset(
            data_cfg.dataset_name,
            data_cfg.get("dataset_config_name"),
            **dataset_kwargs,
        )

        train_dataset: Dataset = dataset[data_cfg.train_split]
        eval_dataset: Dataset = dataset[data_cfg.eval_split]
        text_column = data_cfg.text_column

        block_size = int(
            min(
                data_cfg.max_length,
                getattr(tokenizer, "model_max_length", data_cfg.max_length),
            )
        )
        remove_columns = list(train_dataset.column_names)

        tokenized_train = train_dataset.map(
            lambda batch: tokenizer(
                batch[text_column], truncation=True, max_length=block_size
            ),
            batched=True,
            remove_columns=remove_columns,
        )
        tokenized_eval = eval_dataset.map(
            lambda batch: tokenizer(
                batch[text_column], truncation=True, max_length=block_size
            ),
            batched=True,
            remove_columns=remove_columns,
        )

        def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
            """Flatten token lists and chunk them into fixed block_size segments."""
            result: Dict[str, Any] = {}
            for key, sequences in examples.items():
                concatenated = [tok for seq in sequences for tok in seq]
                total_length = (len(concatenated) // block_size) * block_size
                if total_length == 0:
                    # no full block, return empty batch for this shard
                    return {k: [] for k in examples}
                result[key] = [
                    concatenated[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
            return result

        train_blocks = tokenized_train.map(group_texts, batched=True)
        eval_blocks = tokenized_eval.map(group_texts, batched=True)

        if rank == 0:
            logger.info(
                "Prepared datasets with %d train blocks and %d eval blocks (block size %d)",
                len(train_blocks),
                len(eval_blocks),
                block_size,
            )

        return train_blocks, eval_blocks

    def _build_training_arguments(
        self, cfg: DictConfig, out_dir: Path, rank: int
    ) -> TrainingArguments:
        """Construct TrainingArguments from resolved Hydra config."""
        train_cfg = cfg.trainer.train
        dist_cfg = cfg.trainer.dist
        logging_cfg = cfg.trainer.logging

        eval_strategy = train_cfg.eval_strategy

        kwargs: Dict[str, Any] = {
            "output_dir": str(out_dir),
            "overwrite_output_dir": True,
            "num_train_epochs": float(train_cfg.num_train_epochs),
            "per_device_train_batch_size": int(train_cfg.per_device_train_batch_size),
            "per_device_eval_batch_size": int(train_cfg.per_device_eval_batch_size),
            "gradient_accumulation_steps": int(train_cfg.gradient_accumulation_steps),
            "learning_rate": float(train_cfg.learning_rate),
            "weight_decay": float(train_cfg.weight_decay),
            "warmup_steps": int(train_cfg.warmup_steps),
            "logging_strategy": "steps",
            "logging_steps": int(train_cfg.logging_steps),
            "disable_tqdm": rank != 0,
            "eval_strategy": eval_strategy,
            "eval_steps": int(train_cfg.eval_steps),
            "save_strategy": train_cfg.save_strategy,
            "save_steps": int(train_cfg.save_steps),
            "save_total_limit": int(train_cfg.save_total_limit),
            "lr_scheduler_type": train_cfg.lr_scheduler_type,
            "max_grad_norm": float(train_cfg.max_grad_norm),
            "fp16": bool(dist_cfg.get("fp16", False)),
            "bf16": bool(dist_cfg.get("bf16", False)),
            "optim": train_cfg.optim,
            "report_to": list(logging_cfg.report_to),
            "ddp_find_unused_parameters": False,
        }

        # FSDP configuration
        fsdp_mode = str(dist_cfg.mode).lower()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if fsdp_mode == "fsdp" and world_size >= 2:
            kwargs["fsdp"] = dist_cfg.fsdp
            fsdp_config = OmegaConf.to_container(
                dist_cfg.get("fsdp_config"), resolve=True
            )
            if fsdp_config:
                kwargs["fsdp_config"] = fsdp_config
        else:
            # Omit FSDP keys entirely if not used
            kwargs.pop("fsdp", None)
            kwargs.pop("fsdp_config", None)

        return TrainingArguments(**kwargs)

    def __call__(self, cfg: DictConfig):
        """Execute fine-tuning, handling checkpoints, training, and evaluation."""
        rank, world_size = self._setup_distributed_environment()

        out_dir = Path(cfg.paths.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self._latest_checkpoint(out_dir)

        set_seed(int(cfg.trainer.seed))

        model_cfg = OmegaConf.to_container(cfg.trainer.model, resolve=True) or {}
        dtype = self._resolve_dtype(model_cfg)

        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"],
            revision=model_cfg.get("revision"),
            use_fast=True,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model_kwargs: Dict[str, Any] = {
            "revision": model_cfg.get("revision"),
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
            "torch_dtype": dtype,
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            **model_kwargs,
        )

        train_dataset, eval_dataset = self._prepare_dataset(cfg, tokenizer, rank)

        training_args = self._build_training_arguments(cfg, out_dir, rank)

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=collator,
            callbacks=[RankZeroLoggingCallback()],
        )

        if rank == 0:
            logger.info("Starting training loop")

        train_result = trainer.train(resume_from_checkpoint=self.ckpt_dir)

        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if rank == 0:
            logger.info("Training complete. Beginning evaluation.")

        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics:
            eval_metrics["perplexity"] = math.exp(min(eval_metrics["eval_loss"], 20))
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        if rank == 0:
            logger.info(
                "All done. Train metrics: %s Eval metrics: %s", metrics, eval_metrics
            )
        else:
            print("All done.")

        # Cleanup distributed process group
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # Ensure all ranks finish
            torch.distributed.destroy_process_group()

    def checkpoint(
        self, *args: Any, **kwargs: Any
    ) -> submitit.helpers.DelayedSubmission:
        """Return a Submitit requeue submission that resumes this trainer."""
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
