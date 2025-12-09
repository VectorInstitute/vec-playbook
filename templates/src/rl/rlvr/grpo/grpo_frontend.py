"""Non-ML data logic related to GRPO."""

import logging
import pathlib

import torch
from transformers import AutoModelForCausalLM

from .config import GRPOHyperparameters
from .data_types import (
    AdvantageData,
    GRPOMetrics,
)
from .grpo_backend import optimize_grpo_one_epoch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def grpo_optimization_step(
    advantage_data: AdvantageData,
    current_policy_path: pathlib.Path,
    kl_ref_path: pathlib.Path,
    checkpoint_output_path: pathlib.Path,
    hyperparameters: GRPOHyperparameters,
    optimizer_path: pathlib.Path | None,
) -> GRPOMetrics:
    """Run one GRPO optimization step given advantages."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading kl_ref weights to CUDA: {kl_ref_path}")
    kl_ref_model = AutoModelForCausalLM.from_pretrained(kl_ref_path)
    kl_ref_model = kl_ref_model.to(device).bfloat16()  # type: ignore[argument]

    logger.info(f"Loading weights to CUDA {current_policy_path}")
    policy_model = AutoModelForCausalLM.from_pretrained(current_policy_path)
    policy_model = policy_model.to(device).bfloat16()  # type: ignore[argument]

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=hyperparameters.learning_rate,
        betas=hyperparameters.adam_betas,
        weight_decay=hyperparameters.adam_weight_decay,
    )
    if optimizer_path and optimizer_path.exists():
        logger.info(f"Loading optimizer state: {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        logger.info(
            "Re-initializing optimizer state since optimizer_path "
            "is None or does not exist"
        )

    policy_model, optimizer, metrics = optimize_grpo_one_epoch(
        batcher=advantage_data.get_iterator_for_training(
            batch_size=hyperparameters.batch_size_backprop,
            pad_to_length=hyperparameters.max_model_len,
        ),
        model_pi_ref=kl_ref_model,
        model=policy_model,
        optimizer=optimizer,
        gradient_accumulation_steps=hyperparameters.grad_acc_steps,
    )
    logger.info(f"metrics: {metrics.model_dump_json(indent=2)}")
    logger.info(f"Writing model to: {checkpoint_output_path}")
    policy_model.save_pretrained(checkpoint_output_path)

    if optimizer_path:
        logger.info(f"Writing optimizer to: {optimizer_path}")
        torch.save(optimizer.state_dict(), optimizer_path)
    else:
        logger.info("Not saving optimizer since optimizer_path is None.")

    return metrics
