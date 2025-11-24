"""Non-ML data logic related to GRPO."""

import logging
import pathlib
from typing import Sequence

import agents
import torch
from rlvr.grpo.grpo_backend import optimize_grpo_one_epoch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)

from templates.src.rlvr.grpo.data_types import (
    AdvantageData,
    GRPOHyperparameters,
    GRPOMetrics,
    RewardDetailTokenized,
)
from templates.src.rlvr.grpo.rollout_generation import (
    GRPORollout,
    RewardDetail,
    RLVRDataItem,
    RLVREvaluator,
)
from templates.src.rlvr.submitit_vllm import SubmititVLLM


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def roll_out(
    policy_agent: agents.Agent,
    data: Sequence[RLVRDataItem],
    policy_submitit_vllm: SubmititVLLM,
    evaluator_submitit_vllm: SubmititVLLM,
) -> Sequence[RewardDetail]:
    """Rollout online to get reward details, not yet tokenized."""
    evaluator = RLVREvaluator(eval_agent, submitit_vllm=evaluator_submitit_vllm)
    rollout_generator = GRPORollout(policy_agent, evaluator=evaluator)
    return await rollout_generator.generate(
        data=data, submitit_vllm=policy_submitit_vllm
    )


async def get_grpo_advantage(
    policy_agent: agents.Agent,
    data_items: Sequence[RLVRDataItem],
    policy_vllm: SubmititVLLM,
    evaluator_vllm: SubmititVLLM,
    tokenizer: PreTrainedTokenizerFast,
    max_len: int,
) -> AdvantageData:
    """Generate and compute GRPO advantages for a given data split."""
    return AdvantageData.from_list_of_rewards(
        [
            RewardDetailTokenized.from_messages(
                _detail.rollout.messages,
                reward=_detail.result,
                tokenizer=tokenizer,
                pad_to=max_len,
            )
            for _detail in await roll_out(
                policy_agent=policy_agent,
                data=data_items,
                policy_submitit_vllm=policy_vllm,
                evaluator_submitit_vllm=evaluator_vllm,
            )
        ]
    )


def grpo_optimization_step(
    advantage_data: AdvantageData,
    current_policy_path: pathlib.Path,
    kl_ref_path: pathlib.Path,
    checkpoint_output_path: pathlib.Path,
    hyperparameters: GRPOHyperparameters,
    optimizer_path: pathlib.Path,
) -> GRPOMetrics:
    """Run one GRPO optimization step given advantages."""
    device = torch.device("cuda:0")

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
    if optimizer_path.exists():
        logger.info(f"Loading optimizer state: {optimizer_path}")
        optimizer.load_state_dict(torch.load(optimizer_path))
    else:
        logger.info(
            f"Initializing optimizer state since {optimizer_path} does not exist"
        )

    policy_model, optimizer, metrics = optimize_grpo_one_epoch(
        batcher=advantage_data.get_iterator_for_training(
            batch_size=hyperparameters.train_batch_size,
            pad_to_length=hyperparameters.max_model_len,
        ),
        model_pi_ref=kl_ref_model,
        model=policy_model,
        optimizer=optimizer,
        gradient_accumulation_steps=1,
    )
    logger.info(f"metrics: {metrics.model_dump_json(indent=2)}")
    logger.info(f"Writing model to: {checkpoint_output_path}")
    policy_model.save_pretrained(checkpoint_output_path)

    logger.info(f"Writing optimizer to: {optimizer_path}")
    torch.save(optimizer.state_dict(), optimizer_path)

    return metrics
