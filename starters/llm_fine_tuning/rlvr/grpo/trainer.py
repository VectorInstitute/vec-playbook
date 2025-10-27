"""
GRPO Trainer for RL with Verifiable Reward.

See:
arxiv.org/pdf/2402.03300
(Algorithm 1 on page 14) for exact formulation

Usage:

PYTHONPATH="." \
uv run starters/llm_fine_tuning/rlvr/grpo/trainer.py \
--tokenizer /model-weights/Qwen3-0.6B \
--base_model /model-weights/Qwen3-0.6B \
--checkpoint_folder $SCRATCH/checkpoints/20251027-GRPO-Qwen3-0.6B
"""

import argparse
import asyncio
import logging
import os
import pathlib
from typing import Literal, Sequence

import agents
import datasets
import numpy as np
import openai
import torch
from rich.progress import track
from torch.nn.functional import softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from vec_inf.client import LaunchOptions
from vec_inf.client.oai_compatibility import AsyncVecInf

from starters.llm_fine_tuning.rlvr.grpo.data_types import (
    AdvantageData,
    BatchForInference,
    GRPOData,
    GRPOMetrics,
    PerTokenProbs,
    RewardDetailTokenized,
)
from starters.llm_fine_tuning.rlvr.grpo.grpo import optimize_grpo_one_epoch
from starters.llm_fine_tuning.rlvr.grpo.rollout_generation import (
    GRPORollout,
    RewardDetails,
    RLVRDataItem,
    RLVREvaluator,
    eval_agent,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_per_token_probs(
    batch: BatchForInference, model: "PreTrainedModel"
) -> PerTokenProbs:
    """Obtain per-token probs for a given batch and a given model.

    Params:
        batch: Tokenized _Batch (batch, length).
        model: Pretrained Causal LM.

    Return:
    -------
        np.ndarray (batch, length - 1, vocab_size)
    """
    device = next(model.parameters()).device
    input_ids = torch.as_tensor(batch.input_ids, dtype=torch.long, device=device)
    attention_mask = torch.as_tensor(
        batch.attention_mask, dtype=torch.long, device=device
    )

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits: (batch, length, vocab) where
        # (j, k, :) contains distribution for tokens k+1 of batch j.
        logits = output.logits
        # (batch, length - 1, vocab) skipping last token which has no label.
        # for all possible vocabs.
        probabilities_all = softmax(logits, dim=-1)[:, :-1, :]
        # (batch, length - 1, 1) skipping first token which is always given.
        target_token_ids = input_ids[:, 1:].unsqueeze(-1)
        # (batch, length - 1) for input_ids only.
        probabilities_selected = torch.gather(
            probabilities_all, dim=-1, index=target_token_ids
        ).squeeze(-1)

    return PerTokenProbs.from_batch(
        attention_mask=batch.attention_mask,
        num_valid=batch.num_valid,
        full=probabilities_all.float().cpu().numpy(),
        selected=probabilities_selected.float().cpu().numpy(),
    )


async def roll_out(
    policy_agent: agents.Agent,
    data: Sequence[RLVRDataItem],
    client: "openai.AsyncOpenAI",
    model_name: str,
) -> list[RewardDetails]:
    """Rollout online to get reward details, not yet tokenized."""
    example_agent_config = agents.RunConfig(
        model=agents.OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    )
    evaluator = RLVREvaluator(
        eval_agent,
        agent_run_config=example_agent_config,
        semaphore=semaphore_llm_judge,
    )

    rollout_generator = GRPORollout(policy_agent, evaluator=evaluator)
    return await rollout_generator.generate(
        data=data,
        agent_run_config=example_agent_config,
        semaphore=semaphore_rollout,
    )


async def grpo_step(
    policy_agent: agents.Agent,
    dataset: dict[Literal["train", "test"], Sequence[RLVRDataItem]],
    current_policy_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    kl_ref_path: pathlib.Path,
    args,
) -> GRPOMetrics:
    """Run one GRPO step.

    Returns updated policy path.
    """
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        current_policy_path
    )
    model_name = current_policy_path.parts[-1]

    async with AsyncVecInf(
        model=model_name,
        num_replicas=args.num_rollout_workers,
        options=LaunchOptions(
            qos="scavenger",
            model_weights_parent_dir=current_policy_path.parent.as_posix(),
        ),
    ) as policy_client:
        advantage_data = {
            _split: AdvantageData.from_list_of_rewards(
                [
                    RewardDetailTokenized.from_messages(
                        _detail.rollout.messages,
                        reward=_detail.score,
                        tokenizer=tokenizer,
                    )
                    for _detail in await roll_out(
                        policy_agent=policy_agent,
                        data=_data,
                        client=policy_client,
                        model_name=model_name,
                    )
                ]
            )
            for _split, _data in dataset.items()
        }

    device = torch.device("cuda:0")

    logger.info("Loading kl_ref weights")
    kl_ref_model = AutoModelForCausalLM.from_pretrained(kl_ref_path)
    logger.info("Loading kl_ref weights to CUDA.")
    kl_ref_model = kl_ref_model.to(device)

    per_token_probs_ref = sum(
        get_per_token_probs(_batch, kl_ref_model)
        for _batch in track(
            advantage_data["train"].get_iterator_for_inference(batch_size=5),
            description="Inference: pi_ref",
        )
    )
    del kl_ref_model

    base_model = AutoModelForCausalLM.from_pretrained(current_policy_path)
    logger.info("Loading weights to CUDA.")
    base_model = base_model.to(device)

    per_token_probs_base = sum(
        get_per_token_probs(_batch, base_model)
        for _batch in track(
            advantage_data["train"].get_iterator_for_inference(batch_size=5),
            description="Inference: pi_old",
        )
    )

    assert isinstance(per_token_probs_ref, PerTokenProbs)
    assert isinstance(per_token_probs_base, PerTokenProbs)
    grpo_data = GRPOData(
        **advantage_data["train"].model_dump(),
        ref_probs=per_token_probs_ref,
        base_probs=per_token_probs_base,
    )

    base_model = optimize_grpo_one_epoch(
        batcher=grpo_data.get_iterator_for_training(batch_size=args.bsz_train),
        model=base_model,
        gradient_accumulation_steps=1,
    )
    base_model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    return GRPOMetrics(
        eval_advantage=np.mean(advantage_data["test"]._group_rewards()).item(),
        train_advantage=np.mean(advantage_data["test"]._group_rewards()).item(),
    )


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True)
parser.add_argument("--checkpoint_folder", required=True)
parser.add_argument(
    "--keep_all_checkpoints",
    action="store_true",
    help="If not set, only the most recent checkpoint would be kept.",
)
parser.add_argument("--tokenizer", required=True)
parser.add_argument(
    "--bsz_inference", type=int, default=16, help="batch size for getting pi_ref"
)
parser.add_argument("--bsz_train", type=int, default=8, help="batch size for backprop")
parser.add_argument("--num_steps", type=int, default=8, help="number of update steps")
parser.add_argument("--num_rollout_workers", type=int, default=2)
parser.add_argument("--concurrency_rollout", type=int, default=36)
parser.add_argument("--concurrency_llm_judge", type=int, default=36)


policy_agent = agents.Agent(
    "Math Problem Solver", instructions="Solve the math problem."
)


if __name__ == "__main__":
    args = parser.parse_args()

    dataset_hf = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset_hf, datasets.DatasetDict)

    dataset_full: dict[Literal["train", "test"], Sequence[RLVRDataItem]] = {
        k: [
            RLVRDataItem(
                query=_row["question"],
                target=_row["answer"].split("#")[-1].strip(),
            )
            for _row in _split
        ]
        for k, _split in dataset_hf.items()
    }
    dataset = {
        "train": dataset_full["train"][:1000],
        "test": dataset_full["test"][:100],
    }
    print({k: len(v) for k, v in dataset.items()})

    # start using base model as kl. Update this after each step.
    checkpoint_folder = pathlib.Path(args.checkpoint_folder)
    current_policy_path = pathlib.Path(args.base_model)
    kl_ref_path = current_policy_path

    for _step_index in range(args.num_steps):
        updated_checkpoint_path = checkpoint_folder / f"step_{_step_index}"
        metrics = asyncio.run(
            grpo_step(
                policy_agent=policy_agent,
                dataset=dataset,
                current_policy_path=current_policy_path,
                checkpoint_path=updated_checkpoint_path,
                kl_ref_path=kl_ref_path,
                args=args,
            )
        )

        if (not args.keep_all_checkpoints) and (_step_index > 0):
            os.rmdir(kl_ref_path)

        kl_ref_path = current_policy_path
        current_policy_path = updated_checkpoint_path
