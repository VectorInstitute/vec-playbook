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
--kl_ref_model /model-weights/Qwen3-0.6B
"""

import argparse
import asyncio
import logging
from typing import Literal, Sequence

import agents
import numpy as np
import openai
import torch
from rich.progress import track
from torch.nn.functional import softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from starters.llm_fine_tuning.rlvr.agents.examples import weather_agent
from starters.llm_fine_tuning.rlvr.grpo.data_types import (
    AdvantageData,
    BatchForInference,
    GRPOData,
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
    data: Sequence[RLVRDataItem],
    client: "openai.AsyncOpenAI",
    model_name: str,
    semaphore_rollout: "asyncio.Semaphore",
    semaphore_llm_judge: "asyncio.Semaphore",
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

    rollout_generator = GRPORollout(weather_agent, evaluator=evaluator)
    return await rollout_generator.generate(
        data=data,
        agent_run_config=example_agent_config,
        semaphore=semaphore_rollout,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True)
parser.add_argument("--kl_ref_model", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument(
    "--bsz_inference", type=int, default=16, help="batch size for getting pi_ref"
)
parser.add_argument("--bsz_train", type=int, default=8, help="batch size for backprop")
parser.add_argument("--concurrency_rollout", type=int, default=36)
parser.add_argument("--concurrency_llm_judge", type=int, default=36)

example_data = [
    RLVRDataItem(query="Weather in Shanghai", target="28 degrees Celsius, clear"),
    RLVRDataItem(
        query="Weather in Vancouver", target="-10 degrees Celsius, heavy snow"
    ),
]
example_dataset: dict[Literal["train", "test"], Sequence[RLVRDataItem]] = {
    "train": example_data,
    "test": example_data,
}

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info(f"Tokenizer: {tokenizer}")
    semaphore_rollout = asyncio.Semaphore(args.concurrency_rollout)
    semaphore_llm_judge = asyncio.Semaphore(args.concurrency_llm_judge)

    for _step in range(72):
        # Generate on-policy rollouts and rewards- not tokenized yet.
        # TODO: replace client and model_name with OpenAI-compatible vec-inf wrapper.
        advantage_data = {
            _split: AdvantageData.from_list_of_rewards(
                [
                    RewardDetailTokenized.from_messages(
                        _detail.rollout.messages,
                        reward=_detail.score,
                        tokenizer=tokenizer,
                    )
                    for _detail in asyncio.run(
                        roll_out(
                            data=_data,
                            client=openai.AsyncOpenAI(api_key="EMPTY"),
                            model_name="Qwen3-0.6B",
                            semaphore_rollout=semaphore_rollout,
                            semaphore_llm_judge=semaphore_llm_judge,
                        )
                    )
                ]
            )
            for _split, _data in example_dataset.items()
        }
        eval_advantage = np.mean(advantage_data["test"]._group_rewards()).item()
        print(f"Step {_step}, eval_advantage: {eval_advantage:.3f}")

        device = torch.device("cuda:0")

        logger.info("Loading kl_ref weights")
        kl_ref_model = AutoModelForCausalLM.from_pretrained(args.kl_ref_model)
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

        base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
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
