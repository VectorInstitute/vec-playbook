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
import logging

import torch
from rich.progress import track
from torch.nn.functional import softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from starters.llm_fine_tuning.rlvr.grpo.data_types import (
    AdvantageData,
    BatchForInference,
    GRPOData,
    PerTokenProbs,
    RewardDetail,
)
from starters.llm_fine_tuning.rlvr.grpo.grpo import optimize_grpo_one_epoch


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


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True)
parser.add_argument("--kl_ref_model", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument(
    "--bsz_inference", type=int, default=16, help="batch size for getting pi_ref"
)
parser.add_argument("--bsz_train", type=int, default=8, help="batch size for backprop")


# def parameter_update(model: AutoModelForCausalLM, advantage_batch: AdvantageBatch)


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info(f"Tokenizer: {tokenizer}")
    example_reward_details = [
        RewardDetail.from_messages(
            messages=[
                {"role": "system", "content": "Example system instructions."},
                {"role": "user", "content": "Example user message."},
                {"role": "assistant", "content": _message},
            ],
            reward=_reward,
            tokenizer=tokenizer,
        )
        for _reward, _message in [
            (1.0, "Example correct response."),
            (1.0, "Another correct response."),
            (0.0, "Example incorrect response."),
            (1.0, "Example correct response 1."),
            (1.0, "Another correct response 1."),
            (0.0, "Example incorrect response 1."),
        ]
    ]

    advantage_data = AdvantageData.from_list_of_rewards(example_reward_details)

    device = torch.device("cuda:0")

    logger.info("Loading kl_ref weights")
    kl_ref_model = AutoModelForCausalLM.from_pretrained(args.kl_ref_model)
    logger.info("Loading kl_ref weights to CUDA.")
    kl_ref_model = kl_ref_model.to(device)

    per_token_probs_ref = sum(
        get_per_token_probs(_batch, kl_ref_model)
        for _batch in track(
            advantage_data.get_iterator_for_inference(batch_size=5),
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
            advantage_data.get_iterator_for_inference(batch_size=5),
            description="Inference: pi_old",
        )
    )

    assert isinstance(per_token_probs_ref, PerTokenProbs)
    assert isinstance(per_token_probs_base, PerTokenProbs)
    grpo_data = GRPOData(
        **advantage_data.model_dump(),
        ref_probs=per_token_probs_ref,
        base_probs=per_token_probs_base,
    )

    for training_batch in grpo_data.get_iterator_for_training(
        batch_size=args.bsz_train
    ):
        print({k: type(v) for k, v in training_batch.model_dump().items()})
        break

    base_model = optimize_grpo_one_epoch(
        batcher=grpo_data.get_iterator_for_training(batch_size=args.bsz_train),
        model=base_model,
        gradient_accumulation_steps=1,
    )
