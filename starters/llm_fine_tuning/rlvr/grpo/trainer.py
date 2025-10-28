"""
GRPO Trainer for RL with Verifiable Reward.

See:
arxiv.org/pdf/2402.03300
(Algorithm 1 on page 14) for exact formulation

Usage:

PYTHONPATH="." \
uv run starters/llm_fine_tuning/rlvr/grpo/trainer.py \
--tokenizer /model-weights/Qwen3-0.6B \
--evaluator_model /model-weights/Qwen3-8B \
--base_model /model-weights/Qwen3-0.6B \
--checkpoint_folder $SCRATCH/checkpoints/20251027-GRPO-Qwen3-0.6B \
--max_model_len 2048 \
--bsz_train 2 \
--bsz_inference 2 \
--vllm_cache_dir $SCRATCH/.cache/vllm_compiled_graphs/ \
--vllm_uv_prefix "./run_in_container.sh uv run python" \
--vllm_partition a40 \
--vllm_qos m5 \
--vllm_num_replicas 16 \
--vllm_concurrency 128
"""

import argparse
import asyncio
import logging
import os
import pathlib
import time
from typing import Literal, Sequence

import agents
import datasets
import numpy as np
import submitit
import torch
from rich.progress import track
from torch.nn.functional import softmax
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from vllm import EngineArgs
from vllm.config import CompilationConfig

from starters.llm_fine_tuning.rlvr.grpo.data_types import (
    AdvantageData,
    BatchForInference,
    GRPOData,
    GRPOHyperparameters,
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
from starters.llm_fine_tuning.rlvr.submitit_vllm import SubmititArgs, SubmititVLLM


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def roll_out(
    policy_agent: agents.Agent,
    data: Sequence[RLVRDataItem],
    policy_submitit_vllm: SubmititVLLM,
    evaluator_submitit_vllm: SubmititVLLM,
) -> Sequence[RewardDetails]:
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
                reward=_detail.score,
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


def get_per_token_probs(
    batch: BatchForInference, model: "PreTrainedModel"
) -> PerTokenProbs:
    """Obtain per-token probs for a given batch and a given model.

    Params:
        batch: Tokenized _Batch (batch, length).
        model: Pretrained Causal LM.

    Returns
    -------
        PerTokenProbs storing dense torch tensors on the model device.
    """
    device = model.device
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

    return PerTokenProbs(
        attention_mask=attention_mask[: batch.num_valid],
        full=probabilities_all.detach()[: batch.num_valid],
        selected=probabilities_selected.detach()[: batch.num_valid],
    )


def grpo_optimization_step(
    advantage_data: AdvantageData,
    current_policy_path: pathlib.Path,
    kl_ref_path: pathlib.Path,
    checkpoint_output_path: pathlib.Path,
    hyperparameters: GRPOHyperparameters,
) -> GRPOMetrics:
    """Run one GRPO optimization step given advantages."""
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        current_policy_path
    )

    device = torch.device("cuda:0")

    logger.info("Loading kl_ref weights")
    kl_ref_model = AutoModelForCausalLM.from_pretrained(kl_ref_path)
    logger.info("Loading kl_ref weights to CUDA.")
    kl_ref_model = kl_ref_model.to(device).bfloat16()  # type: ignore[argument]

    per_token_probs_ref_items = [
        get_per_token_probs(_batch, kl_ref_model)
        for _batch in track(
            advantage_data.get_iterator_for_inference(
                batch_size=hyperparameters.inference_batch_size,
                pad_to_length=hyperparameters.max_model_len,
            ),
            description="Inference: pi_ref",
        )
    ]
    per_token_probs_ref = sum(per_token_probs_ref_items)
    del kl_ref_model

    base_model = AutoModelForCausalLM.from_pretrained(current_policy_path)
    logger.info("Loading weights to CUDA.")
    base_model = base_model.to(device).bfloat16()  # type: ignore[argument]

    per_token_probs_base = sum(
        get_per_token_probs(_batch, base_model)
        for _batch in track(
            advantage_data.get_iterator_for_inference(
                batch_size=hyperparameters.inference_batch_size,
                pad_to_length=hyperparameters.max_model_len,
            ),
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

    base_model, metrics = optimize_grpo_one_epoch(
        batcher=grpo_data.get_iterator_for_training(
            pad_to_length=hyperparameters.max_model_len,
            batch_size=hyperparameters.train_batch_size,
        ),
        model=base_model,
        gradient_accumulation_steps=1,
    )
    base_model.save_pretrained(checkpoint_output_path)
    tokenizer.save_pretrained(checkpoint_output_path)

    return GRPOMetrics(
        advantage=np.mean(advantage_data._group_rewards()).item(),
        **metrics.model_dump(),
    )


parser = argparse.ArgumentParser()
parser.add_argument("--evaluator_model", default="/model-weights/Qwen3-8B")
parser.add_argument("--evaluator_model_replicas", type=int, default=2)
parser.add_argument("--evaluator_model_concurrency", type=int, default=32)
parser.add_argument("--base_model", required=True)
parser.add_argument("--checkpoint_folder", required=True)
parser.add_argument(
    "--keep_all_checkpoints",
    action="store_true",
    help="If not set, only the most recent checkpoint would be kept.",
)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--max_model_len", type=int, default=2048)
parser.add_argument(
    "--bsz_inference", type=int, default=2, help="batch size for getting pi_ref and kl"
)
parser.add_argument("--bsz_train", type=int, default=2, help="batch size for backprop")
parser.add_argument("--num_steps", type=int, default=8, help="number of update steps")
parser.add_argument("--num_rollout_workers", type=int, default=2)
parser.add_argument(
    "--submitit_log_dir",
    type=str,
    default=pathlib.Path(os.environ["HOME"]) / ".submitit",
)
parser.add_argument(
    "--vllm_uv_prefix",
    type=str,
    default="uv run python",
    help="Add extra prefixes for e.g., running vLLM in singularity",
)
parser.add_argument("--vllm_cache_dir", type=str, default="/tmp/")
parser.add_argument("--vllm_partition", type=str)
parser.add_argument("--vllm_qos", type=str)
parser.add_argument("--vllm_account", type=str)
parser.add_argument(
    "--vllm_num_replicas", type=int, default=5, help="number of vLLM worker jobs"
)
parser.add_argument(
    "--vllm_concurrency", type=int, default=36, help="per-worker concurrency"
)


policy_agent = agents.Agent(
    "Math Problem Solver", instructions="Solve the math problem."
)

LOAD_FROM_CACHE = True

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    hyperparameters = GRPOHyperparameters(
        train_batch_size=args.bsz_train,
        inference_batch_size=args.bsz_inference,
        max_model_len=args.max_model_len,
    )

    dataset_hf = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset_hf, datasets.DatasetDict)

    dataset_full = {
        k: [
            RLVRDataItem(
                query=_row["question"],
                target=_row["answer"].split("#")[-1].strip(),
            )
            for _row in _split
        ]
        for k, _split in dataset_hf.items()
    }

    dataset: dict[Literal["train", "test"], Sequence[RLVRDataItem]]
    dataset = {
        "train": dataset_full["train"][:32],
        "test": dataset_full["test"][:32],
    }
    print({k: len(v) for k, v in dataset.items()})

    # start using base model as kl. Update this after each step.
    checkpoint_folder = pathlib.Path(args.checkpoint_folder)
    current_policy_path = pathlib.Path(args.base_model)
    kl_ref_path = current_policy_path
    tokenizer = AutoTokenizer.from_pretrained(current_policy_path)
    submitit_executor = submitit.SlurmExecutor(
        folder=args.submitit_log_dir, python=args.vllm_uv_prefix
    )
    submitit_args = SubmititArgs(
        partition=args.vllm_partition,
        account=args.vllm_account,
        qos=args.vllm_qos,
    )
    submitit_executor.update_parameters(**submitit_args.to_submitit_parameters())

    for _step_index in range(args.num_steps):
        updated_checkpoint_path = checkpoint_folder / f"step_{_step_index}"
        os.makedirs(updated_checkpoint_path, exist_ok=True)

        if LOAD_FROM_CACHE:
            advantage_data = {}
            for _split in ("test", "train"):
                _path = checkpoint_folder / f"data_{_step_index}_{_split}.json"
                if _path.exists():
                    with open(_path, "r") as data_file:
                        advantage_data[_split] = AdvantageData.model_validate_json(
                            data_file.read()
                        )

        else:
            with (
                SubmititVLLM(
                    engine_args=EngineArgs(
                        model=current_policy_path.as_posix(),
                        compilation_config=CompilationConfig(
                            cache_dir=f"{args.vllm_cache_dir}/policy"
                        ),
                        max_model_len=hyperparameters.max_model_len,
                    ),
                    submitit_executor=submitit_executor,
                    concurrency_per_worker=args.vllm_concurrency,
                    num_replicas=args.vllm_num_replicas,
                    logger_name_prefix="submitit_vllm.policy",
                ) as policy_vllm,
                SubmititVLLM(
                    engine_args=EngineArgs(
                        model=args.evaluator_model,
                        compilation_config=CompilationConfig(
                            cache_dir=f"{args.vllm_cache_dir}/evaluator"
                        ),
                    ),
                    submitit_executor=submitit_executor,
                    concurrency_per_worker=args.evaluator_model_concurrency,
                    num_replicas=args.evaluator_model_replicas,
                    logger_name_prefix="submitit_vllm.evaluator",
                ) as evaluator_vllm,
            ):
                # TODO: update rich.progress setup and run the two splits concurrently.
                advantage_data = {
                    _split: asyncio.get_event_loop().run_until_complete(
                        get_grpo_advantage(
                            policy_agent,
                            dataset[_split],
                            policy_vllm=policy_vllm,
                            evaluator_vllm=evaluator_vllm,
                            tokenizer=tokenizer,
                            max_len=args.max_model_len,
                        )
                    )
                    for _split in ("train", "test")
                }

        for _split, _data in advantage_data.items():
            _path = checkpoint_folder / f"data_{_step_index}_{_split}.json"
            with open(_path, "w") as data_file:
                data_file.write(_data.model_dump_json())

        metrics = grpo_optimization_step(
            advantage_data=advantage_data["train"],
            current_policy_path=current_policy_path,
            kl_ref_path=kl_ref_path,
            checkpoint_output_path=updated_checkpoint_path,
            hyperparameters=hyperparameters,
        )
        with open(checkpoint_folder / f"metrics_{_step_index}", "w") as log_file:
            log_file.write(metrics.model_dump_json())

        if (not args.keep_all_checkpoints) and (_step_index > 0):
            os.rmdir(kl_ref_path)

        kl_ref_path = current_policy_path
        current_policy_path = updated_checkpoint_path
