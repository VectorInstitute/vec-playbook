"""
GRPO Trainer for RL with Verifiable Reward.

See:
arxiv.org/pdf/2402.03300
(Algorithm 1 on page 14) for exact formulation

Usage:

PYTHONPATH="." \
uv run starters/llm_fine_tuning/rlvr/grpo/trainer.py \
--tokenizer /model-weights/Qwen2.5-1.5B-Instruct \
--evaluator_model /model-weights/Qwen3-8B \
--base_model /model-weights/Qwen2.5-1.5B-Instruct \
--checkpoint_folder $SCRATCH/checkpoints/20251027-GRPO-Qwen2.5-1.5B-Instruct \
--max_model_len 2048 \
--bsz_train 2 \
--bsz_inference 2 \
--vllm_cache_dir $SCRATCH/.cache/vllm_compiled_graphs/ \
--vllm_uv_prefix "./run_in_container.sh uv run python" \
--vllm_partition a40 \
--vllm_qos m5 \
--vllm_num_replicas 1 \
--vllm_concurrency 128 \
--evaluator_model_replicas 1 \
--evaluator_model_concurrency 32
"""

import argparse
import asyncio
import logging
import os
import pathlib
from shutil import rmtree
from typing import Literal, Sequence

import agents
import datasets
import submitit
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from vllm import EngineArgs
from vllm.config import CompilationConfig

from starters.llm_fine_tuning.rlvr.grpo.data_types import (
    AdvantageData,
    GRPOHyperparameters,
    GRPOMetrics,
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


parser = argparse.ArgumentParser()
parser.add_argument("--evaluator_model", default="/model-weights/Qwen3-8B")
parser.add_argument("--evaluator_model_replicas", type=int, default=2)
parser.add_argument("--evaluator_model_concurrency", type=int, default=32)
parser.add_argument("--evaluator_max_model_len", type=int, default=2048)
parser.add_argument("--base_model", required=True)
parser.add_argument("--checkpoint_folder", required=True)
parser.add_argument(
    "--optimizer_path", help="Use checkpoint_folder/optimizer_state if not specified"
)
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
parser.add_argument(
    "--logging_level", type=int, default=36, help="per-worker concurrency"
)


policy_agent = agents.Agent(
    "Math Problem Solver", instructions="Solve the math problem."
)
logger = logging.getLogger(__name__)

# TODO: delete after validating backprop logic.
LOAD_FROM_CACHE = False

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

    submitit_executor = submitit.SlurmExecutor(
        folder=args.submitit_log_dir, python=args.vllm_uv_prefix
    )
    submitit_args = SubmititArgs(
        partition=args.vllm_partition,
        account=args.vllm_account,
        qos=args.vllm_qos,
    )
    submitit_executor.update_parameters(**submitit_args.to_submitit_parameters())

    # Initial: base model for kl. Update this to one step behind after each step.
    checkpoint_folder = pathlib.Path(args.checkpoint_folder)
    optimizer_path = (
        pathlib.Path(args.optimizer_path)
        if args.optimizer_path
        else checkpoint_folder / "optimizer_state"
    )
    # TODO: add support for checkpoint-restore
    rmtree(optimizer_path, ignore_errors=True)
    base_model_path = pathlib.Path(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # The following are updated after each update.
    current_policy_path = base_model_path
    kl_ref_path = current_policy_path

    for _step_index in range(args.num_steps):
        updated_checkpoint_path = checkpoint_folder / f"step_{_step_index}"
        logger.info(f"updated_checkpoint_path: {updated_checkpoint_path}")
        os.makedirs(updated_checkpoint_path, exist_ok=True)

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
                    max_model_len=args.evaluator_max_model_len,
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
            logger.info(f"{_split} split: avg reward {_data.avg_reward}")
            _path = checkpoint_folder / f"data_{_step_index}_{_split}.json"
            with open(_path, "w") as data_file:
                data_file.write(_data.model_dump_json())

        logger.info(f"kl_ref_path: {kl_ref_path}")
        metrics = grpo_optimization_step(
            advantage_data=advantage_data["train"],
            current_policy_path=current_policy_path,
            kl_ref_path=kl_ref_path,
            checkpoint_output_path=updated_checkpoint_path,
            optimizer_path=optimizer_path,
            hyperparameters=hyperparameters,
        )
        logger.info(f"Writing tokenizer to: {updated_checkpoint_path}")
        tokenizer.save_pretrained(updated_checkpoint_path)
        logger.info(metrics)
        with open(checkpoint_folder / f"metrics_{_step_index}", "w") as log_file:
            log_file.write(metrics.model_dump_json())

        if (not args.keep_all_checkpoints) and (kl_ref_path != base_model_path):
            rmtree(kl_ref_path)
            logger.info(f"Deleting: {kl_ref_path}")

        kl_ref_path = current_policy_path
        current_policy_path = updated_checkpoint_path
