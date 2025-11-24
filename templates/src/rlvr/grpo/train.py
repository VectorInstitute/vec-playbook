"""GRPO training using Hydra + Submitit."""

import asyncio
import logging
import re
from collections import defaultdict
from os import makedirs
from typing import Sequence

import agents
import datasets
import submitit
from transformers import AutoTokenizer
from vllm import EngineArgs
from vllm.config import CompilationConfig

from templates.src.rlvr.agents_integration.rollout_translation import (
    Rollout,
    translate_rollout,
)
from templates.src.rlvr.async_utils import gather_with_progress
from templates.src.rlvr.grpo.config import DataConfig, GRPOConfig
from templates.src.rlvr.grpo.data_types import AdvantageData, RewardDetailTokenized
from templates.src.rlvr.grpo.grpo_frontend import RLVRDataItem
from templates.src.rlvr.grpo.rollout_generation import EvalResult, RLVREvaluator
from templates.src.rlvr.submitit_vllm import SubmititVLLM


policy_agent = agents.Agent(
    "Math Problem Solver", instructions="Solve the math problem."
)

# Eval agent must support structured output and use output_type EvalResult
eval_agent = agents.Agent(
    "Evaluator",
    instructions=(
        "Evaluate if the `proposed` response matches the ground-truth `target`."
        "Give a score of 0.0 if incorrect and 1.0 if correct."
    ),
    output_type=EvalResult,
)


def load_data(data_cfg: DataConfig):
    """Extract question-answer pairs from the given HuggingFace data source."""
    output: dict[str, list[RLVRDataItem]] = {}
    for _split, _split_cfg in [
        ("train", data_cfg.train_split),
        ("test", data_cfg.test_split),
    ]:
        _dataset = datasets.load_dataset(
            data_cfg.dataset_name, data_cfg.subset, split=_split_cfg
        )
        assert isinstance(_dataset, datasets.Dataset)
        output[_split] = []

        for _row in _dataset:
            assert isinstance(_row, dict)
            _query = _row[data_cfg.query_column]
            _target = _row[data_cfg.target_column]

            if data_cfg.target_regexp is not None:
                _match = re.search(data_cfg.target_regexp, _target, re.DOTALL)
                if _match:
                    _target = _match.group(1)

            output[_split].append(RLVRDataItem(query=_query, target=_target))

    return output


class GRPOTrainer(submitit.helpers.Checkpointable):
    """GRPO using on-policy SLURM rollout and one-GPU backprop."""

    def __init__(self, cfg: GRPOConfig):
        self.ckpt_dir = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.cfg = cfg
        self.submitit_executor = submitit.SlurmExecutor(
            folder=cfg.submitit_logs_folder, python=cfg.rollout_vllm.submitit_python
        )
        self.submitit_executor.update_parameters(
            **cfg.rollout_vllm.submitit_args.to_submitit_parameters()
        )
        self.optimizer_path = (
            cfg.optimizer_folder
            if cfg.optimizer_folder
            else cfg.checkpoint_folder / "optimizer_state"
        )

        # Initialize policy and KL ref to base model.
        self.current_policy_path = cfg.base_model
        self.kl_ref_path = cfg.base_model
        self.evaluator = RLVREvaluator(eval_agent)

    def generate_rollout(self, data: Sequence[RLVRDataItem]) -> Sequence[Rollout]:
        """Generate rollouts using policy model/agent."""
        with SubmititVLLM(
            engine_args=EngineArgs(
                model=self.current_policy_path,
                compilation_config=CompilationConfig(
                    cache_dir=self.cfg.rollout_vllm.cache_dir.as_posix()
                ),
                max_model_len=self.cfg.rollout_vllm.max_model_len,
            ),
            submitit_executor=self.submitit_executor,
            concurrency_per_worker=self.cfg.rollout_vllm.concurrency_per_replica,
            num_replicas=self.cfg.rollout_vllm.num_replicas,
            logger_name_prefix="submitit_vllm.policy",
        ) as policy_vllm:
            coros = [policy_vllm.run_agent(policy_agent, _item.query) for _item in data]
            coro = gather_with_progress(coros, "Rollout...")
            rollouts = asyncio.get_event_loop().run_until_complete(coro)

        return [_result.final_output for _result in rollouts]

    def score_rollouts(self, data: Sequence[RLVRDataItem], answers: Sequence[str]):
        """Score a list of answers, using LLM judge."""
        with SubmititVLLM(
            engine_args=EngineArgs(
                model=self.cfg.llm_judge_vllm.model_name,
                compilation_config=CompilationConfig(
                    cache_dir=self.cfg.llm_judge_vllm.cache_dir.as_posix()
                ),
                max_model_len=self.cfg.llm_judge_vllm.max_model_len,
            ),
            submitit_executor=self.submitit_executor,
            concurrency_per_worker=self.cfg.llm_judge_vllm.concurrency_per_replica,
            num_replicas=self.cfg.llm_judge_vllm.num_replicas,
            logger_name_prefix="submitit_vllm.evaluator",
        ) as evaluator_vllm:
            coros = [
                self.evaluator(_item, proposed=_answer, submitit_vllm=evaluator_vllm)
                for _item, _answer in zip(data, answers)
            ]
            coro = gather_with_progress(coros, "LLM-Judge...")
            return asyncio.get_event_loop().run_until_complete(coro)

    def calculate_advantage(
        self,
        data: Sequence[RLVRDataItem],
        run_results: "Sequence[agents.RunResult]",
        evals: Sequence[EvalResult],
    ):
        """Calculate advantage given rollouts and eval results."""
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        tokenized_reward_details = [
            RewardDetailTokenized.from_messages(
                translate_rollout(_run_result, _item.query, policy_agent).messages,
                reward=_eval.score,
                tokenizer=tokenizer,
                pad_to=self.cfg.hyperparameters.max_model_len,
            )
            for _item, _run_result, _eval in zip(data, run_results, evals)
        ]

        return AdvantageData.from_list_of_rewards(tokenized_reward_details)

    def run_step(self, index: int):
        """Run one GRPO step."""
        updated_checkpoint_path = self.cfg.checkpoint_folder / f"step_{index}"
        self.logger.info(f"updated_checkpoint_path: {updated_checkpoint_path}")
        makedirs(updated_checkpoint_path, exist_ok=True)

    def __call__(self, cfg: GRPOConfig) -> None:
        """Run full GRPO loop."""
        dataset = load_data(cfg.data)
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

        raise ValueError(cfg.model_dump_json(indent=2))

    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        """Save state and launch the same callable with the same arguments."""
        return super().checkpoint(*args, **kwargs)
