"""GRPO training using Hydra + Submitit."""

import asyncio
import datetime
import logging
import pathlib
import re
from os import environ, makedirs
from shutil import rmtree
from typing import Sequence

import agents
import datasets
import pydantic
import submitit
from transformers import AutoTokenizer
from vllm import EngineArgs
from vllm.config import CompilationConfig

from ..agents_integration.logging_utils import set_up_logging
from ..agents_integration.rollout_translation import translate_rollout
from ..async_utils import gather_with_progress
from ..langfuse import (
    TraceID,
    add_score,
    initialize_lf_dataset,
    maybe_traced,
)
from ..progress_utils import spinner
from ..submitit_vllm import ExecutorConfig, SubmititVLLM
from .config import DataConfig, GRPOConfig
from .data_types import (
    AdvantageData,
    GRPOMetrics,
    RewardDetailTokenized,
)
from .grpo_frontend import grpo_optimization_step
from .rollout_generation import (
    EvalResult,
    RLVRDataItem,
    RLVREvaluator,
)


class _CheckpointPaths(pydantic.BaseModel):
    """Checkpoint paths."""

    current: pathlib.Path
    output: pathlib.Path
    kl_ref: pathlib.Path


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

    def __init__(self, cfg: GRPOConfig):
        self.ckpt_dir = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.cfg = cfg
        self.run_name = (
            f"{cfg.run_name}-{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
        )

        self.optimizer_path = (
            cfg.optimizer_folder
            if cfg.optimizer_folder
            else cfg.checkpoint_folder / "optimizer_state"
        )

        # Keep history of checkpoints for KL calculations
        # these are automatically recycled.
        self.checkpoints: list[pathlib.Path] = []

        # Initialize policy and KL ref to base model.
        self.evaluator = RLVREvaluator(self.eval_agent)

        # Run one replica locally while running other replicas on SLURM.
        self.local_executor = submitit.LocalExecutor(
            folder=cfg.submitit_logs_folder, python="uv run python"
        )
        visible_gpus = list(
            map(int, environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        )
        self.local_executor.update_parameters(
            visible_gpus=visible_gpus,
            gpus_per_node=len(visible_gpus),
            timeout_min=cfg.rollout_vllm.submitit_args.time_in_minutes,
        )
        local_executor_config = ExecutorConfig(
            name="local",
            executor=self.local_executor,
            num_replicas=1,
            concurrency=cfg.rollout_vllm.concurrency_per_replica,
        )

        slurm_executor = submitit.SlurmExecutor(
            folder=cfg.submitit_logs_folder, python=cfg.rollout_vllm.submitit_python
        )
        slurm_executor.update_parameters(
            **cfg.rollout_vllm.submitit_args.to_submitit_parameters()
        )

        self.rollout_executor_configs = [
            local_executor_config,
            ExecutorConfig(
                name="slurm",
                executor=slurm_executor,
                num_replicas=cfg.rollout_vllm.num_replicas,
                concurrency=cfg.rollout_vllm.concurrency_per_replica,
            ),
        ]
        self.llm_judge_executor_configs = [
            local_executor_config,
            ExecutorConfig(
                name="slurm",
                executor=slurm_executor,
                num_replicas=cfg.llm_judge_vllm.num_replicas,
                concurrency=cfg.llm_judge_vllm.concurrency_per_replica,
            ),
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)

    def get_checkpoint_paths(self, step_index: int) -> _CheckpointPaths:
        """Return checkpoint paths and recycle previous checkpoints if enabled.

        - At startup, use base_model as base policy and KL ref.
        - At the next step, use new output as base model, and base model as KL ref.
        - Subsequently, at the Nth step, use N-1 as base policy and N-2 as KL ref.
        """
        output_checkpoint = self.cfg.checkpoint_folder / f"step_{step_index:03d}"
        self.checkpoints.append(output_checkpoint)
        self.logger.info(f"Next checkpoint path: {output_checkpoint}")
        makedirs(output_checkpoint, exist_ok=True)

        # At start up, use base model for both KL and as base policy
        # At step 1, use base model for KL.
        checkpoints_including_base = [
            self.cfg.base_model,  # Initial KL
            self.cfg.base_model,  # Initial base policy
            *self.checkpoints,
        ]
        *_, _kl_ref, _current, _output = checkpoints_including_base

        # Delete all but the two most recent newly-created checkpoints (kl, current)
        # base model should not be deleted.
        if (not self.cfg.keep_all_checkpoints) and (len(self.checkpoints) > 2):
            *_to_delete, _kl_ref, _current = self.checkpoints
            for _folder in _to_delete:
                rmtree(_folder)

        return _CheckpointPaths(current=_current, output=_output, kl_ref=_kl_ref)

    async def generate_rollout(
        self,
        data: Sequence[RLVRDataItem],
        model_name: str | pathlib.Path,
        run_name: str,
    ) -> tuple[Sequence[agents.RunResult], Sequence[TraceID | None], Sequence[str]]:
        """Generate rollouts using policy model/agent."""
        if isinstance(model_name, pathlib.Path):
            model_name = model_name.as_posix()

        with SubmititVLLM(
            logger_name_prefix="submitit_vllm.policy",
            executor_configs=self.rollout_executor_configs,
            engine_args=EngineArgs(
                model=model_name,
                compilation_config=CompilationConfig(
                    cache_dir=self.cfg.rollout_vllm.cache_dir.as_posix()
                ),
                max_model_len=self.cfg.rollout_vllm.max_model_len,
            ),
        ) as _vllm:
            coros = [
                maybe_traced(
                    lambda _item=_item: _vllm.run_agent(self.policy_agent, _item.query),
                    _item.lf_dataset_client,
                    run_name=run_name,
                )
                for _item in data
            ]
            rollouts = await gather_with_progress(coros, "Rollout...")

        results = [_result for _result, _ in rollouts]
        trace_ids = [_trace_id for _, _trace_id in rollouts]
        answers = [_result.final_output for _result in results]

        return results, trace_ids, answers

    async def score_rollouts(
        self, data: Sequence[RLVRDataItem], answers: Sequence[str]
    ) -> Sequence[EvalResult]:
        """Score a list of answers, using LLM judge."""
        with SubmititVLLM(
            logger_name_prefix="submitit_vllm.evaluator",
            executor_configs=self.llm_judge_executor_configs,
            engine_args=EngineArgs(
                model=self.cfg.llm_judge_vllm.model_name,
                compilation_config=CompilationConfig(
                    cache_dir=self.cfg.llm_judge_vllm.cache_dir.as_posix()
                ),
                max_model_len=self.cfg.llm_judge_vllm.max_model_len,
            ),
        ) as evaluator_vllm:
            coros = [
                self.evaluator(_item, proposed=_answer, submitit_vllm=evaluator_vllm)
                for _item, _answer in zip(data, answers)
            ]
            return await gather_with_progress(coros, "LLM-Judge...")

    def calculate_advantage(
        self,
        data: Sequence[RLVRDataItem],
        run_results: "Sequence[agents.RunResult]",
        evals: Sequence[EvalResult],
    ) -> AdvantageData:
        """Calculate advantage given rollouts and eval results."""
        tokenized_reward_details = [
            RewardDetailTokenized.from_messages(
                translate_rollout(_run_result, _item.query, self.policy_agent).messages,
                reward=_eval.score,
                tokenizer=self.tokenizer,
                pad_to=self.cfg.hyperparameters.max_model_len,
            )
            for _item, _run_result, _eval in zip(data, run_results, evals)
        ]

        return AdvantageData.from_list_of_rewards(tokenized_reward_details)

    async def run_step(self, index: int, data: dict[str, list[RLVRDataItem]]):
        """Run one GRPO step."""
        _paths = self.get_checkpoint_paths(index)

        num_train = len(data["train"])
        data_all = data["train"] + data["test"]  # process both splits in one pass

        run_results, trace_ids, answers = await self.generate_rollout(
            data_all, _paths.current, run_name=f"step_{index:03d}"
        )
        evals = await self.score_rollouts(data=data_all, answers=answers)
        advantages_train = self.calculate_advantage(
            data_all[:num_train], run_results[:num_train], evals=evals[:num_train]
        )
        advantages_test = self.calculate_advantage(
            data_all[num_train:], run_results[num_train:], evals=evals[num_train:]
        )

        # Optional: attach score to LangFuse
        add_score(evals[num_train:], trace_ids[num_train:], "Accuracy")

        with spinner("Running GRPO backprop..."):
            metrics = self.local_executor.submit(
                lambda: grpo_optimization_step(
                    advantages_train,
                    current_policy_path=_paths.current,
                    kl_ref_path=_paths.kl_ref,
                    checkpoint_output_path=_paths.output,
                    optimizer_path=self.cfg.optimizer_folder,
                    hyperparameters=self.cfg.hyperparameters,
                )
            ).result()

        return metrics, advantages_test.avg_reward

    async def async_main(self):
        """Run full GRPO loop."""
        data = load_data(self.cfg.data)
        initialize_lf_dataset(
            data["test"], f"{self.run_name}-test", self.cfg.model_dump()
        )

        metrics: list[tuple[GRPOMetrics, float | None]] = []
        for _epoch in range(self.cfg.num_epochs):
            metrics.append(await self.run_step(_epoch, data))

        return metrics

    def __call__(self):
        """Launch the async main function."""
        set_up_logging()
        return asyncio.run(self.async_main())

    def checkpoint(self, *args, **kwargs) -> submitit.helpers.DelayedSubmission:
        """Save state and launch the same callable with the same arguments."""
        return super().checkpoint(*args, **kwargs)
