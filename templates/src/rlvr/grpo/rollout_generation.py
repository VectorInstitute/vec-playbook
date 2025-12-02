"""Rollout Generator for RL use cases.

Usage:
PYTHONPATH="." uv run starters/llm_fine_tuning/rlvr/grpo/rollout_generation.py
"""

from typing import Callable

import agents
import backoff
import openai
import pydantic

from templates.src.rlvr.agents_integration.rollout_translation import (
    Rollout,
)
from templates.src.rlvr.grpo.data_types import RLVRDataItem
from templates.src.rlvr.langfuse import maybe_setup_langfuse_instrumentation
from templates.src.rlvr.submitit_vllm import SubmititVLLM


# Only if LangFuse API keys are set in env var.
maybe_setup_langfuse_instrumentation()


class EvalResult(pydantic.BaseModel):
    """Evaluation score, including explanation."""

    explanation: str
    score: float


class RewardDetail(pydantic.BaseModel):
    """_EvalResult plus full ChatCompletion/HF-compatible rollout info."""

    source_item: RLVRDataItem
    rollout: Rollout
    result: EvalResult


def split_reasoning(response: str) -> tuple[str, str]:
    """Extract reasoning tokens from response if matched."""
    reasoning = ""
    for _token in ("</think>",):
        if _token in response:
            reasoning, _, response = response.partition(_token)

    return reasoning, response


class RLVREvaluator:
    """Base class for stateful evaluator of verifiable rewards."""

    def __init__(
        self,
        evaluator_agent: "agents.Agent",
        split_reasoning: Callable[[str], tuple[str, str]] | None = split_reasoning,
    ):
        self.agent = evaluator_agent
        self.split_reasoning = split_reasoning

    @backoff.on_exception(backoff.expo, openai.APIConnectionError)
    async def __call__(
        self,
        item: RLVRDataItem,
        proposed: str,
        submitit_vllm: SubmititVLLM,
    ) -> EvalResult:
        """Evaluate one proposed response to one data item."""
        if self.split_reasoning:
            _, proposed = self.split_reasoning(proposed)

        query = f"Ground Truth: {item.model_dump_json(indent=2)} \nProposed: {proposed}"
        try:
            response = await submitit_vllm.run_agent(self.agent, query=query)
        except (openai.BadRequestError, agents.exceptions.ModelBehaviorError) as e:
            # Input is too long- judge as False.
            return EvalResult(explanation=str(e), score=0)

        return response.final_output_as(EvalResult)
