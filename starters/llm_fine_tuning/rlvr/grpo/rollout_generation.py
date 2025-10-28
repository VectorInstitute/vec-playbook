"""Rollout Generator for RL use cases.

Usage:
PYTHONPATH="." uv run starters/llm_fine_tuning/rlvr/grpo/rollout_generation.py
"""

import contextlib
import logging
import os
from typing import Any, Callable, Sequence

import agents
import backoff
import openai
import pydantic

from starters.llm_fine_tuning.rlvr.agents_integration.rollout_translation import (
    Rollout,
    translate_rollout,
)
from starters.llm_fine_tuning.rlvr.async_utils import gather_with_progress
from starters.llm_fine_tuning.rlvr.submitit_vllm import SubmititVLLM


# Optional: Instrument OpenAI Agents SDK with Langfuse via Logfire (OTel)
def _setup_langfuse_instrumentation(service_name: str = "rlvr-grpo") -> None:
    """Configure Logfire to instrument OpenAI Agents and initialize Langfuse.

    Follows guidance from Langfuse docs on OpenAI Agents SDK instrumentation.
    This is a no-op if required packages or credentials are missing.
    """
    # Only attempt setup if Langfuse credentials seem present
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return

    try:
        import logfire  # type: ignore
    except Exception:
        return

    try:
        # Configure logfire to export spans via OTLP (Langfuse client handles export)
        logfire.configure(
            service_name=os.getenv("LANGFUSE_SERVICE_NAME", service_name),
            send_to_logfire=False,
        )
        # Patch OpenAI Agents SDK to emit spans
        logfire.instrument_openai_agents()

        # Initialize Langfuse client (reads env vars for auth and host)
        from langfuse import get_client  # type: ignore

        _lf = get_client()
        # Optionally validate credentials; do not raise to avoid breaking flows
        with contextlib.suppress(Exception):
            _ = _lf.auth_check()
    except Exception:
        # Swallow all errors to keep core functionality intact
        return


# Perform best-effort setup at import time. This is safe and no-ops if disabled.
_setup_langfuse_instrumentation()


class RLVRDataItem(pydantic.BaseModel):
    """One row in the RLVR training dataset."""

    query: str
    target: str | float
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class _EvalResult(pydantic.BaseModel):
    """Evaluation score, including explanation."""

    explanation: str
    score: float


class RewardDetails(_EvalResult):
    """_EvalResult plus full ChatCompletion/HF-compatible rollout info."""

    source_item: RLVRDataItem
    rollout: Rollout


EVAL_AGENT_INSTRUCTIONS = """\
Evaluate if the `proposed` response matches the ground-truth `target`.

Give a score of 0.0 if incorrect and 1.0 if correct.
"""

# Eval agent must support structured output and use output_type _EvalResult
eval_agent = agents.Agent(
    "Evaluator", instructions=EVAL_AGENT_INSTRUCTIONS, output_type=_EvalResult
)


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
        submitit_vllm: SubmititVLLM,
        split_reasoning: Callable[[str], tuple[str, str]] | None = split_reasoning,
    ):
        self.agent = evaluator_agent
        self.split_reasoning = split_reasoning
        self.submitit_vllm = submitit_vllm

    @backoff.on_exception(backoff.expo, openai.APIConnectionError)
    async def evaluate(
        self,
        item: RLVRDataItem,
        proposed: str,
    ) -> _EvalResult:
        """Evaluate one proposed response to one data item."""
        import agents

        if self.split_reasoning:
            _, proposed = self.split_reasoning(proposed)

        query = f"Ground Truth: {item.model_dump_json(indent=2)} \nProposed: {proposed}"
        async with self.submitit_vllm.get_oai_agents_config() as run_config:
            try:
                response = await agents.Runner.run(
                    self.agent, input=query, run_config=run_config
                )
            except openai.BadRequestError:
                # Input is too long- judge as False.
                return _EvalResult(explanation="<openai.BadRequestError>", score=0)

        return response.final_output_as(_EvalResult)


class GRPORollout:
    """GRPO Rollout Generation and reward calculation.

    TODO: create a base class interface based on this class.
    """

    def __init__(
        self,
        agent: "agents.Agent",
        evaluator: RLVREvaluator,
    ):
        self.agent = agent
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)

    @backoff.on_exception(backoff.expo, openai.APIConnectionError)
    async def _run_one(
        self, data_item: RLVRDataItem, submitit_vllm: SubmititVLLM
    ) -> RewardDetails:
        """Run and evaluate on one data item.

        Handles async get_run_config.
        """
        # Generate agent rollouts
        async with submitit_vllm.get_oai_agents_config() as run_config:
            result = await agents.Runner.run(
                self.agent, data_item.query, run_config=run_config
            )

        # Evaluate
        eval_result = await self.evaluator.evaluate(
            item=data_item,
            proposed=str(result.final_output),
        )

        # Add raw rollout texts to output
        try:
            full_rollout = translate_rollout(
                resp_obj=result.final_output,
                user_text=data_item.query,
                agent_obj=self.agent,
            )
            return RewardDetails(
                **eval_result.model_dump(), source_item=data_item, rollout=full_rollout
            )

        except (
            TypeError,
            pydantic.ValidationError,
            agents.exceptions.ModelBehaviorError,
        ) as e:
            self.logger.info(e)
            return RewardDetails(
                explanation=f"Exception: {e}",
                score=0,
                source_item=data_item,
                rollout=full_rollout,
            )

    async def generate(
        self,
        data: Sequence[RLVRDataItem],
        submitit_vllm: SubmititVLLM,
    ) -> Sequence[RewardDetails]:
        """Generate RLVR reward details on given data and policy.

        Specify LLM client and model name in agent_run_config.
        """
        coros = [self._run_one(_item, submitit_vllm) for _item in data]
        return await gather_with_progress(coros, description="Rollout ...")
