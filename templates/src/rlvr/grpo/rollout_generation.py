"""Rollout Generator for RL use cases.

Usage:
PYTHONPATH="." uv run starters/llm_fine_tuning/rlvr/grpo/rollout_generation.py
"""

import contextlib
import os
from typing import Any, Callable

import agents
import backoff
import openai
import pydantic

from templates.src.rlvr.agents_integration.rollout_translation import (
    Rollout,
)
from templates.src.rlvr.submitit_vllm import SubmititVLLM


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
