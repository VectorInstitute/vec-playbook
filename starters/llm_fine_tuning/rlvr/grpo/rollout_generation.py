"""Rollout Generator for RL use cases.

Usage:
PYTHONPATH="." uv run starters/llm_fine_tuning/rlvr/grpo/rollout_generation.py
"""

import asyncio
import contextlib
import os
from typing import Any, Sequence

import agents
import pydantic

from starters.llm_fine_tuning.rlvr.agents.rollout_translation import (
    Rollout,
    translate_rollout,
)
from starters.llm_fine_tuning.rlvr.async_utils import gather_with_progress, rate_limited


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


class RLVREvaluator:
    """Base class for stateful evaluator of verifiable rewards."""

    def __init__(
        self,
        evaluator_agent: "agents.Agent",
        agent_run_config: "agents.RunConfig",
        semaphore: asyncio.Semaphore,
    ):
        self.agent = evaluator_agent
        self.semaphore = semaphore
        self.agent_run_config = agent_run_config

    async def evaluate(self, item: RLVRDataItem, proposed: str) -> RewardDetails:
        """Evaluate one proposed response to one data item."""
        import agents

        query = f"Ground Truth: {item.model_dump_json(indent=2)} \nProposed: {proposed}"
        async with self.semaphore:
            response = await agents.Runner.run(
                self.agent, input=query, run_config=self.agent_run_config
            )

        return response.final_output_as(RewardDetails)


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

    async def generate(
        self,
        data: Sequence[RLVRDataItem],
        agent_run_config: "agents.RunConfig",
        semaphore: asyncio.Semaphore,
    ) -> list[RewardDetails]:
        """Generate RLVR reward details on given data and policy.

        Specify LLM client and model name in agent_run_config.
        """
        import agents

        # Generate agent rollouts
        rollout_coros = [
            rate_limited(
                lambda _item=_item: agents.Runner.run(
                    self.agent, _item.query, run_config=agent_run_config
                ),
                semaphore=semaphore,
            )
            for _item in data
        ]

        outputs = await gather_with_progress(rollout_coros, description="Rollout ...")
        text_responses = [str(_result.final_output) for _result in outputs]

        # Translate Agent SDK rollouts to Chat Completion format
        full_rollouts = [
            translate_rollout(
                resp_obj=_output, user_text=_item.query, agent_obj=self.agent
            )
            for _item, _output in zip(data, outputs)
        ]

        # Obtain verifiable reward scores
        eval_coros = [
            self.evaluator.evaluate(item=_item, proposed=_response_text)
            for _item, _response_text in zip(data, text_responses)
        ]
        eval_results = await gather_with_progress(
            eval_coros, description="Evaluate ..."
        )

        return [
            RewardDetails(**_eval.model_dump(), source_item=_item, rollout=_rollout)
            for _item, _eval, _rollout in zip(data, eval_results, full_rollouts)
        ]


async def main():
    """Integration tests."""
    import openai

    from starters.llm_fine_tuning.rlvr.agents.examples import weather_agent

    client = openai.AsyncOpenAI()
    example_agent_config = agents.RunConfig(
        model=agents.OpenAIChatCompletionsModel(
            model="Qwen3-0.6B", openai_client=client
        )
    )
    evaluator = RLVREvaluator(
        eval_agent,
        agent_run_config=example_agent_config,
        semaphore=asyncio.Semaphore(1),
    )

    rollout_generator = GRPORollout(weather_agent, evaluator=evaluator)
    example_data = [
        RLVRDataItem(query="Weather in Shanghai", target="28 degrees Celsius, clear"),
        RLVRDataItem(query="Weather in Vancouver", target="10 degrees Celsius, cloudy"),
    ]
    rollouts = await rollout_generator.generate(
        data=example_data,
        agent_run_config=example_agent_config,
        semaphore=asyncio.Semaphore(1),
    )

    print([_rollout.model_dump() for _rollout in rollouts][0])


if __name__ == "__main__":
    asyncio.run(main())
