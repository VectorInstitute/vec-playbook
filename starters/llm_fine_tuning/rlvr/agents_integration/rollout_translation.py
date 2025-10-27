"""Entrypoint for testing agents SDK integration.

Usage:
PYTHONPATH="." uv run starters/llm_fine_tuning/rlvr/agents/main.py
"""

import asyncio
from typing import Any, cast

import agents
import openai
import pydantic
import transformers
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from starters.llm_fine_tuning.rlvr.shared_types import ChatMessage


class Rollout(pydantic.BaseModel):
    """Rollout in ChatCompletion / HF-compatible format."""

    messages: list[ChatMessage]
    tools: list[dict[str, Any]]


def _introspect_tools(agent_obj: agents.AgentBase) -> list[dict[str, Any]]:
    """Convert registered agent tools to OpenAI-compatible tools schema.

    Best-effort introspection; silently falls back to no tools on failure.
    """
    tools_out: list[dict[str, Any]] = []
    try:
        agent_tools = getattr(agent_obj, "tools", [])
        for t in agent_tools or []:
            name = getattr(t, "name", None)
            params = getattr(t, "params_json_schema", None) or getattr(
                t, "parameters", None
            )
            desc = getattr(t, "description", None)
            if name and params:
                fn = {"name": name, "parameters": params}
                if isinstance(desc, str) and desc:
                    fn["description"] = desc
                tools_out.append({"type": "function", "function": fn})
    except Exception:
        tools_out = []
    return tools_out


def _derive_final_assistant_text(resp_obj: agents.RunResult) -> str:
    """Extract a plain assistant text as a fallback."""
    for attr in ("final_output", "output", "text"):
        if hasattr(resp_obj, attr):
            val = getattr(resp_obj, attr)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return str(resp_obj)


def _get_items_list(resp_obj: agents.RunResult) -> list[object] | None:
    """Return a list of new items from a run, if available."""
    for attr in ("new_items", "items"):
        if hasattr(resp_obj, attr):
            value = getattr(resp_obj, attr)
            if isinstance(value, list):
                return value
    return None


def _collect_call_id_to_name(items: list[object]) -> dict[str, str]:
    """Build a mapping from tool call id to function name."""
    call_id_to_name: dict[str, str] = {}
    for item in items:
        item_type = getattr(item, "type", None)
        if item_type != "tool_call_item":
            continue

        raw = getattr(item, "raw_item", None)
        call_id = getattr(raw, "call_id", None) if raw is not None else None
        # name is not needed for tool message shape

        if call_id is None and isinstance(raw, dict):
            call_id = raw.get("call_id")
            name = raw.get("name")

            if isinstance(call_id, str) and isinstance(name, str):
                call_id_to_name[call_id] = name

    return call_id_to_name


def _append_tool_output_messages(
    messages_out: list[ChatMessage],
    items: list[object],
    call_id_to_name: dict[str, str],
) -> None:
    """Append tool output messages to the chat messages list.

    Produces OpenAI-compatible tool messages including optional `name` and
    `tool_call_id` fields when available.
    """
    for item in items:
        if getattr(item, "type", None) != "tool_call_output_item":
            continue

        raw = getattr(item, "raw_item", None)
        call_id = getattr(raw, "call_id", None) if raw is not None else None
        output_text = getattr(raw, "output", None) if raw is not None else None
        # No need to extract name for tool messages

        if call_id is None and isinstance(raw, dict):
            call_id = raw.get("call_id")
            output_text = raw.get("output")

        if output_text is None:
            v = getattr(item, "output", None)
            if isinstance(v, str) and v.strip():
                output_text = v.strip()

        if isinstance(output_text, str) and output_text:
            # ChatCompletionToolMessageParam requires role, content, and tool_call_id
            tool_msg_typed: ChatCompletionToolMessageParam = {
                "role": "tool",
                "content": output_text,
                "tool_call_id": call_id or "",
            }
            messages_out.append(cast(ChatMessage, tool_msg_typed))


def _find_first_tool_output_index(items: list[object]) -> int | None:
    """Return the index of the first tool output item, if any."""
    for i, it in enumerate(items):
        if getattr(it, "type", None) == "tool_call_output_item":
            return i
    return None


def _collect_reason_text_before(items: list[object], end: int) -> list[str]:
    """Collect assistant "reason" texts before the first tool output."""
    parts: list[str] = []
    for it in items[:end]:
        if getattr(it, "type", None) != "message_output_item":
            continue
        try:
            from agents.items import ItemHelpers

            it_any: Any = it
            text = ItemHelpers.text_message_output(it_any)
        except Exception:
            text = None
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return parts


def _collect_function_tool_calls_before(
    items: list[object], end: int
) -> list[dict[str, Any]]:
    """Collect function tool calls before the first tool output as `tool_calls`."""
    out: list[dict[str, Any]] = []
    for it in items[:end]:
        if getattr(it, "type", None) != "tool_call_item":
            continue
        raw = getattr(it, "raw_item", None)
        name = getattr(raw, "name", None) if raw is not None else None
        arguments = getattr(raw, "arguments", None) if raw is not None else None
        _type = getattr(raw, "type", None) if raw is not None else None
        call_id = getattr(raw, "call_id", None) if raw is not None else None

        if isinstance(raw, dict):
            name = raw.get("name")
            arguments = raw.get("arguments")
            _type = raw.get("type")
            call_id = raw.get("call_id")

        if (
            _type == "function_call"
            and isinstance(name, str)
            and isinstance(arguments, str)
        ):
            entry: dict[str, Any] = {
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            if isinstance(call_id, str) and call_id:
                entry["id"] = call_id
            out.append(entry)
    return out


def _build_assistant_tool_invocation_message(items: list[object]) -> ChatMessage | None:
    """Build an assistant message that invokes tools via `tool_calls`.

    - Aggregates function tool calls from run items.
    - Includes preceding assistant text ("reason") before the first tool output, if any.
    - Note: HF chat templates (e.g., Qwen3) ignore `function_call` and expect
      `tool_calls`.
    """
    first_tool_output_idx = _find_first_tool_output_index(items)
    scan_upto = (
        first_tool_output_idx if first_tool_output_idx is not None else len(items)
    )

    reason_text_parts = _collect_reason_text_before(items, scan_upto)
    tool_calls = _collect_function_tool_calls_before(items, scan_upto)

    if not tool_calls:
        return None

    content_str = "\n".join(reason_text_parts).strip()
    asst_typed: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": content_str or None,
        "tool_calls": cast(
            "list[ChatCompletionMessageFunctionToolCallParam]", tool_calls
        ),
    }
    return cast(ChatMessage, asst_typed)


def translate_rollout(
    resp_obj: agents.RunResult, user_text: str, agent_obj: agents.AgentBase
) -> Rollout:
    """Build messages and tools suitable for HF chat templates.

    - Includes tool outputs and subsequent assistant messages when present.
    - Falls back to a simple user/assistant exchange otherwise.
    """
    messages_out: list[ChatMessage] = [
        cast(
            ChatMessage, ChatCompletionUserMessageParam(role="user", content=user_text)
        )
    ]

    tools_out = _introspect_tools(agent_obj)

    items = _get_items_list(resp_obj)
    if items:
        # Add assistant message that invokes tools (if any)
        assistant_invocation = _build_assistant_tool_invocation_message(items)
        if assistant_invocation is not None:
            messages_out.append(assistant_invocation)

        # Add tool output messages
        call_id_to_name = _collect_call_id_to_name(items)
        _append_tool_output_messages(messages_out, items, call_id_to_name)

    # Append final assistant response derived from run
    last_asst: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": _derive_final_assistant_text(resp_obj),
    }
    messages_out.append(cast(ChatMessage, last_asst))

    return Rollout(
        messages=messages_out,
        tools=tools_out,
    )


# Integration test
async def main():
    """Run async logic."""
    client = openai.AsyncOpenAI(api_key="EMPTY")
    run_config = agents.RunConfig(
        model=agents.OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)
    )
    return await agents.Runner.run(
        weather_agent, input="Weather in Auckland", run_config=run_config
    )


if __name__ == "__main__":
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    from starters.llm_fine_tuning.rlvr.agents_integration.examples import weather_agent

    MODEL_NAME = "Qwen3-0.6B"
    MODEL_PATH = f"/model-weights/{MODEL_NAME}"

    agent_response = asyncio.run(main())
    # Build messages and tools for the tokenizer, including tool outputs
    user_input = "Weather in Auckland"
    rollout = translate_rollout(agent_response, user_input, weather_agent)

    # Ensure output is serializable
    assert isinstance(rollout.model_dump_json(indent=2), str)

    # Verify roles in rollout
    rollout_roles = [_message["role"] for _message in rollout.messages]
    print(rollout_roles)
    print(rollout.model_dump_json(indent=2))

    assert rollout_roles == [
        "user",  # "input"
        "assistant",  # assistant invokes tool
        "tool",  # response from tool
        "assistant",  # message to user
    ], rollout_roles

    # Verify compatibility with HF tokenizer
    tokenizer: "PreTrainedTokenizerFast" = transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
    formatted = tokenizer.apply_chat_template(
        cast("list[dict[str, Any]]", rollout.messages),
        tools=cast(Any, (rollout.tools or None)),
        tokenize=False,
        add_generation_prompt=False,
    )
    assert isinstance(formatted, str)
    print(formatted)

    # Verify function call details appear in the tokenizer output
    # Previously, using `function_call` was ignored by Qwen3 chat template,
    # so the tool invocation JSON was missing from `formatted`.
    assert "<tool_call>" in formatted, "Missing <tool_call> block in template output"
    assert '"name": "get_weather"' in formatted, "Missing function name in tool_call"
    assert '"arguments":' in formatted, "Missing arguments field in tool_call"
    assert "Auckland" in formatted, "Missing argument value in tool_call"
