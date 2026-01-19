"""
Shared types.

With reference to github.com/willccbb/verifiers
"""

from typing import TYPE_CHECKING, Annotated

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import SkipValidation


# typing aliases
if TYPE_CHECKING:
    ChatMessage = ChatCompletionMessageParam
else:
    # Use SkipValidation to preserve arbitrary keys like `tool_calls` as raw dicts
    # without Pydantic coercion that can break downstream consumption by
    # transformers' chat templates.
    ChatMessage = Annotated[ChatCompletionMessageParam, SkipValidation]
