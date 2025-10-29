"""Data types for GRPO."""

from typing import Annotated, Any, Sequence

import numpy as np
import pydantic
import torch
from pydantic.json_schema import SkipJsonSchema
from transformers import PreTrainedTokenizerFast

from starters.llm_fine_tuning.rlvr.data_collation.batch import (
    FieldConfig,
    TypedBatcher,
)
from starters.llm_fine_tuning.rlvr.data_collation.data_types import TypedBatch
from starters.llm_fine_tuning.rlvr.shared_types import ChatMessage


def _assistant_char_spans(
    messages: Sequence[ChatMessage], formatted: str
) -> list[tuple[int, int]]:
    """Find [start, end) character spans of assistant contents inside formatted string.

    We search sequentially to disambiguate duplicate substrings across turns.
    If a piece isn't found (e.g., template alters content), it is skipped.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content: str = message.get("content", "")  # type: ignore
        if not content:
            continue
        idx = formatted.find(content, cursor)
        if idx != -1:
            spans.append((idx, idx + len(content)))
            cursor = idx + len(content)
    return spans


class RewardDetailTokenized(pydantic.BaseModel):
    """Reward details for one rollout, outcome-supervised only.

    Attention mask is not included.
    Rather, input_ids should be truncated on both sides.
    """

    input_ids: list[int]
    loss_mask: list[bool]
    reward: float

    def __init__(self, input_ids: list[int], loss_mask: list[bool], **kwargs):
        """Verify input_ids and loss_mask shape."""
        input_ids_np = np.asarray(input_ids)
        loss_mask_np = np.asarray(loss_mask)

        _shape = input_ids_np.shape
        assert len(_shape) == 1, f"Flat list of indices required; got {_shape}"
        assert loss_mask_np.shape == _shape, "Loss mask shape must match input_ids."

        super().__init__(input_ids=input_ids, loss_mask=loss_mask, **kwargs)

    @staticmethod
    def from_messages(
        messages: list[ChatMessage],
        reward: float,
        tokenizer: PreTrainedTokenizerFast,
        pad_to: int,
    ) -> "RewardDetailTokenized":
        """Tokenize a chat and create a per-token boolean loss mask."""
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is None:
            raise ValueError(f"Template is required but not available in: {tokenizer}")
        formatted = tokenizer.apply_chat_template(
            messages,  # type: ignore[arg-type]
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(formatted, str)
        add_special_tokens = False  # per HF docs when tokenizing the formatted string

        assistant_spans = _assistant_char_spans(messages, formatted)

        # TODO: Verify HF typing
        enc: Any = tokenizer(
            formatted,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,  # requires FastTokenizer
            truncation=True,
            padding="max_length",
            max_length=pad_to,
        )
        input_ids: list[int] = enc["input_ids"]
        offsets: list[tuple[int, int]] = enc["offset_mapping"]

        # One bool per token.
        # Mark as True if the token span is completely in any assistant span
        loss_mask: list[bool] = [
            any(span_a <= _a < _b <= span_b for span_a, span_b in assistant_spans)
            for _a, _b in offsets
        ]

        return RewardDetailTokenized(
            input_ids=input_ids, loss_mask=loss_mask, reward=reward
        )


class _AdvantageDetail(RewardDetailTokenized):
    """Reward details plus group-relative advantage info.

    Important:
        advantages is a per-token list for extensibility.
        However, in outcome supervision formulations (e.g., DeepSeek Math)
        advantages is the same number repeated for all tokens.

    """

    advantages: list[float]  # per-token, same length as input_ids.


class PerTokenProbs(pydantic.BaseModel):
    """Per-token probability output stored as dense tensors.

    Attributes
    ----------
    full:
        Tensor of shape (batch, max_len, vocab_size) with next-token distributions.
    selected:
        Tensor of shape (batch, max_len) containing probabilities for taken tokens.
    attention_mask:
        Boolean tensor of shape (batch, max_len) indicating valid token positions.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    full: torch.Tensor
    selected: torch.Tensor
    attention_mask: torch.Tensor

    def __add__(self, other: "PerTokenProbs") -> "PerTokenProbs":
        """Concatenate along the batch axis (first dimension)."""
        if not isinstance(other, PerTokenProbs):
            return NotImplemented

        if self.attention_mask.shape[1:] != other.attention_mask.shape[1:]:
            msg = (
                "Cannot concatenate PerTokenProbs with mismatched token dimensions. "
                f"left={tuple(self.attention_mask.shape)} "
                f"right={tuple(other.attention_mask.shape)}"
            )
            raise ValueError(msg)

        return PerTokenProbs(
            full=torch.cat((self.full, other.full), dim=0),
            selected=torch.cat((self.selected, other.selected), dim=0),
            attention_mask=torch.cat(
                (self.attention_mask, other.attention_mask), dim=0
            ),
        )

    def to(self, device: torch.device) -> "PerTokenProbs":
        """Return copy where tensors are on the specified device."""
        return PerTokenProbs(
            full=self.full.to(device),
            selected=self.selected.to(device),
            attention_mask=self.attention_mask.to(device),
        )

    def __radd__(self, other: object) -> "PerTokenProbs":
        """Support sum(..., other=0)."""
        if isinstance(other, int) and other == 0:
            return self

        if isinstance(other, PerTokenProbs):
            return other + self

        return NotImplemented


class BatchForInference(TypedBatch):
    """Typed batch for getting probs."""

    # (batch, input_tokens)
    input_ids: Annotated[torch.Tensor, SkipJsonSchema[None]]


class BatchForGRPO(TypedBatch):
    """Typed batch for GRPO training."""

    # (batch, input_tokens)
    input_ids: Annotated[torch.Tensor, SkipJsonSchema[None]]
    loss_masks: Annotated[torch.Tensor, SkipJsonSchema[None]]
    per_token_advantage: Annotated[torch.Tensor, SkipJsonSchema[None]]


class AdvantageData(pydantic.BaseModel):
    """Batch of outputs for group-relative advantage estimate."""

    advantage_details: list[_AdvantageDetail]

    def _group_rewards(self) -> list[float]:
        """All rewards in this group."""
        return [_detail.reward for _detail in self.advantage_details]

    @staticmethod
    def from_list_of_rewards(
        reward_details: list[RewardDetailTokenized],
    ) -> "AdvantageData":
        """Compute advantage for a given batch of rewards."""
        if len(reward_details) == 0:
            msg = "reward_details must be a non-empty list, but an empty list is given."
            raise ValueError(msg)

        # per-trace only, not token-level
        rewards = np.asarray([_detail.reward for _detail in reward_details])
        advantages_np: np.ndarray = np.asarray(
            (rewards - rewards.mean()) / rewards.std()
        )
        advantages: list[float] = advantages_np.flatten().tolist()

        advantage_details: list[_AdvantageDetail] = []
        for _reward_detail, _advantage in zip(reward_details, advantages):
            # Same (sequence-level) advantage for each token.
            _per_token_advantages = [_advantage] * len(_reward_detail.input_ids)
            _advantage_detail = _AdvantageDetail(
                **_reward_detail.model_dump(), advantages=_per_token_advantages
            )
            advantage_details.append(_advantage_detail)

        return AdvantageData(advantage_details=advantage_details)

    def get_iterator_for_training(
        self, batch_size: int, pad_to_length: int
    ) -> TypedBatcher[BatchForGRPO]:
        """Obtain batched iterator for training."""
        return TypedBatcher(
            field_data={
                "input_ids": [
                    _advantage.input_ids for _advantage in self.advantage_details
                ],
                "loss_masks": [
                    _advantage.loss_mask for _advantage in self.advantage_details
                ],
                "per_token_advantage": [
                    _advantage.advantages for _advantage in self.advantage_details
                ],
            },
            batch_model=BatchForGRPO,
            field_configs={
                "input_ids": FieldConfig(padding_value=0, dtype=int),
                "loss_masks": FieldConfig(padding_value=False, dtype=bool),
                "per_token_advantage": FieldConfig(padding_value=0, dtype=float),
            },
            batch_size=batch_size,
            pad_to_length=pad_to_length,
        )

    @property
    def avg_reward(self) -> float | None:
        """Return average reward of rollouts in advantage_details."""
        rewards = [_advantage.reward for _advantage in self.advantage_details]
        if len(rewards) > 0:
            return sum(rewards) / len(rewards)

        return None


GRPOBatcher = TypedBatcher[BatchForGRPO]


class GRPOMetrics(pydantic.BaseModel):
    """Metrics for GRPO."""

    avg_loss: float | None = None
    grad_norm: float | None = None


class GRPOHyperparameters(pydantic.BaseModel):
    """GRPO Hyperparameters.

    TODO: divide into performance-related parameters and
    parameters related to numerical values.
    """

    max_model_len: int
    train_batch_size: int

    # for forward pass only, and not for vLLM rollout.
    inference_batch_size: int

    learning_rate: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_weight_decay: float = 0.0
