"""Data types for GRPO."""

from typing import TYPE_CHECKING, Annotated, Any, Sequence

import numpy as np
import pydantic
from pydantic.json_schema import SkipJsonSchema
from transformers import PreTrainedTokenizerFast

from starters.llm_fine_tuning.rlvr.data_collation.batch import (
    FieldConfig,
    TypedBatcher,
)
from starters.llm_fine_tuning.rlvr.data_collation.data_types import TypedBatch
from starters.llm_fine_tuning.rlvr.shared_types import ChatMessage


if TYPE_CHECKING:
    import torch


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
            truncation=False,
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
    """Per-token probability output.

    If input_id is int array of shape (batch, length, vocab_size), then
    - "full" would be float array of shape (batch, length - 1, vocab_size)
    - "selected" would be float array of shape (batch, length - 1)
    """

    full: list[list[list[float]]]
    selected: list[list[float]]

    def __add__(self, other: "PerTokenProbs") -> "PerTokenProbs":
        """Concatenate along the batch axis (first dimension)."""
        return PerTokenProbs(
            full=self.full + other.full,
            selected=self.selected + other.selected,
        )

    def __radd__(self, other: object) -> "PerTokenProbs":
        """Support sum(..., other=0)."""
        if not isinstance(other, PerTokenProbs):
            return self

        return self + other

    @staticmethod
    def from_batch(
        attention_mask: np.ndarray,
        num_valid: int,
        full: np.ndarray,
        selected: np.ndarray,
    ):
        """Slice batch and extract non-padding probs only.

        The outputs (full and selected) both start at the second token.
        The first token in attention mask is not included in the output.

        Params:
        -------
            attention_mask: bool array of shape (batch, input_tokens)
            num_valid: keep up to the first num_valid items.
                (assume that the following rows are paddings.)
            full: float array of shape (batch, input_tokens - 1, vocab)
            selected: float array of shape (batch, input_tokens - 1)
        """
        batch_size, num_input_tokens = attention_mask.shape
        if not (
            (full.shape[:2] == (batch_size, num_input_tokens - 1))
            and (selected.shape == (batch_size, num_input_tokens - 1))
        ):
            msg = (
                f"attention_mask.shape {attention_mask.shape} "
                "(batch, num_gen_tokens) did not match "
                f"full.shape {full.shape} (batch, num_gen_tokens - 1, vocab) "
                f"selected.shape {selected.shape} (batch, num_gen_tokens - 1)."
            )
            raise ValueError(msg)

        # Exclude first input token, which is not part of any output.
        attention_mask_shifted = attention_mask[:, 1:]

        return PerTokenProbs(
            full=[
                np.compress(_mask, _full, axis=0).tolist()
                for _mask, _full in zip(
                    attention_mask_shifted[:num_valid], full[:num_valid]
                )
            ],
            selected=[
                np.compress(_mask, _selected, axis=0).tolist()
                for _mask, _selected in zip(
                    attention_mask_shifted[:num_valid], selected[:num_valid]
                )
            ],
        )


class BatchForInference(TypedBatch):
    """Typed batch for getting probs."""

    # (batch, input_tokens)
    input_ids: Annotated[np.ndarray, SkipJsonSchema[None]]


class BatchForGRPO(TypedBatch):
    """Typed batch for GRPO training."""

    # (batch, input_tokens)
    input_ids: Annotated[np.ndarray, SkipJsonSchema[None]]
    loss_masks: Annotated[np.ndarray, SkipJsonSchema[None]]
    per_token_advantage: Annotated[np.ndarray, SkipJsonSchema[None]]

    # (batch, input_tokens - 1, vocab)
    pi_selected_ref: Annotated[np.ndarray, SkipJsonSchema[None]]
    pi_selected_base: Annotated[np.ndarray, SkipJsonSchema[None]]


class BatchForGRPOTorch(pydantic.BaseModel):
    """Typed batch for GRPO training, but in Torch format."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    num_valid: int
    attention_mask: Annotated["torch.Tensor", SkipJsonSchema[None]]

    # (batch, input_tokens) for all of the following
    # int/long
    input_ids: Annotated["torch.Tensor", SkipJsonSchema[None]]

    # bool
    loss_masks: Annotated["torch.Tensor", SkipJsonSchema[None]]

    # float
    per_token_advantage: Annotated["torch.Tensor", SkipJsonSchema[None]]
    pi_selected_ref: Annotated["torch.Tensor", SkipJsonSchema[None]]
    pi_selected_base: Annotated["torch.Tensor", SkipJsonSchema[None]]


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

    def get_iterator_for_inference(
        self, batch_size: int, pad_to_length: int | None = None
    ) -> TypedBatcher[BatchForInference]:
        """Obtain batched iterator for inference."""
        return TypedBatcher(
            field_data={
                "input_ids": [
                    _advantage.input_ids for _advantage in self.advantage_details
                ]
            },
            batch_model=BatchForInference,
            field_configs={"input_ids": FieldConfig(padding_value=0, dtype=int)},
            batch_size=batch_size,
            pad_to_length=pad_to_length,
        )


class GRPOData(AdvantageData):
    """Advantage data, plus probabilities from ref policy and optionally base policy."""

    ref_probs: PerTokenProbs
    base_probs: PerTokenProbs

    def get_iterator_for_training(
        self, batch_size: int, pad_to_length: int | None = None
    ) -> TypedBatcher[BatchForGRPO]:
        """Obtain batched iterator for training."""
        probability_selected_field = FieldConfig(padding_value=0, dtype=float)
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
                "pi_selected_ref": self.ref_probs.selected,
                "pi_selected_base": self.base_probs.selected,
            },
            batch_model=BatchForGRPO,
            field_configs={
                "input_ids": FieldConfig(padding_value=0, dtype=int),
                "loss_masks": FieldConfig(padding_value=False, dtype=bool),
                "per_token_advantage": FieldConfig(padding_value=0, dtype=float),
                "pi_selected_ref": probability_selected_field,
                "pi_selected_base": probability_selected_field,
            },
            batch_size=batch_size,
            pad_to_length=pad_to_length,
        )


GRPOBatcher = TypedBatcher[BatchForGRPO]

class GRPOMetrics(pydantic.BaseModel):
    """Metrics for GRPO."""

    eval_advantage: float
    train_advantage: float
    train_loss: float | None = None
    grad_norm: float
