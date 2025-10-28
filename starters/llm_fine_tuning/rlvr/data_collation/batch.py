"""Utils for generating strongly-typed batches."""

from typing import Annotated, Generic, Iterator, Type, TypeVar

import numpy as np
import pydantic
from pydantic.json_schema import SkipJsonSchema

from starters.llm_fine_tuning.rlvr.data_collation.data_types import TypedBatch


T = TypeVar("T", bound=TypedBatch)


class FieldConfig(pydantic.BaseModel):
    """Field configs.

    Extra dimension(s) will be added after the sequence dimension.
    To add extra dimensions, specify the shape along these dimensions.
    For (batch, size, vocab), set extra_dimensions to (vocab,)
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    padding_value: float | int | bool
    dtype: Annotated[type, SkipJsonSchema[None]]
    extra_dimensions: tuple[int, ...] = ()


class TypedBatcher(Generic[T]):
    """Util for generating strongly typed batches."""

    def __init__(
        self,
        field_data: dict[str, list[list]], # TODO: allow for numpy/torch tensor input
        batch_model: Type[T],
        field_configs: dict[str, FieldConfig],
        batch_size: int,
        pad_to_length: int,
    ):
        self.field_data = field_data
        self.batch_model = batch_model
        self.batch_size = batch_size
        self.fields = field_configs

        self.pad_to_length = pad_to_length

        # Validate all fields have same number of sequences
        seq_lens = {k: len(seqs) for k, seqs in field_data.items()}
        assert len(set(seq_lens.values())) == 1, f"Inconsistent seq count: {seq_lens}"

        # Check for additional dimensions.
        token_data_shapes = {
            k: np.asarray(v[0][0]).shape for k, v in field_data.items()
        }
        extra_dimensions = {
            k: _cfg.extra_dimensions for k, _cfg in field_configs.items()
        }
        for k, _token_data_shape in token_data_shapes.items():
            if _token_data_shape != extra_dimensions[k]:
                msg = (
                    "Extra dimension mismatch. Did you mean to specify extra dimensions? "
                    f"token_data_shapes: {token_data_shapes} vs "
                    f"extra_dimensions from field_configs: {extra_dimensions}, "
                    f"mismatch at field '{k}'"
                )
                raise ValueError(msg)

    def __len__(self) -> int:
        """Estimate number of batches to return."""
        num_sequences = len(next(iter(self.field_data.values())))
        return (num_sequences + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[T]:
        """Yield strongly-typed batches."""
        # Initialize batch arrays
        batch_arrays = {
            name: np.full(
                # place extra dimensions after the token dimension.
                (self.batch_size, self.pad_to_length, *cfg.extra_dimensions),
                cfg.padding_value,
                dtype=cfg.dtype,
            )
            for name, cfg in self.fields.items()
        }
        attention_mask = np.zeros((self.batch_size, self.pad_to_length), dtype=bool)

        pos_in_batch = 0
        num_sequences = len(next(iter(self.field_data.values())))

        for i in range(num_sequences):
            # Copy data for all fields
            for name, arrays in batch_arrays.items():
                # actual sequence length, capped at pad_to_length
                seq_len = min(len(self.field_data[name][i]), self.pad_to_length)
                arrays[pos_in_batch, :seq_len] = self.field_data[name][i][:seq_len]
                attention_mask[pos_in_batch, :seq_len] = True

            pos_in_batch += 1

            if pos_in_batch == self.batch_size:
                yield self.batch_model(
                    **{k: v.copy() for k, v in batch_arrays.items()},
                    attention_mask=attention_mask.copy(),
                    num_valid=pos_in_batch,
                )

                # Reset batch arrays
                attention_mask.fill(False)
                for k, array in batch_arrays.items():
                    array.fill(self.fields[k].padding_value)

                pos_in_batch = 0

        if pos_in_batch > 0:
            yield self.batch_model(
                **{k: v.copy() for k, v in batch_arrays.items()},
                attention_mask=attention_mask.copy(),
                num_valid=pos_in_batch,
            )
