"""Utils for generating strongly-typed batches."""

from typing import Annotated, Generic, Iterator, Type, TypeVar

import pydantic
import torch
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
        field_data: dict[str, list[list] | torch.Tensor],
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
        token_data_shapes = {k: _infer_token_shape(v) for k, v in field_data.items()}
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
        field_devices = {
            name: _infer_device(self.field_data[name]) for name in self.fields
        }

        # Initialize batch tensors on a per-field device
        batch_arrays = {}
        for name, cfg in self.fields.items():
            torch_dtype = _torch_dtype(cfg.dtype)
            batch_arrays[name] = torch.full(
                (self.batch_size, self.pad_to_length, *cfg.extra_dimensions),
                cfg.padding_value,
                dtype=torch_dtype,
                device=field_devices[name],
            )

        mask_device = field_devices.get(
            "input_ids", next(iter(field_devices.values()), torch.device("cpu"))
        )
        attention_mask = torch.zeros(
            (self.batch_size, self.pad_to_length), dtype=torch.bool, device=mask_device
        )

        pos_in_batch = 0
        num_sequences = len(next(iter(self.field_data.values())))

        for i in range(num_sequences):
            # Copy data for all fields
            for name, arrays in batch_arrays.items():
                torch_dtype = arrays.dtype
                seq_source = self.field_data[name]
                seq = seq_source[i]
                # actual sequence length, capped at pad_to_length
                seq_len = min(len(seq), self.pad_to_length)
                seq_tensor = _coerce_to_tensor(
                    seq, dtype=torch_dtype, device=arrays.device
                )
                arrays[pos_in_batch, :seq_len] = seq_tensor[:seq_len]
                attention_mask[pos_in_batch, :seq_len] = True

            pos_in_batch += 1

            if pos_in_batch == self.batch_size:
                yield self.batch_model(
                    **{k: v.clone() for k, v in batch_arrays.items()},
                    attention_mask=attention_mask.clone(),
                    num_valid=pos_in_batch,
                )

                # Reset batch arrays
                attention_mask.fill_(False)
                for k, array in batch_arrays.items():
                    array.fill_(self.fields[k].padding_value)

                pos_in_batch = 0

        if pos_in_batch > 0:
            yield self.batch_model(
                **{k: v.clone() for k, v in batch_arrays.items()},
                attention_mask=attention_mask.clone(),
                num_valid=pos_in_batch,
            )


def _torch_dtype(python_dtype: type | torch.dtype) -> torch.dtype:
    if isinstance(python_dtype, torch.dtype):
        return python_dtype

    mapping: dict[type, torch.dtype] = {
        bool: torch.bool,
        int: torch.long,
        float: torch.float32,
    }

    if python_dtype in mapping:
        return mapping[python_dtype]

    msg = f"Unsupported dtype for torch conversion: {python_dtype}"
    raise TypeError(msg)


def _infer_device(field_value: object) -> torch.device:
    if isinstance(field_value, torch.Tensor):
        return field_value.device

    if isinstance(field_value, (list, tuple)) and field_value:
        first = field_value[0]
        if isinstance(first, torch.Tensor):
            return first.device

    return torch.device("cpu")


def _infer_token_shape(field_value: object) -> tuple[int, ...]:
    if isinstance(field_value, torch.Tensor):
        return tuple(field_value.shape[2:])

    if isinstance(field_value, (list, tuple)) and field_value:
        first_sequence = field_value[0]
        if isinstance(first_sequence, torch.Tensor):
            return tuple(first_sequence.shape[1:])

        if isinstance(first_sequence, (list, tuple)) and first_sequence:
            first_token = first_sequence[0]
            if isinstance(first_token, torch.Tensor):
                return tuple(first_token.shape)

            if hasattr(first_token, "__len__") and not isinstance(
                first_token, (str, bytes)
            ):
                try:
                    return (len(first_token),)
                except TypeError:
                    return ()

    return ()


def _coerce_to_tensor(
    sequence: object, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if isinstance(sequence, torch.Tensor):
        return sequence.to(device=device, dtype=dtype)

    return torch.tensor(sequence, dtype=dtype, device=device)
