"""Types for RLVR."""

from typing import Annotated, Any, Type, TypeVar

import numpy as np
import pydantic
import torch
from pydantic.json_schema import SkipJsonSchema


T = TypeVar("T", bound=pydantic.BaseModel)


class TypedBatch(pydantic.BaseModel):
    """A batch of data with dynamic fields."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    attention_mask: Annotated[torch.Tensor, SkipJsonSchema[None]]
    num_valid: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate all arrays have same batch dimension
        batch_dims: dict[str, int] = {}
        for k, v in kwargs.items():
            if k == "num_valid":
                continue
            if isinstance(v, torch.Tensor):
                batch_dims[k] = v.shape[0]
            else:
                batch_dims[k] = len(v)
        assert len(set(batch_dims.values())) == 1, (
            f"Inconsistent batch dim: {batch_dims}"
        )

    def to_torch(self, device: torch.device, target_class: Type[T]) -> T:
        """Transfer tensors to torch.

        Params:
            device: target torch device.
            target_class: should be a Pydantic class
                where the fields are named the same but
                of type torch.Tensor in place of np.ndarray.
        """
        output_dict: dict[str, Any] = {}
        for field_name in self.__pydantic_fields__:
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                output_dict[field_name] = value.to(device)
            elif isinstance(value, np.ndarray):
                output_dict[field_name] = torch.as_tensor(value, device=device)
            else:
                output_dict[field_name] = value

        return target_class(**output_dict)
