"""Types for RLVR."""

from typing import TYPE_CHECKING, Annotated, Type, TypeVar

import numpy as np
import pydantic
from pydantic.json_schema import SkipJsonSchema


if TYPE_CHECKING:
    import torch


T = TypeVar("T", bound=pydantic.BaseModel)


class TypedBatch(pydantic.BaseModel):
    """A batch of data with dynamic fields."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    attention_mask: Annotated[np.ndarray, SkipJsonSchema[None]]
    num_valid: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate all arrays have same batch dimension
        batch_dims = {k: len(v) for k, v in kwargs.items() if k != "num_valid"}
        assert (
            len(set(batch_dims.values())) == 1
        ), f"Inconsistent batch dim: {batch_dims}"

    def to_torch(self, device: "torch.device", target_class: Type[T]) -> T:
        """Transfer tensors to torch.

        Params:
            device: target torch device.
            target_class: should be a Pydantic class
                where the fields are named the same but
                of type torch.Tensor in place of np.ndarray.
        """
        import torch

        output_dict = {}
        for k, v in self.model_dump().items():
            if isinstance(v, np.ndarray):
                output_dict[k] = torch.Tensor(v).to(device)
            else:
                output_dict[k] = v

        return target_class(**output_dict)
