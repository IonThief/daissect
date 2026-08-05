from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
from torch import nn


@dataclass
class BaseResult:
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.data:
            return ""
        return ", ".join(f"{k}={v}" for k, v in self.data.items())


class BasePlugin(nn.Module):
    """
                         [ Input Tensor ]
                                │
                                ▼
    +--------------------------Plugin---------------------------+
    │  1. pre_forward()                                         │
    │           │                                               │
    │           ▼                                               │
    │  2. target_module()                                       │
    │           │                                               │
    │           ▼                                               │
    │  3. post_forward() ──(register_hook)──> 4. backward_hook()│
    │           │                                       ▲       │
    +-----------│---------------------------------------│-------+
                ▼                                       │
         [ Output Tensor ] ────────(loss.backward())────┘
    """

    def __init__(self, target_module: nn.Module) -> None:
        super().__init__()
        if not isinstance(target_module, nn.Module):
            raise TypeError(
                f"target_module must be an instance of nn.Module, "
                f"got {type(target_module)}"
            )

        self.target_module = target_module
        self._result = BaseResult()

    @property
    def result(self) -> BaseResult:
        return self._result

    def pre_forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Intercepts inputs before they reach the target_module"""
        return args, kwargs

    def post_forward(self, output: Any) -> Any:
        """Intercepts outputs"""
        return output

    def backward_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Triggered during loss.backward()"""
        return grad

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes pre_forward -> target_module -> post_forward.
        Attaches backward_hook if output requires_grad.
        """
        mod_args, mod_kwargs = self.pre_forward(*args, **kwargs)

        output = self.target_module(*mod_args, **mod_kwargs)

        output = self.post_forward(output)

        self._attach_backward_hook(output)

        return output

    def _attach_backward_hook(self, tensor_or_collection: Any) -> None:
        """
        Recursively searches the output for tensors that require gradients
        and attaches the backward_hook.
        """
        if isinstance(tensor_or_collection, torch.Tensor):
            if tensor_or_collection.requires_grad:
                tensor_or_collection.register_hook(self.backward_hook)

        elif isinstance(tensor_or_collection, (tuple, list)):
            for item in tensor_or_collection:
                self._attach_backward_hook(item)

        elif isinstance(tensor_or_collection, dict):
            for item in tensor_or_collection.values():
                self._attach_backward_hook(item)
