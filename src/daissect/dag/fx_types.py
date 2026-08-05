from typing import Any, Tuple

from torch import nn
from torch.utils import _pytree as pytree


class NodePlaceholder:
    """Used to identify input locations in PyTree schemas"""

    def __repr__(self) -> str:
        return "<NodePlaceholder>"


class OpModule(nn.Module):
    """Wraps functions and methods into standard nn.Modules"""

    def __init__(
        self,
        target: Any,
        args_schema: Any = None,
        kwargs_schema: Any = None,
        is_method: bool = False,
    ) -> None:
        super().__init__()
        if target is None:
            raise ValueError("Target function or method cannot be None.")

        self.target = target
        self.is_method = is_method

        self.args_flat, self.args_spec = pytree.tree_flatten(args_schema or tuple())
        self.kwargs_flat, self.kwargs_spec = pytree.tree_flatten(
            kwargs_schema or dict()
        )

        self._arg_placeholders = [
            i for i, x in enumerate(self.args_flat) if isinstance(x, NodePlaceholder)
        ]
        self._kwarg_placeholders = [
            i for i, x in enumerate(self.kwargs_flat) if isinstance(x, NodePlaceholder)
        ]

    def bind_inputs(self, inputs: Tuple[Any, ...]) -> Tuple[Any, Any]:
        """Maps incoming positional arguments to their designated placeholders."""
        input_iter = iter(inputs)

        resolved_args_flat = list(self.args_flat)
        for idx in self._arg_placeholders:
            try:
                resolved_args_flat[idx] = next(input_iter)
            except StopIteration:
                raise RuntimeError("Insufficient inputs provided for args schema.")

        resolved_kwargs_flat = list(self.kwargs_flat)
        for idx in self._kwarg_placeholders:
            try:
                resolved_kwargs_flat[idx] = next(input_iter)
            except StopIteration:
                raise RuntimeError("Insufficient inputs provided for kwargs schema.")

        resolved_args = pytree.tree_unflatten(resolved_args_flat, self.args_spec)
        resolved_kwargs = pytree.tree_unflatten(resolved_kwargs_flat, self.kwargs_spec)

        return resolved_args, resolved_kwargs

    def forward(self, *inputs: Any) -> Any:
        resolved_args, resolved_kwargs = self.bind_inputs(inputs)

        if self.is_method:
            if not resolved_args:
                raise RuntimeError(
                    "Method call requires an object instance as the first argument."
                )
            obj, *rest_args = resolved_args
            return getattr(obj, self.target)(*rest_args, **resolved_kwargs)

        return self.target(*resolved_args, **resolved_kwargs)
