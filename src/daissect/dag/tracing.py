from typing import Any, Dict, Optional, Set, Tuple, Type

from torch import fx, nn


class ControlFlowDetected(Exception):
    """
    Exception raised when data-dependent control flow prevents tracing inside a submodule.
    """

    def __init__(self, module_type: Type[nn.Module]) -> None:
        self.module_type = module_type
        super().__init__(f"Control flow detected in submodule {module_type.__name__}.")


class SymbolicTracer(fx.Tracer):
    """
    Extended FX Tracer that isolates control flow into leaf nodes.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.blacklist_leaves: Set[Type[nn.Module]] = set()

    def trace(
        self,
        root: nn.Module,
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> fx.Graph:
        if root is None:
            raise ValueError("Root module for tracing cannot be None.")

        while True:
            try:
                self.graph = fx.Graph(tracer_cls=type(self))
                return super().trace(root, concrete_args=concrete_args)

            except ControlFlowDetected as e:
                if e.module_type in self.blacklist_leaves:
                    raise RuntimeError(
                        "Unresolvable trace failure: "
                        f"'{e.module_type.__name__}' was already "
                        "blacklisted as a leaf, but tracing still failed."
                    ) from e

                # We found a new module causing issues. Blacklist it and try again.
                self.blacklist_leaves.add(e.module_type)

            except fx.proxy.TraceError as e:
                raise RuntimeError(
                    "Data-dependent control flow detected "
                    f"in the root module '{type(root).__name__}'. "
                    "The root module itself cannot be blacklisted as a leaf."
                ) from e

    def is_leaf_module(self, m: nn.Module, module_qualname: str) -> bool:
        if type(m) in self.blacklist_leaves:
            return True
        if not hasattr(m, "__module__"):
            return True
        return super().is_leaf_module(m, module_qualname)

    def call_module(
        self,
        m: nn.Module,
        forward: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        try:
            return super().call_module(m, forward, args, kwargs)
        except fx.proxy.TraceError:
            raise ControlFlowDetected(type(m))
