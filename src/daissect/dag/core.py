import copy
from functools import wraps
from typing import Any, Callable, Dict

from torch import nn

from .structure import Topology


def require_mutable(func: Callable) -> Callable:
    """Decorator to enforce that the DAG is unlocked before mutation."""

    @wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if self._is_locked:
            raise RuntimeError(
                f"Cannot call '{func.__name__}': DAG is locked. "
                "Mutations must occur inside a 'with dag:' block."
            )
        return func(self, *args, **kwargs)

    return wrapper


class DAG:
    def __init__(self, is_locked: bool = True) -> None:
        self._is_locked = is_locked
        self._topology = Topology()
        self._module_pool: Dict[str, nn.Module] = {}

    def clone(self) -> "DAG":
        """Returns a shallow copy: duplicates topology, shares weights"""
        new_dag = DAG(is_locked=True)
        new_dag._topology.nodes = copy.deepcopy(self._topology.nodes)
        new_dag._module_pool = self._module_pool
        return new_dag

    def deepcopy(self) -> "DAG":
        """Returns a deep copy: duplicates topology and tensor weights"""
        new_dag = DAG(is_locked=True)
        new_dag._topology.nodes = copy.deepcopy(self._topology.nodes)
        new_dag._module_pool = copy.deepcopy(self._module_pool)
        return new_dag

    def lock(self) -> None:
        if self._topology.has_cycles():
            raise RuntimeError("Cannot lock DAG: Cycle detected in topology")
        self._is_locked = True

    def __enter__(self) -> "DAG":
        """Unlocks the graph"""
        self._is_locked = False
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Automatically locks the graph upon exiting the context block"""
        self.lock()
