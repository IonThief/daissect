import copy
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

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

    def find(
        self,
        pattern: Optional[str] = None,
        module_type: Optional[Type[nn.Module]] = None,
    ) -> List[str]:
        """
        Queries the topology for matching nodes.
        """
        matches = list(self._topology.nodes.keys())

        if pattern is not None:
            regex_str = f"^{pattern.replace('*', '.*')}$"
            try:
                regex = re.compile(regex_str)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

            matches = [n for n in matches if regex.match(n)]

        if module_type is not None:
            type_matches = []
            for name in matches:
                mod = self._module_pool.get(name)
                if mod is not None and isinstance(mod, module_type):
                    type_matches.append(name)
            matches = type_matches

        return matches
