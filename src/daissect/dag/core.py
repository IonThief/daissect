import copy
import re
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from torch import nn

from .fx_types import NodePlaceholder, OpModule
from .structure import Node, OpType, Topology


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

    @require_mutable
    def insert(
        self,
        target: str,
        name: str,
        module: nn.Module,
        loc: str,
    ) -> "DAG":
        """Inserts a new module into the graph relative to a target"""
        node = Node(name=name, type=OpType.CALL_MODULE, operator=name)
        self._topology.insert(target, node, loc=loc)
        self._module_pool[name] = module

        return self

    @require_mutable
    def remove(self, target: str) -> "DAG":
        """Removes a module from the graph and bridges its connections"""
        self._topology.remove(target)
        if target in self._module_pool:
            del self._module_pool[target]

        return self

    @require_mutable
    def replace(self, target: str, module: nn.Module) -> "DAG":
        """Replaces an existing module in the graph"""
        if target not in self._topology.nodes:
            raise ValueError(f"Target node '{target}' not found in topology.")

        self._module_pool[target] = module
        node = self._topology.nodes[target]
        node.type = OpType.CALL_MODULE
        node.operator = target

        return self

    @require_mutable
    def wrap(self, target: str, wrapper_cls: Type[nn.Module], **kwargs: Any) -> "DAG":
        """
        Wraps an existing module with a custom wrapper class
        NOTE: The wrapper_cls must accept the original module as its first argument
        """
        if target not in self._module_pool:
            raise ValueError(
                f"Target '{target}' has no module to wrap (might be a placeholder/output)."
            )

        original_module = self._module_pool[target]
        wrapped_module = wrapper_cls(original_module, **kwargs)
        self._module_pool[target] = wrapped_module

        return self

    @require_mutable
    def branch(
        self,
        target: str,
        name: str,
        module: nn.Module,
        merge_fn: Callable,
    ) -> "DAG":
        """Creates a parallel execution branch and merges it back into the main flow"""
        branch_node = Node(name=name, type=OpType.CALL_MODULE, operator=name)
        merge_name = f"{name}_merge"

        merge_module = OpModule(
            target=merge_fn,
            args_schema=(NodePlaceholder(), NodePlaceholder()),
        )
        merge_node = Node(
            name=merge_name,
            type=OpType.CALL_FUNCTION,
            operator=merge_name,
        )

        self._topology.branch(target, branch_node, merge_node)

        self._module_pool[name] = module
        self._module_pool[merge_name] = merge_module

        return self
