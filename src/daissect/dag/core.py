import copy
from typing import Dict

from torch import nn

from .structure import Topology


class DAG:
    def __init__(self) -> None:
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
