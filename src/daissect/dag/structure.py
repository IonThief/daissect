"""
[ Input Data ]                [ Model Weights ]
         |                            |
         |                            |
         +------------+---------------+
                      |
                      v
           [ Matrix Multiplication ]
                      |
                      v
                  [ ReLU ]
                      |
                      v
                 [ Output ]

1. Input "x"
    +----------------------------------------------------------------------+
    | Node(name="x", type=OpType.PLACEHOLDER, operator="x")                |
    | predecessors = ()                                                    |
    | successors   = ["linear"]                                            |
    +----------------------------------------------------------------------+

2. Weights "w"
    +----------------------------------------------------------------------+
    | Node(name="w", type=OpType.GET_ATTR, operator="model.layer.weight")  |
    | predecessors = ()                                                    |
    | successors   = ["linear"]                                            |
    +----------------------------------------------------------------------+

3. Linear Math
    +----------------------------------------------------------------------+
    | Node(name="linear", type=OpType.CALL_FUNCTION, operator=torch.matmul)|
    | predecessors = ("x", "w")   <-- Order matters                        |
    | successors   = ["relu"]                                              |
    +----------------------------------------------------------------------+

4. Activation
    +----------------------------------------------------------------------+
    | Node(name="relu", type=OpType.CALL_FUNCTION, operator=torch.relu)    |
    | predecessors = ("linear",)                                           |
    | successors   = ["out"]                                               |
    +----------------------------------------------------------------------+

5. Output
    +----------------------------------------------------------------------+
    | Node(name="out", type=OpType.OUTPUT, operator="output")              |
    | predecessors = ("relu",)                                             |
    | successors   = []                                                    |
    +----------------------------------------------------------------------+
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union


class OpType(Enum):
    PLACEHOLDER = "placeholder"
    GET_ATTR = "get_attr"
    CALL_MODULE = "call_module"
    CALL_FUNCTION = "call_function"
    CALL_METHOD = "call_method"
    OUTPUT = "output"


@dataclass
class Node:
    """
    Parameters:
        name (str): Unique identifier for the node.
        type (OpType): The category of operation this node represents.
        operator (Union[str, Callable]): The actual function, method, or module to execute.
        predecessors (Tuple[str, ...]): Names of nodes providing input to this node.
        kwargs (Dict[str, Any]): Keyword arguments passed directly to the target.
        successors (List[str]): Names of nodes that consume this node's output.
    """

    name: str
    type: OpType
    operator: Union[str, Callable]
    predecessors: Tuple[str, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    successors: List[str] = field(default_factory=list)


class Topology:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}

    def insert(
        self,
        target: str,
        node: Node,
        loc: str,
        intercept_edge: str = None,
    ) -> None:
        """
        Inserts a node into the graph relative to a target node.

        Parameters:
            target (str): The name of the existing node to anchor the insertion.
            node (Node): The new Node instance to insert into the graph.
            loc (str): Insertion strategy; either "before" or "after" the target.
            intercept_edge (str, optional): When loc="before", specifies exactly which incoming
                                            edge to intercept. Required if target has multiple inputs.


        - Topology.insert(loc="after")

            [ ORIGINAL ]

                  +---------------+                     +---------------+
                  |  Target Node  |-------------------->|   Succ Node   |
                  |succ=["Succ"]  |                     |pred=["Target"]|
                  +---------------+                     +---------------+

            [ MODIFIED ]

                  +---------------+      +--------+     +---------------+
                  |  Target Node  |----->|NEW NODE|---->|   Succ Node   |
                  | succ=["NEW"]  |      |        |     | pred=["NEW"]  |
                  +---------------+      +--------+     +---------------+
                                         pred=["Target"]
                                         succ=["Succ"]


        - Topology.insert(loc="before")

            [ ORIGINAL ]

                  +---------------+                     +---------------+
                  |   Pred Node   |-------------------->|  Target Node  |
                  |succ=["Target"]|                     | pred=["Pred"] |
                  +---------------+                     +---------------+

            [ MODIFIED ]

                  +---------------+      +--------+     +---------------+
                  |   Pred Node   |----->|NEW NODE|---->|  Target Node  |
                  | succ=["NEW"]  |      |        |     | pred=["NEW"]  |
                  +---------------+      +--------+     +---------------+
                                         pred=["Pred"]
                                         succ=["Target"]

        """
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found.")
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists.")
        if loc not in ("before", "after"):
            raise ValueError("loc must be 'before' or 'after'.")

        self.nodes[node.name] = node
        target_node = self.nodes[target]

        if loc == "after":
            # Node inherits target's successors, and target points only to Node
            node.predecessors = (target,)
            node.successors = list(target_node.successors)

            for s in target_node.successors:
                succ_node = self.nodes[s]
                succ_node.predecessors = tuple(
                    node.name if p == target else p for p in succ_node.predecessors
                )

            target_node.successors = [node.name]

        elif loc == "before":
            if intercept_edge is None:
                if len(target_node.predecessors) == 1:
                    intercept_edge = target_node.predecessors[0]
                elif len(target_node.predecessors) > 1:
                    raise ValueError(
                        f"Target node '{target}' has multiple predecessors {target_node.predecessors}. "
                        "You must explicitly specify 'intercept_edge'."
                    )
                else:
                    raise ValueError(
                        f"Target node '{target}' has no predecessors to intercept."
                    )

            if intercept_edge not in target_node.predecessors:
                raise ValueError(
                    f"'{intercept_edge}' is not a valid predecessor of '{target}'."
                )

            # Node inherits ONLY the intercepted edge
            node.predecessors = (intercept_edge,)
            node.successors = [target]

            # Update the specific predecessor to point to the new node
            arg_node = self.nodes[intercept_edge]
            arg_node.successors = list(
                dict.fromkeys(
                    node.name if s == target else s for s in arg_node.successors
                )
            )

            # Update target to take the new node in place of the specific intercepted edge
            target_node.predecessors = tuple(
                node.name if p == intercept_edge else p
                for p in target_node.predecessors
            )

    def remove(self, target: str) -> None:
        """
        Removes a node and bridges its predecessors directly to its successors.

        Parameters:
            target (str): The name of the node to remove from the graph.

        - Topology.remove

            [ ORIGINAL ]

                  +---------------+      +--------+     +---------------+
                  |   Pred Node   |----->| TARGET |---->|   Succ Node   |
                  |succ=["Target"]|      |        |     | pred=["Target"]
                  +---------------+      +--------+     +---------------+


            [ MODIFIED ]

                  +---------------+                     +---------------+
                  |   Pred Node   |-------------------->|   Succ Node   |
                  | succ=["Succ"] |                     |  pred=["Pred"]|
                  +---------------+                     +---------------+
                                         +--------+
                                         | TARGET |
                                         | (Del)  |
                                         +--------+

        """
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found.")

        target_node = self.nodes[target]

        # Bridge predecessors to point to successors
        for p in target_node.predecessors:
            arg_node = self.nodes[p]
            raw_successors = [
                s for s in arg_node.successors if s != target
            ] + target_node.successors
            arg_node.successors = list(dict.fromkeys(raw_successors))

        # Bridge successors to point to predecessors
        for s in target_node.successors:
            succ_node = self.nodes[s]
            new_args = []
            for arg in succ_node.predecessors:
                if arg == target:
                    new_args.extend(target_node.predecessors)
                else:
                    new_args.append(arg)
            succ_node.predecessors = tuple(new_args)

        del self.nodes[target]

    def branch(
        self,
        target: str,
        branch_node: Node,
        merge_node: Node,
        target_is_first_arg: bool = True,
    ) -> None:
        """
        Creates a parallel branch from target and merges it back.

        Parameters:
            target (str): The name of the node where the branch begins.
            branch_node (Node): The new node executing in parallel with the main flow.
            merge_node (Node): The new node responsible for combining the main flow and branch.
            target_is_first_arg (bool): Defines execution order. If True -> merge(target, branch).
                                        If False -> merge(branch, target).

        - Topology.branch

            [ ORIGINAL ]
                  +---------------+                     +---------------+
                  |  Target Node  |-------------------->| Original Succ |
                  +---------------+                     +---------------+

            [ MODIFIED ]
                                    +---------------+
                               +--->|  Branch Node  |---+
                               |    +---------------+   |
                  +------------+--+                     v   +---------------+
                  |  Target Node  |------------------------>|  Merge Node   |---> Original Succ
                  +---------------+                         +---------------+
        """
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found.")
        if branch_node.name in self.nodes or merge_node.name in self.nodes:
            raise ValueError("Branch or merge node already exists.")

        target_node = self.nodes[target]
        original_successors = list(target_node.successors)

        # Branch node connects from target to merge
        branch_node.predecessors = (target,)
        branch_node.successors = [merge_node.name]

        if target_is_first_arg:
            merge_node.predecessors = (target, branch_node.name)
        else:
            merge_node.predecessors = (branch_node.name, target)

        merge_node.successors = original_successors

        # Update original users to receive input from merge node instead of target
        for s in original_successors:
            succ_node = self.nodes[s]
            succ_node.predecessors = tuple(
                merge_node.name if p == target else p for p in succ_node.predecessors
            )

        # Update target to output to both branch and merge
        target_node.successors = [branch_node.name, merge_node.name]

        self.nodes[branch_node.name] = branch_node
        self.nodes[merge_node.name] = merge_node

    def sort(self) -> List[str]:
        """
        Topologically sort using Kahn's algorithm.

        Returns:
            List[str]: A sequence of node names ordered such that for every
                       directed edge u -> v, node u comes before v.

        Raises:
            RuntimeError: If the graph contains a cycle.

        - Topology.sort

            [ Queue ] -> Pops Node with 0 incoming edges (In-Degree = 0)
               |
               v
            Adds to sorted_nodes -> Decrements In-Degree of successors
               |
               v
            Cycle Detected if (len(sorted_nodes) != len(nodes))
        """
        in_degree = {
            name: len(set(node.predecessors)) for name, node in self.nodes.items()
        }
        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            current = queue.pop(0)
            sorted_nodes.append(current)
            for user in self.nodes[current].successors:
                in_degree[user] -= 1
                if in_degree[user] == 0:
                    queue.append(user)

        if len(sorted_nodes) != len(self.nodes):
            raise RuntimeError("Cycle detected in topology.")

        return sorted_nodes

    def has_cycles(self) -> bool:
        """
        Verifies if the graph structure constitutes a valid Directed Acyclic Graph (DAG).

        Returns:
            bool: True if at least one cycle exists (invalid DAG), False otherwise.
        """
        try:
            self.sort()
            return False
        except RuntimeError:
            return True
