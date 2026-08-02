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

    def insert(self, target: str, node: Node, loc: str) -> None:
        """
        Parameters:
            target (str): The name of the existing node to anchor the insertion.
            node (Node): The new Node instance to insert into the graph.
            loc (str): Insertion strategy; either "before" or "after" the target.


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
                    node.name if a == target else p for p in succ_node.predecessors
                )

            target_node.successors = [node.name]

        elif loc == "before":
            # Node inherits target's predecessors, and Node points only to target
            node.predecessors = tuple(target_node.predecessors)
            node.successors = [target]

            for p in target_node.predecessors:
                arg_node = self.nodes[p]
                arg_node.successors = [
                    node.name if u == target else u for u in arg_node.successors
                ]

            target_node.predecessors = (node.name,)
        else:
            raise ValueError(
                f"Invalid insertion strategy: '{loc}'." "Expected 'before' or 'after'."
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
            arg_node.successors = [
                u for u in arg_node.successors if u != target
            ] + target_node.successors

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
