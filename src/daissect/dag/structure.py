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
