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
