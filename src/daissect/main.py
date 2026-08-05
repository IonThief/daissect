from typing import Any, Dict, Optional

from .dag.core import DAG


def daissect(source: Any, concrete_args: Optional[Dict[str, Any]] = None) -> DAG:
    """
    Parses the input source into a locked DAG.

    Supports:
        - torch.nn.Module.
    """
    return DAG.from_source(source, concrete_args=concrete_args)
