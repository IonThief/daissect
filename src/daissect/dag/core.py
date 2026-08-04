import copy
import re
import types
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from torch import fx, nn

from ..plugins.base import Plugin
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
    def __init__(
        self,
        original_module: Optional[nn.Module] = None,
        is_locked: bool = True,
    ) -> None:
        self._original_module = original_module
        self._is_locked = is_locked
        self._topology = Topology()
        self._module_pool: Dict[str, nn.Module] = {}

    def clone(self) -> "DAG":
        """Returns a shallow copy: duplicates topology, shares weights"""
        new_dag = DAG(original_module=self._original_module, is_locked=True)
        new_dag._topology.nodes = copy.deepcopy(self._topology.nodes)
        new_dag._module_pool = self._module_pool
        return new_dag

    def deepcopy(self) -> "DAG":
        """Returns a deep copy: duplicates topology and tensor weights"""
        new_dag = DAG(original_module=self._original_module, is_locked=True)
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

    def build(self) -> nn.Module:
        """
        Compiles the topology and injects the optimized FX graph
        into an instance of the original module.

        [ DAG Topology ]          [ Original nn.Module ]
              │                             │
              ▼                             ▼
        (Topological Sort)             (Deepcopy)
              │                             │
              ▼                             ▼
        [ Ordered Nodes ]         [ Original Instance ]
              │                             │
              ├─────────────────────────────┤ (add_module / Sync Weights)
              ▼                             │
        [ fx.Graph Construction ]           │
              │                             │
              ▼                             │
        [ fx.GraphModule ] ─────────────────┤
              │                             │
              ▼                             │
        [ Dynamic forward() ] ──────────────┤
                                            │
                                            ▼
                                [ Transparent Executable ]
        """
        ordered_nodes = self._topology.sort()

        if self._original_module is not None:
            root_module = copy.deepcopy(self._original_module)
        else:
            root_module = nn.Module()

        graph = fx.Graph()
        fx_nodes: Dict[str, fx.Node] = {}

        for name in ordered_nodes:
            meta = self._topology.nodes[name]

            if meta.type == OpType.INPUT:
                fx_nodes[name] = graph.placeholder(name)

            elif meta.type == OpType.OUTPUT:
                out_args = tuple(fx_nodes[arg] for arg in meta.predecessors)
                graph.output(out_args[0] if len(out_args) == 1 else out_args)

            else:
                inputs = tuple(fx_nodes[arg] for arg in meta.predecessors)
                mod = self._module_pool[name]

                if isinstance(mod, OpModule):
                    res_args, res_kwargs = mod.bind_inputs(inputs)
                    if mod.is_method:
                        fx_nodes[name] = graph.call_method(
                            mod.target, tuple(res_args), res_kwargs
                        )
                    else:
                        fx_nodes[name] = graph.call_function(
                            mod.target, tuple(res_args), res_kwargs
                        )
                else:
                    root_module.add_module(name, mod)
                    fx_nodes[name] = graph.call_module(name, args=inputs)

        compiled_graph_module = fx.GraphModule(root_module, graph)

        root_module.forward = types.MethodType(
            compiled_graph_module.__class__.forward,
            root_module,
        )

        root_module.code = compiled_graph_module.code

        return root_module

    def draw(
        self,
        filepath: Optional[str] = None,
        format: str = "ascii",
        theme: str = "light",
    ) -> Any:
        if format == "ascii":
            out = self._draw_ascii()
            if filepath:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(out)
            else:
                print(out)
            return out
        elif format == "mermaid":
            out = self._draw_mermaid(theme=theme)
            if filepath:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(out)
            return out
        elif format in ("svg", "graphviz"):
            return self._draw_graphviz(
                filepath, format="svg" if format == "svg" else "png"
            )
        else:
            raise ValueError(
                f"Unsupported draw format '{format}'. Use 'ascii', 'mermaid', or 'svg'."
            )

    def _get_node_label_data(self, node_key: str) -> Dict[str, Any]:
        meta = self._topology.nodes[node_key]
        is_input = meta.type == OpType.INPUT
        is_output = meta.type == OpType.OUTPUT

        mod = self._module_pool.get(node_key)

        is_plugin = isinstance(mod, Plugin)
        base_mod = getattr(mod, "target_module", mod) if is_plugin else mod

        if is_input:
            signature = "Input"
        elif is_output:
            signature = "Output"
        elif isinstance(base_mod, OpModule):
            target_name = getattr(base_mod.target, "__name__", str(base_mod.target))

            constants = []
            for arg in base_mod.args_flat + base_mod.kwargs_flat:
                if not isinstance(arg, NodePlaceholder):
                    val_str = str(arg)
                    if len(val_str) > 15:
                        val_str = val_str[:12] + "..."
                    constants.append(val_str)

            if constants:
                signature = f"{target_name}({', '.join(constants)})"
            else:
                signature = target_name

        elif base_mod is not None:
            signature = base_mod.__class__.__name__
        else:
            signature = "Unknown"

        plugin_name = mod.__class__.__name__ if is_plugin else None
        plugin_result = getattr(mod, "result", None) if is_plugin else None

        return {
            "is_input": is_input,
            "is_output": is_output,
            "is_plugin": is_plugin,
            "signature": signature,
            "plugin_name": plugin_name,
            "plugin_result": plugin_result,
        }

    def _draw_ascii(self) -> str:
        lines: List[str] = []
        visited = set()

        def dfs(node_name: str, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            data = self._get_node_label_data(node_name)

            label = f"{node_name} : {data['signature']}"
            if data["is_plugin"]:
                res_str = str(data["plugin_result"])
                arrow_str = f" -> {res_str}" if res_str else ""
                label += f"  [{data['plugin_name']}{arrow_str}]"

            lines.append(f"{prefix}{connector}{label}")

            if node_name in visited:
                lines[-1] += " (shared reference)"
                return
            visited.add(node_name)

            users = self._topology.nodes[node_name].successors
            for i, user in enumerate(users):
                extension = "    " if is_last else "│   "
                dfs(user, prefix + extension, i == (len(users) - 1))

        roots = [
            name for name, meta in self._topology.nodes.items() if not meta.predecessors
        ]

        lines.append("[ DAG Topology ]")
        for i, root in enumerate(roots):
            dfs(root, "", i == (len(roots) - 1))

        return "\n".join(lines)

    def _draw_graphviz(self, filepath: Optional[str], format: str) -> Any:
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "The 'graphviz' library is required for this backend. "
                "Install it using: pip install graphviz"
            )

        dot = graphviz.Digraph(comment="DAG Topology")
        dot.attr(
            rankdir="LR",
            splines="spline",
            nodesep="0.6",
            ranksep="0.8",
            fontname="Helvetica",
        )
        dot.attr("edge", color="#94a3b8", penwidth="1.5", arrowsize="0.8")

        for key in self._topology.sort():
            data = self._get_node_label_data(key)

            mod_label = (
                f"<BR/><FONT POINT-SIZE='10' COLOR='#64748b'>{data['signature']}</FONT>"
            )
            tag = ""
            stats_label = ""

            fillcolor, color, penwidth = "#f8fafc", "#cbd5e1", "1.5"

            if data["is_input"]:
                tag = "<BR/><FONT POINT-SIZE='9' COLOR='#166534'>[Input]</FONT>"
                fillcolor, color, penwidth = "#dcfce7", "#22c55e", "2"
            elif data["is_output"]:
                tag = "<BR/><FONT POINT-SIZE='9' COLOR='#1e3a8a'>[Output]</FONT>"
                fillcolor, color, penwidth = "#dbeafe", "#3b82f6", "3"

            if data["is_plugin"]:
                res_str = str(data["plugin_result"])
                arrow_str = f" &#8594; {res_str}" if res_str else ""
                stats_label = f"<BR/><FONT POINT-SIZE='9' COLOR='#b91c1c'>[{data['plugin_name']}{arrow_str}]</FONT>"
                mod_label = f"<BR/><FONT POINT-SIZE='10' COLOR='#64748b'>{data['signature']} (Wrapped)</FONT>"
                if not data["is_input"] and not data["is_output"]:
                    fillcolor, color, penwidth = "#fef08a", "#ca8a04", "2"

            label = f"<{key}{mod_label}{tag}{stats_label}>"
            dot.node(
                key,
                label=label,
                shape="rect",
                style="rounded,filled",
                fillcolor=fillcolor,
                color=color,
                penwidth=penwidth,
                fontname="Helvetica",
            )

        for key in self._topology.sort():
            meta = self._topology.nodes[key]
            for succ in meta.successors:
                dot.edge(key, succ)

        if filepath:
            dot.render(filepath, format=format, cleanup=True)

        return dot

    def _draw_mermaid(self, theme: str = "light") -> str:
        if theme == "dark":
            text_color = "color:#f8fafc;"
            styles = {
                "input": f"fill:#064e3b,stroke:#10b981,stroke-width:2px,{text_color}",
                "output": f"fill:#1e3a8a,stroke:#3b82f6,stroke-width:3px,{text_color}",
                "plugin": f"fill:#713f12,stroke:#eab308,stroke-width:2px,{text_color}",
                "default": f"fill:#0f172a,stroke:#475569,stroke-width:1.5px,{text_color}",
            }
            init_directive = "%%{init: {'theme': 'dark'}}%%"
        else:
            text_color = "color:#0f172a;"
            styles = {
                "input": f"fill:#dcfce7,stroke:#22c55e,stroke-width:2px,{text_color}",
                "output": f"fill:#dbeafe,stroke:#3b82f6,stroke-width:3px,{text_color}",
                "plugin": f"fill:#fef08a,stroke:#ca8a04,stroke-width:2px,{text_color}",
                "default": f"fill:#f8fafc,stroke:#cbd5e1,stroke-width:1.5px,{text_color}",
            }
            init_directive = "%%{init: {'theme': 'default'}}%%"

        lines = [
            init_directive,
            "graph LR",
            f"classDef input {styles['input']}",
            f"classDef output {styles['output']}",
            f"classDef plugin {styles['plugin']}",
            f"classDef default {styles['default']}",
        ]

        for key in self._topology.sort():
            data = self._get_node_label_data(key)

            label = f"<b>{key}</b><br><i>{data['signature']}</i>"
            if data["is_plugin"]:
                res_str = str(data["plugin_result"])
                arrow_str = f" &rarr; {res_str}" if res_str else ""
                plugin_text_color = "#fca5a5" if theme == "dark" else "#b91c1c"
                label += f"<br><font color='{plugin_text_color}'>[{data['plugin_name']}{arrow_str}]</font>"

            shape_start, shape_end = (
                ("([", "])") if data["is_input"] or data["is_output"] else ("[", "]")
            )
            lines.append(f'    {key}{shape_start}"{label}"{shape_end}')

            if data["is_input"]:
                lines.append(f"    class {key} input")
            elif data["is_output"]:
                lines.append(f"    class {key} output")
            elif data["is_plugin"]:
                lines.append(f"    class {key} plugin")
            else:
                lines.append(f"    class {key} default")

        lines.append("")
        for key in self._topology.sort():
            meta = self._topology.nodes[key]
            for succ in meta.successors:
                lines.append(f"    {key} --> {succ}")

        return "\n".join(lines)
