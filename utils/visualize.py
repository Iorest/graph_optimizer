import tensorflow.compat.v1 as tf
from typing import Set, Optional


def export_to_dot(
    graph_def: tf.GraphDef, highlight_nodes: Optional[Set[str]] = None
) -> str:
    """
    Exports a TensorFlow GraphDef to GraphViz DOT format.

    Args:
        graph_def: The GraphDef proto to export.
        highlight_nodes: Optional set of node names to highlight in the diagram.

    Returns:
        A string containing the DOT representation of the graph.
    """
    highlight_nodes = highlight_nodes or set()
    dot = ["digraph G {"]
    dot.append('  node [shape=box, style=filled, fillcolor=white, fontname="Courier"];')
    dot.append('  edge [fontname="Courier"];')

    for node in graph_def.node:
        color = "lightblue" if node.name in highlight_nodes else "white"
        label = f"{node.name}\\n({node.op})"
        dot.append(f'  "{node.name}" [label="{label}", fillcolor="{color}"];')

        for input_name in node.input:
            # Handle control dependencies
            is_control = input_name.startswith("^")
            base_input = input_name.lstrip("^").split(":")[0]

            style = "dashed" if is_control else "solid"
            color = "gray" if is_control else "black"
            arrowhead = "empty" if is_control else "normal"

            dot.append(
                f'  "{base_input}" -> "{node.name}" [style="{style}", color="{color}", arrowhead="{arrowhead}"];'
            )

    dot.append("}")
    return "\n".join(dot)


def save_dot(
    graph_def: tf.GraphDef, path: str, highlight_nodes: Optional[Set[str]] = None
):
    """Saves the DOT representation of a graph to a file."""
    dot_content = export_to_dot(graph_def, highlight_nodes)
    with open(path, "w") as f:
        f.write(dot_content)
