import tensorflow.compat.v1 as tf
from .graph_utils import extract_base_name


def graph_to_mermaid(graph_def: tf.GraphDef, direction: str = "TD") -> str:
    """
    Converts a GraphDef to a Mermaid flowchart.
    """
    lines = [f"graph {direction}"]

    for node in graph_def.node:
        # Node definition with shape
        clean_name = node.name.replace("/", "_").replace(":", "_")
        lines.append(f'    {clean_name}["{node.name}<br/>({node.op})"]')

        # Edges
        for inp in node.input:
            base_inp = extract_base_name(inp).replace("/", "_").replace(":", "_")
            if inp.startswith("^"):
                # Control dependency (dotted line)
                lines.append(f"    {base_inp} -. control .-> {clean_name}")
            else:
                lines.append(f"    {base_inp} --> {clean_name}")

    return "\n".join(lines)


def graph_to_dot(graph_def: tf.GraphDef) -> str:
    """
    Converts a GraphDef to Graphviz DOT format.
    """
    lines = [
        "digraph G {",
        '    node [shape=box, style="filled,rounded", color="#E1E4E8", fillcolor="#F6F8FA", fontname="Arial"];',
        "    edge [fontname=Arial, fontsize=10];",
    ]

    for node in graph_def.node:
        clean_name = f'"{node.name}"'
        label = f"{node.name}\\n({node.op})"
        lines.append(f'    {clean_name} [label="{label}"];')

        for inp in node.input:
            base_inp = f'"{extract_base_name(inp)}"'
            if inp.startswith("^"):
                lines.append(
                    f'    {base_inp} -> {clean_name} [style=dotted, label="control"];'
                )
            else:
                lines.append(f"    {base_inp} -> {clean_name};")

    lines.append("}")
    return "\n".join(lines)
