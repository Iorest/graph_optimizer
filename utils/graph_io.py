import os
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import node_def_pb2


def create_node(op, name, inputs=None, attr=None):
    """Creates a NodeDef proto."""
    node = node_def_pb2.NodeDef()
    node.op = op
    node.name = name
    if inputs:
        node.input.extend(inputs)
    if attr:
        for k, v in attr.items():
            node.attr[k].CopyFrom(v)
    return node


def save_graph(graph_def, path):
    """Saves a GraphDef proto to a file (binary or pbtxt)."""
    # Ensure output directory exists
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if path.endswith(".pbtxt"):
        from google.protobuf import text_format

        with open(path, "w") as f:
            f.write(text_format.MessageToString(graph_def))
    else:
        with open(path, "wb") as f:
            f.write(graph_def.SerializeToString())


def load_graph(path):
    """Loads a GraphDef proto from a file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")

    graph_def = tf.GraphDef()
    if path.endswith(".pbtxt"):
        from google.protobuf import text_format

        with open(path, "r") as f:
            text_format.Merge(f.read(), graph_def)
    else:
        with open(path, "rb") as f:
            graph_def.ParseFromString(f.read())
    return graph_def


class SubgraphBuilder:
    """Helper to build a set of nodes for replacement."""

    def __init__(self, name_prefix=""):
        self.nodes = []
        self.prefix = name_prefix

    def add_node(self, op, name, inputs=None, attr=None):
        full_name = self.prefix + name
        node = create_node(op, full_name, inputs, attr)
        self.nodes.append(node)
        return full_name

    def get_nodes(self):
        return self.nodes
