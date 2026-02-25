import tensorflow.compat.v1 as tf
from typing import Dict, Set

from ...utils.logger import logger as logging
from ...utils.graph_utils import (
    canonicalize_axis,
    build_consumer_index,
    compute_reference_counts,
    remove_nodes,
    prune_dead_nodes,
    final_prune,
    extract_base_name,
    update_node_inputs,
    check_external_consumers,
    log_external_consumer_warning,
)


def get_attr_value(attr_proto):
    """Unwraps a TensorFlow AttrValue proto into a Python literal."""
    if attr_proto is None:
        return None
    field = attr_proto.WhichOneof("value")
    if field == "s":
        return attr_proto.s.decode("utf-8")
    if field == "i":
        return attr_proto.i
    if field == "f":
        return attr_proto.f
    if field == "b":
        return attr_proto.b
    if field == "type":
        return attr_proto.type
    if field == "shape":
        return [dim.size for dim in attr_proto.shape.dim]
    if field == "tensor":
        from tensorflow.python.framework import tensor_util
        import numpy as np

        t = tensor_util.MakeNdarray(attr_proto.tensor)
        if np.isscalar(t) or t.ndim == 0:
            return t.item()
        return t
    # Fallback to the proto itself for complex types
    return attr_proto


class GraphState:
    """
    Mutable state container for representing the Graph IR during optimization.
    """

    def __init__(self, graph_def: tf.GraphDef):
        self.load_state(graph_def)

    def load_state(self, graph_def: tf.GraphDef):
        """Load graph state and rebuild consumer index."""
        self.graph_def = graph_def
        self.nodes = {node.name: node for node in graph_def.node}
        self.consumers = build_consumer_index(graph_def)

    def refresh_state(self):
        """Refresh internal state from current graph_def (after in-place modifications)."""
        self.nodes = {node.name: node for node in self.graph_def.node}
        self.consumers = build_consumer_index(self.graph_def)

    def get_node_attr(self, node_or_name, attr_name, default=None):
        node = (
            self.nodes.get(node_or_name)
            if isinstance(node_or_name, str)
            else node_or_name
        )
        if node is None or attr_name not in node.attr:
            return default
        return get_attr_value(node.attr[attr_name])

    def get_node_shape(self, node_or_name):
        node = (
            self.nodes.get(node_or_name)
            if isinstance(node_or_name, str)
            else node_or_name
        )
        if node is None:
            return None
        if "_output_shapes" in node.attr:
            shapes = node.attr["_output_shapes"].list.shape
            if shapes:
                return [dim.size for dim in shapes[0].dim]
        if "shape" in node.attr:
            return [dim.size for dim in node.attr["shape"].shape.dim]
        return None

    def get_node_rank(self, node_or_name):
        shape = self.get_node_shape(node_or_name)
        return len(shape) if shape is not None else None

    def canonicalize_axis(self, axis, rank):
        return canonicalize_axis(axis, rank)

    def compute_reference_counts(self, graph_def: tf.GraphDef = None) -> Dict[str, int]:
        return compute_reference_counts(graph_def or self.graph_def)

    def remove_nodes(self, graph_def, nodes_to_remove, pass_name=None, reason=None):
        return remove_nodes(graph_def, nodes_to_remove, pass_name, reason, logging)

    def prune_dead_nodes(
        self, graph_def, pass_name=None, refs_before=None, protected_nodes=None
    ):
        return prune_dead_nodes(
            graph_def, pass_name, refs_before, protected_nodes, logging
        )

    def final_prune(self, graph_def, pass_name=None, protected_nodes=None):
        return final_prune(graph_def, pass_name, protected_nodes, logger=logging)

    def prune(self, output_names):
        from tensorflow.python.framework import graph_util

        self.graph_def = graph_util.extract_sub_graph(self.graph_def, output_names)
        self.load_state(self.graph_def)
        return self.graph_def

    @staticmethod
    def _extract_base_name(input_name: str) -> str:
        return extract_base_name(input_name)

    def update_node_inputs(
        self,
        node: tf.NodeDef,
        node_mapping: Dict[str, str],
        hoisted_controls: Set[str] = None,
    ):
        update_node_inputs(node, node_mapping, hoisted_controls)

    def check_external_consumers(self, replaced_nodes, all_replaced, internal_names):
        return check_external_consumers(
            self.consumers, replaced_nodes, all_replaced, internal_names
        )

    @staticmethod
    def log_external_consumer_warning(nodes_with_ext_consumers):
        log_external_consumer_warning(nodes_with_ext_consumers, logging)
