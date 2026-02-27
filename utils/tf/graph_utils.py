"""
Graph manipulation utility functions.

This module provides stateless utility functions for common graph operations,
including I/O and analysis, extracted from TFGraphOptimizer to improve modularity
and reusability.
"""

import os
import collections
import numpy as np
from typing import Dict, Set, Optional, List, Any

import tensorflow.compat.v1 as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from google.protobuf import text_format


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


def create_const_node(name: str, value, dtype=None, shape: list = None):
    """
    Creates a Const NodeDef with given value, dtype and shape.
    Handles scalars, lists, and numpy arrays.
    """
    dtype_map = {
        "float32": types_pb2.DT_FLOAT,
        "float64": types_pb2.DT_DOUBLE,
        "int32": types_pb2.DT_INT32,
        "int64": types_pb2.DT_INT64,
        "bool": types_pb2.DT_BOOL,
        "uint8": types_pb2.DT_UINT8,
        "int16": types_pb2.DT_INT16,
        "int8": types_pb2.DT_INT8,
    }
    # Reverse map for mapping integer enums to numpy dtypes
    tf_to_dtype_str = {v: k for k, v in dtype_map.items()}

    # Resolve dtype string
    if dtype is None:
        if isinstance(value, bool):
            dtype_str = "bool"
        elif isinstance(value, int):
            dtype_str = "int32"
        elif isinstance(value, float):
            dtype_str = "float32"
        elif hasattr(value, "dtype"):
            dtype_str = value.dtype.name
        else:
            dtype_str = "float32"
    elif hasattr(dtype, "name"):
        dtype_str = dtype.name
    elif isinstance(dtype, str):
        dtype_str = dtype
    elif isinstance(dtype, int):
        dtype_str = tf_to_dtype_str.get(dtype, "float32")
    else:
        dtype_str = "float32"

    # Resolve TF dtype
    lookup_str = dtype_str.lower()
    if lookup_str.startswith("dt_"):
        lookup_str = lookup_str[3:]
    tf_dtype = dtype_map.get(lookup_str, types_pb2.DT_FLOAT)

    # Create tensor proto
    np_array = np.array(value, dtype=np.dtype(dtype_str))
    tensor = tensor_util.make_tensor_proto(np_array, dtype=tf_dtype, shape=shape)

    node = node_def_pb2.NodeDef()
    node.op = "Const"
    node.name = name
    node.attr["dtype"].type = tf_dtype
    node.attr["value"].tensor.CopyFrom(tensor)
    return node


def make_int_attr(value: int) -> attr_value_pb2.AttrValue:
    """Creates an AttrValue with an integer value."""
    attr = attr_value_pb2.AttrValue()
    attr.i = value
    return attr


def make_type_attr(dtype) -> attr_value_pb2.AttrValue:
    """Creates an AttrValue with a type value."""
    attr = attr_value_pb2.AttrValue()
    attr.type = dtype
    return attr


def make_tensor_attr(value, dtype=None, shape=None) -> attr_value_pb2.AttrValue:
    """Creates an AttrValue with a tensor value."""
    attr = attr_value_pb2.AttrValue()
    np_array = np.array(value)
    if dtype:
        np_array = np_array.astype(np.dtype(dtype) if isinstance(dtype, str) else dtype)
    attr.tensor.CopyFrom(tensor_util.make_tensor_proto(np_array, shape=shape))
    return attr


def make_output_shapes_attr(shapes: List[List[int]]) -> attr_value_pb2.AttrValue:
    """
    Creates an AttrValue proto for _output_shapes.

    Args:
        shapes: List of shapes, where each shape is a list of integers.

    Returns:
        attr_value_pb2.AttrValue: The formatted attribute
    """
    attr = attr_value_pb2.AttrValue()
    for shape in shapes:
        shape_proto = attr.list.shape.add()
        for dim in shape:
            shape_proto.dim.add().size = dim
    return attr


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
        t = tensor_util.MakeNdarray(attr_proto.tensor)
        if np.isscalar(t) or t.ndim == 0:
            return t.item()
        return t
    # Fallback to the proto itself for complex types
    return attr_proto


def get_node_shape(node: tf.NodeDef) -> Optional[List[int]]:
    """Extracts shape from a node's attributes or tensor value."""
    if node is None:
        return None
    if "_output_shapes" in node.attr:
        shapes = node.attr["_output_shapes"].list.shape
        if shapes:
            return [dim.size for dim in shapes[0].dim]
    if "shape" in node.attr:
        return [dim.size for dim in node.attr["shape"].shape.dim]
    if node.op == "Const" and "value" in node.attr:
        tensor = node.attr["value"].tensor
        if tensor.HasField("tensor_shape"):
            return [d.size for d in tensor.tensor_shape.dim]
        # Fallback: try to get shape from actual tensor values if tensor_shape is missing
        try:
            return tensor_util.MakeNdarray(tensor).shape.tolist()
        except Exception:
            return None
    return None


def get_node_rank(node: tf.NodeDef) -> Optional[int]:
    """Returns the rank (number of dimensions) of a node."""
    shape = get_node_shape(node)
    return len(shape) if shape is not None else None


def get_node_dtype(
    node_or_name: Any, nodes: Optional[Dict[str, tf.NodeDef]] = None
) -> int:
    """Gets the data type (enum) of a node or node name."""
    node = None
    if isinstance(node_or_name, str):
        if nodes:
            node = nodes.get(extract_base_name(node_or_name))
    else:
        node = node_or_name

    if not node:
        return types_pb2.DT_FLOAT

    type_attrs = ["T", "dtype", "output_type", "DstT", "SrcT", "Tparams"]
    for attr_name in type_attrs:
        if attr_name in node.attr:
            dtype = node.attr[attr_name].type
            if dtype and dtype != types_pb2.DT_INVALID:
                return dtype

    if node.op == "Const" and "value" in node.attr:
        tensor = node.attr["value"].tensor
        if tensor.dtype and tensor.dtype != types_pb2.DT_INVALID:
            return tensor.dtype

    return types_pb2.DT_FLOAT


def is_scalar(node: tf.NodeDef) -> bool:
    """Returns True if the node's output is a scalar."""
    shape = get_node_shape(node)
    return shape == []


def is_const(node: tf.NodeDef, value: Any) -> bool:
    """Checks if a node is a Const with a specific value."""
    if node is None or node.op != "Const":
        return False
    val = get_attr_value(node.attr.get("value"))
    return np.all(np.equal(val, value))


def get_broadcast_shape(
    s1: Optional[List[int]], s2: Optional[List[int]]
) -> Optional[List[int]]:
    """Computes the result shape of broadcasting s1 and s2."""
    if s1 is None or s2 is None:
        return None
    if s1 == s2:
        return s1
    if not s1:
        return s2
    if not s2:
        return s1
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    result = []
    for i in range(max_len):
        d1 = s1[len1 - 1 - i] if i < len1 else 1
        d2 = s2[len2 - 1 - i] if i < len2 else 1
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            return None
    return result[::-1]


def extract_base_name(input_name: str) -> str:
    """
    Extract base node name from input (strip port and control marker).

    Args:
        input_name: Input name (may contain :port or ^ prefix)

    Returns:
        str: Cleaned base node name

    Examples:
        'node:0' -> 'node'
        '^control_dep' -> 'control_dep'
        'node:1' -> 'node'
        'node' -> 'node'
    """
    return input_name.split(":")[0].lstrip("^")


def canonicalize_axis(axis: Optional[int], rank: Optional[int]) -> Optional[int]:
    """
    Standardizes negative axes for easier comparison.

    Args:
        axis: Axis value (can be negative)
        rank: Tensor rank

    Returns:
        Non-negative axis value, or None if cannot canonicalize
    """
    if axis is None:
        return None
    if axis >= 0:
        return axis
    if rank is None:
        return None
    return axis + rank


def compute_reference_counts(graph_def: tf.GraphDef) -> Dict[str, int]:
    """
    Compute reference count for each node in the graph.

    Args:
        graph_def: The graph definition

    Returns:
        Dict mapping node names to their reference counts
    """
    reference_counts: Dict[str, int] = collections.defaultdict(int)
    for node in graph_def.node:
        for input_name in node.input:
            reference_counts[extract_base_name(input_name)] += 1
    return reference_counts


def update_node_inputs(
    node: tf.NodeDef, node_mapping: Dict[str, str], hoisted_controls: Set[str] = None
):
    """
    Update node's inputs based on node_mapping (old_name -> new_name).
    Preserves port numbers and control dependency markers.

    Args:
        node: Node to update
        node_mapping: Dict mapping old node names to new node names
        hoisted_controls: Optional set of control dependency names (e.g., '^node')
                         to append to this node.
    """
    updated_inputs = []
    existing_controls = set()
    for input_name in node.input:
        is_control = input_name.startswith("^")
        if is_control:
            existing_controls.add(input_name)

        base_name = extract_base_name(input_name)
        port = ""
        if not is_control and ":" in input_name:
            port = ":" + input_name.split(":", 1)[1]

        # Resolve transitively to handle multiple replacements in a single iteration
        target_base = base_name
        visited = {target_base}
        while target_base in node_mapping:
            target_base = node_mapping[target_base]
            if target_base in visited:  # Safeguard against circular mapping
                break
            visited.add(target_base)

        if target_base != base_name:
            new_input = f"^{target_base}" if is_control else f"{target_base}{port}"
            updated_inputs.append(new_input)
            if is_control:
                existing_controls.add(new_input)
        else:
            updated_inputs.append(input_name)

    # Add hoisted controls if they don't already exist
    if hoisted_controls:
        for ctrl in hoisted_controls:
            # Prevention: do not add self-control-dependency cycle
            if ctrl.lstrip("^") == node.name:
                continue
            if ctrl not in existing_controls:
                updated_inputs.append(ctrl)
                existing_controls.add(ctrl)

    del node.input[:]
    node.input.extend(updated_inputs)


def remove_nodes(
    graph_def: tf.GraphDef,
    nodes_to_remove: Set[str],
    pass_name: str = None,
    reason: str = None,
    logger=None,
) -> tf.GraphDef:
    """
    Create new GraphDef without specified nodes.

    Args:
        graph_def: Original graph
        nodes_to_remove: Set of node names to remove
        pass_name: Pass name for logging
        reason: Reason for removal (for logging)
        logger: Optional logger instance

    Returns:
        New GraphDef without the specified nodes
    """
    pruned_graph_def = tf.GraphDef()
    for node in graph_def.node:
        if node.name not in nodes_to_remove:
            pruned_graph_def.node.add().CopyFrom(node)

    if logger and nodes_to_remove:
        prefix = f"[{pass_name}] " if pass_name else ""
        reason_str = f", reason: {reason}" if reason else ""
        for node_name in nodes_to_remove:
            logger.debug(f"{prefix}Deleted: {node_name}{reason_str}")

    return pruned_graph_def


def prune_dead_nodes(
    graph_def: tf.GraphDef,
    pass_name: str = None,
    refs_before: Dict[str, int] = None,
    protected_nodes: Set[str] = None,
    logger=None,
) -> tf.GraphDef:
    """
    Remove nodes that became dead after optimization.

    Args:
        graph_def: The graph definition
        pass_name: Pass name for logging
        refs_before: Reference counts before optimization
        protected_nodes: Set of nodes that should not be pruned
        logger: Optional logger instance

    Returns:
        Pruned graph definition
    """
    refs_after = compute_reference_counts(graph_def)
    protected_nodes = protected_nodes or set()

    dead_nodes = set()
    for node in graph_def.node:
        if node.op == "Placeholder" or node.name in protected_nodes:
            continue

        if refs_before is None:
            # Full-cleanup mode (no scope): prune every zero-ref node,
            # including Const and non-Const alike.
            if refs_after[node.name] == 0:
                dead_nodes.add(node.name)
        else:
            # Scoped mode: only prune nodes that *this pass* is responsible for.
            # Always prune unreferenced Const nodes regardless of refs_before.
            if node.op == "Const" and refs_after[node.name] == 0:
                dead_nodes.add(node.name)
                continue
            # Prune non-Const nodes that became dead (whether they had refs
            # before or not, as long as the pass tracked them in refs_before).
            if node.name in refs_before and refs_after[node.name] == 0:
                dead_nodes.add(node.name)

    if dead_nodes:
        if logger:
            logger.info(
                f"[{pass_name or 'optimize'}] Pruning {len(dead_nodes)} dead nodes: {dead_nodes}"
            )
        return remove_nodes(
            graph_def, dead_nodes, pass_name, "dead node (ref_count=0)", logger
        )

    return graph_def


def final_prune(
    graph_def: tf.GraphDef,
    pass_name: str = None,
    protected_nodes: Set[str] = None,
    max_iterations: int = 100,
    logger=None,
) -> tf.GraphDef:
    """
    Final cleanup pass to remove all remaining dead nodes.
    Iteratively removes nodes with zero references until no more dead nodes exist.

    Args:
        graph_def: The graph definition
        pass_name: Pass name for logging
        protected_nodes: Set of nodes that should not be pruned
        max_iterations: Maximum cleanup iterations
        logger: Optional logger instance

    Returns:
        Pruned graph definition
    """
    protected_nodes = protected_nodes or set()
    iteration = 0
    total_removed = 0

    while iteration < max_iterations:
        refs = compute_reference_counts(graph_def)

        dead_nodes = {
            node.name
            for node in graph_def.node
            if node.op != "Placeholder"
            and node.name not in protected_nodes
            and refs[node.name] == 0
        }

        if not dead_nodes:
            break

        if logger:
            logger.info(
                f"[{pass_name or 'optimize'}] Final prune iteration {iteration + 1}: "
                f"removing {len(dead_nodes)} dead nodes"
            )

        graph_def = remove_nodes(
            graph_def, dead_nodes, pass_name, "final prune (ref_count=0)", logger
        )
        total_removed += len(dead_nodes)
        iteration += 1

    if logger:
        if iteration >= max_iterations:
            logger.warning(
                f"[{pass_name or 'optimize'}] Final prune reached max iterations ({max_iterations})"
            )
        elif total_removed > 0:
            logger.info(
                f"[{pass_name or 'optimize'}] Final prune: removed {total_removed} nodes "
                f"in {iteration} iteration(s)"
            )

    return graph_def


def check_external_consumers(
    consumers: Dict[str, list],
    replaced_nodes: list,
    all_replaced: Set[str],
    internal_names: Set[str],
) -> list:
    """
    Check if replaced nodes have external consumers (not in replaced set).

    Args:
        consumers: Consumer index (node_name -> list of consumer names)
        replaced_nodes: List of replaced node names to check
        all_replaced: Set of all replaced node names
        internal_names: Set of internal node names (part of the match)

    Returns:
        List of (node_name, external_consumers) tuples
    """
    nodes_with_ext_consumers = []
    for replaced_name in replaced_nodes:
        node_consumers = consumers.get(replaced_name, [])
        external_consumers = [
            c
            for c in node_consumers
            if c not in all_replaced and c not in internal_names
        ]
        if external_consumers:
            nodes_with_ext_consumers.append((replaced_name, external_consumers))
    return nodes_with_ext_consumers


def log_external_consumer_warning(nodes_with_ext_consumers: list, logger):
    """Log warning about nodes with external consumers."""
    logger.warning("Nodes marked as replaced still have external consumers:")
    for replaced_name, ext_consumers in nodes_with_ext_consumers[:3]:
        consumer_list = ", ".join(ext_consumers[:5])
        if len(ext_consumers) > 5:
            consumer_list += f" and {len(ext_consumers) - 5} more..."
        logger.warning(f"  - {replaced_name}: consumed by {consumer_list}")
    if len(nodes_with_ext_consumers) > 3:
        logger.warning(f"  ... and {len(nodes_with_ext_consumers) - 3} more nodes")


def build_consumer_index(graph_def: tf.GraphDef) -> Dict[str, list]:
    """
    Build consumer index from graph definition.

    Args:
        graph_def: The graph definition

    Returns:
        Dict mapping node names to lists of consumer node names
    """
    consumers = collections.defaultdict(list)
    for node in graph_def.node:
        for input_name in node.input:
            base_name = extract_base_name(input_name)
            consumers[base_name].append(node.name)
    return consumers
