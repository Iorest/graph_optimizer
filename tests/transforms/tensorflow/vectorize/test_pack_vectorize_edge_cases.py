"""Edge case tests for PackVectorizePass (TensorFlow)."""

import pytest
import tensorflow.compat.v1 as tf
from graph_optimizer.core.tensorflow import TFGraphOptimizer
from graph_optimizer.utils.tf.graph_utils import create_node, make_output_shapes_attr
from graph_optimizer.transforms.tensorflow.vectorize import PackVectorizePass
from tensorflow.core.framework import attr_value_pb2

tf.disable_v2_behavior()


def _ph(name, shape, dtype=tf.float32):
    n = create_node("Placeholder", name)
    n.attr["shape"].shape.CopyFrom(tf.TensorShape(shape).as_proto())
    n.attr["dtype"].type = dtype.as_datatype_enum
    return n


def _pack(inputs, axis, out_shape):
    n = create_node("Pack", "pack", inputs=inputs)
    n.attr["axis"].i = axis
    n.attr["N"].i = len(inputs)
    if out_shape:
        n.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([out_shape]))
    return n


def _run(*nodes):
    gd = tf.GraphDef()
    gd.node.extend(nodes)
    opt = TFGraphOptimizer(gd)
    result = PackVectorizePass().transform(opt)
    return {n.name: n for n in result.node}, result


def test_mixed_op_types_fail():
    """Fail hoisting if branches have different op types."""
    x1 = _ph("x1", [10])
    x2 = _ph("x2", [10])
    r1 = create_node("Relu", "r1", inputs=["x1"])
    s2 = create_node("Sigmoid", "s2", inputs=["x2"])
    pk = _pack(["r1", "s2"], 0, [2, 10])
    node_map, result = _run(x1, x2, r1, s2, pk)
    # Should not hoist: Pack node should still exist with original inputs
    assert "pack" in node_map
    assert list(node_map["pack"].input) == ["r1", "s2"]


def test_attribute_mismatch_fails():
    """Fail hoisting if branches have different attributes (e.g., Cast to diff types)."""
    x1 = _ph("x1", [10])
    x2 = _ph("x2", [10])
    c1 = create_node("Cast", "c1", inputs=["x1"])
    c1.attr["DstT"].type = tf.float32.as_datatype_enum
    c2 = create_node("Cast", "c2", inputs=["x2"])
    c2.attr["DstT"].type = tf.float64.as_datatype_enum
    pk = _pack(["c1", "c2"], 0, [2, 10])
    node_map, result = _run(x1, x2, c1, c2, pk)
    # Should not hoist
    assert "pack" in node_map


def test_incompatible_broadcast_fails():
    """Fail hoisting if secondary inputs cannot be broadcasted or packed appropriately."""
    x1 = _ph("x1", [10])
    x2 = _ph("x2", [10])
    y1 = _ph("y1", [1, 5])  # Incompatible shapes
    y2 = _ph("y2", [1, 7])
    add1 = create_node("Add", "add1", inputs=["x1", "y1"])
    add2 = create_node("Add", "add2", inputs=["x2", "y2"])
    pk = _pack(["add1", "add2"], 0, [2, 10])
    node_map, result = _run(x1, x2, y1, y2, add1, add2, pk)
    # Shouldn't hoist because y1 and y2 have mismatched shapes that can't be packed
    assert "pack" in node_map


def test_strided_slice_elimination():
    """Test StridedSlice elimination when reversing a Pack operation."""
    x = _ph("x", [2, 10])
    b0 = create_node(
        "Const",
        "b0",
        attr={
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto([0], dtype=tf.int32)
            )
        },
    )
    e0 = create_node(
        "Const",
        "e0",
        attr={
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto([1], dtype=tf.int32)
            )
        },
    )
    s = create_node(
        "Const",
        "s",
        attr={
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto([1], dtype=tf.int32)
            )
        },
    )

    b1 = create_node(
        "Const",
        "b1",
        attr={
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto([1], dtype=tf.int32)
            )
        },
    )
    e1 = create_node(
        "Const",
        "e1",
        attr={
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto([2], dtype=tf.int32)
            )
        },
    )

    # slice_axis=0, shrink_axis_mask=1 (bit 0)
    sl1 = create_node("StridedSlice", "sl1", inputs=["x", "b0", "e0", "s"])
    sl1.attr["shrink_axis_mask"].i = 1
    sl2 = create_node("StridedSlice", "sl2", inputs=["x", "b1", "e1", "s"])
    sl2.attr["shrink_axis_mask"].i = 1

    pk = _pack(["sl1", "sl2"], 0, [2, 10])
    node_map, result = _run(x, b0, e0, b1, e1, s, sl1, sl2, pk)

    # Should eliminate: Pack -> x
    assert "pack" not in node_map
    assert "sl1" not in node_map
    assert "sl2" not in node_map


def test_bias_add_to_addv2():
    """Test BiasAdd conversion to AddV2 since generic vectorized AddV2 is preferred."""
    x1 = _ph("x1", [1, 10])
    x2 = _ph("x2", [1, 10])
    b1 = _ph("b1", [10])
    b2 = _ph("b2", [10])

    ba1 = create_node("BiasAdd", "ba1", inputs=["x1", "b1"])
    ba2 = create_node("BiasAdd", "ba2", inputs=["x2", "b2"])

    pk = _pack(["ba1", "ba2"], 0, [2, 1, 10])
    node_map, result = _run(x1, x2, b1, b2, ba1, ba2, pk)

    # Needs to be hoisted and converted to AddV2
    assert "pack" not in node_map
    addv2 = next((n for n in result.node if n.op == "AddV2"), None)
    assert addv2 is not None
