import tensorflow.compat.v1 as tf
import numpy as np
from graph_optimizer.core.tensorflow.tf_optimizer import TFGraphOptimizer
from graph_optimizer.utils.tf.graph_utils import create_node, create_const_node
from tensorflow.core.framework import attr_value_pb2, types_pb2


def test_cse_list_attributes():
    """Test CSE with list attributes (list_i, list_f, etc.) to cover cse.py gaps."""
    # OP with list_i attribute
    attrs1 = {
        "list_i": attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(i=[1, 2, 3])
        )
    }
    node1 = create_node("TestOp", "node1", inputs=[], attr=attrs1)

    attrs2 = {
        "list_i": attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(i=[1, 2, 3])
        )
    }
    node2 = create_node("TestOp", "node2", inputs=[], attr=attrs2)

    graph_def = tf.GraphDef()
    graph_def.node.extend([node1, node2])

    optimizer = TFGraphOptimizer(graph_def, passes=["cse"], opt_level=1)
    optimized_graph = optimizer.optimize()

    # node2 should be merged into node1
    node_names = [n.name for n in optimized_graph.node]
    assert "node1" in node_names
    assert "node2" not in node_names


def test_constant_fold_error_handling():
    """Test constant fold exception handling to cover code paths in constant_fold.py."""
    # Create an op that will fail during folding (e.g., Cast with invalid attributes)
    node = create_node("Cast", "cast_fail", inputs=["c1"])
    # Missing DstT attribute will cause it to return None in _rewrite_constant_op
    c1 = create_const_node("c1", [1.0], "float32")

    graph_def = tf.GraphDef()
    graph_def.node.extend([c1, node])

    optimizer = TFGraphOptimizer(graph_def, passes=["constant_fold"], opt_level=1)
    optimized_graph = optimizer.optimize()

    # Node should remain un-folded
    assert any(n.name == "cast_fail" for n in optimized_graph.node)


def test_constant_fold_specific_ops():
    """Test Transpose, Reshape, and Cast folding."""
    c1 = create_const_node("c1", [[1, 2], [3, 4]], "float32")

    # Reshape
    shape_const = create_const_node("shape", [4], "int32")
    reshape_node = create_node("Reshape", "reshape", inputs=["c1", "shape"])
    reshape_node.attr["T"].type = types_pb2.DT_FLOAT

    # Transpose
    perm_const = create_const_node("perm", [1, 0], "int32")
    transpose_node = create_node("Transpose", "transpose", inputs=["c1", "perm"])
    transpose_node.attr["T"].type = types_pb2.DT_FLOAT
    transpose_node.attr["Tperm"].type = types_pb2.DT_INT32

    # Cast
    cast_node = create_node("Cast", "cast", inputs=["c1"])
    cast_node.attr["SrcT"].type = types_pb2.DT_FLOAT
    cast_node.attr["DstT"].type = types_pb2.DT_INT32

    # Use Placeholder as sinks to prevent pruning.
    out1 = create_node("Placeholder", "out1", inputs=["reshape"])
    out2 = create_node("Placeholder", "out2", inputs=["transpose"])
    out3 = create_node("Placeholder", "out3", inputs=["cast"])
    for out in [out1, out2, out3]:
        out.attr["dtype"].type = types_pb2.DT_FLOAT

    graph_def = tf.GraphDef()
    graph_def.node.extend(
        [
            c1,
            shape_const,
            reshape_node,
            perm_const,
            transpose_node,
            cast_node,
            out1,
            out2,
            out3,
        ]
    )

    # Need to make sure the optimizer recognizes the ops as foldable
    # ConstantFold usually leverages _is_all_const

    optimizer = TFGraphOptimizer(graph_def, passes=["constant_fold"], opt_level=1)
    # Protection via protected_nodes
    optimized_graph = optimizer.optimize()

    # Check if original nodes are gone (meaning they were replaced)
    node_names = [n.name for n in optimized_graph.node]
    assert "reshape" not in node_names
    assert "transpose" not in node_names
    assert "cast" not in node_names

    # Check sinks
    for out_name in ["out1", "out2", "out3"]:
        out_node = next(n for n in optimized_graph.node if n.name == out_name)
        assert "_folded" in out_node.input[0], (
            f"{out_name} input {out_node.input[0]} not folded"
        )


def test_graph_utils_complex_attrs():
    """Test get_attr_value with tensor and shape attributes."""
    from graph_optimizer.utils.tf.graph_utils import get_attr_value

    # Shape attr
    shape_attr = attr_value_pb2.AttrValue()
    shape_attr.shape.dim.add().size = 2
    shape_attr.shape.dim.add().size = 3
    assert get_attr_value(shape_attr) == [2, 3]

    # Tensor attr (scalar)
    tensor_attr = attr_value_pb2.AttrValue()
    from tensorflow.python.framework import tensor_util

    tensor_proto = tensor_util.make_tensor_proto(5.0, dtype=tf.float32)
    tensor_attr.tensor.CopyFrom(tensor_proto)
    assert get_attr_value(tensor_attr) == 5.0


def test_pack_vectorize_matmul_diff_weights():
    """Target MatMul hoisting with different weights (covered by LS 1377-1403)."""
    # This involves multiple branches where weights are different, triggering Transpose/Reshape logic
    # Creating a simplified version of this scenario
    x0 = create_node("Placeholder", "x0")
    x0.attr["dtype"].type = types_pb2.DT_FLOAT
    x1 = create_node("Placeholder", "x1")
    x1.attr["dtype"].type = types_pb2.DT_FLOAT

    w0 = create_const_node("w0", [[1.0, 2.0]], "float32")
    w1 = create_const_node("w1", [[3.0, 4.0]], "float32")

    m0 = create_node("MatMul", "m0", inputs=["x0", "w0"])
    m1 = create_node("MatMul", "m1", inputs=["x1", "w1"])

    pack = create_node("Pack", "pack_out", inputs=["m0", "m1"])
    pack.attr["N"].i = 2
    pack.attr["axis"].i = 0
    pack.attr["T"].type = types_pb2.DT_FLOAT

    graph_def = tf.GraphDef()
    graph_def.node.extend([x0, x1, w0, w1, m0, m1, pack])

    # PackVectorize is opt_level 3
    optimizer = TFGraphOptimizer(graph_def, passes=["pack_vectorize"], opt_level=3)
    optimized_graph = optimizer.optimize()

    # Verification: pack should be hoisted above MatMul
    ops = [n.op for n in optimized_graph.node]
    assert "BatchMatMulV2" in ops
    assert "pack_out" not in [n.name for n in optimized_graph.node]
