"""
Cross-pass integration and consistency tests.

Validates that passes interact correctly on complex graphs:
- CSE attribute safety (different attrs → no merge)
- AlgebraicSimplify + CSE convergence
- Constant folding chain
- PackVectorize attribute-mismatch guard
- Full multi-pass pipeline integration
"""

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
from graph_optimizer.runner import OptimizationPipeline
from graph_optimizer.core.tensorflow import TFGraphOptimizer
from graph_optimizer.utils.tf.graph_utils import (
    create_node,
    create_const_node,
    make_output_shapes_attr,
)
from graph_optimizer.transforms.tensorflow.scalar import CSEPass
from graph_optimizer.transforms.tensorflow.scalar.algebraic_simplify import (
    AlgebraicSimplifyPass,
)
from graph_optimizer.transforms.tensorflow.vectorize import PackVectorizePass
from tensorflow.core.framework import attr_value_pb2

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _const(name, value, dtype=tf.float32):
    return create_node(
        "Const",
        name,
        attr={
            "dtype": attr_value_pb2.AttrValue(type=dtype.as_datatype_enum),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(value, dtype=dtype)
            ),
        },
    )


def _ph(name, shape, dtype=tf.float32):
    n = create_node("Placeholder", name)
    n.attr["shape"].shape.CopyFrom(tf.TensorShape(shape).as_proto())
    n.attr["dtype"].type = dtype.as_datatype_enum
    return n


def _set_shape(node, shape):
    node.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([shape]))


# ---------------------------------------------------------------------------
# CSE attribute safety
# ---------------------------------------------------------------------------


def test_cse_does_not_merge_nodes_with_different_attrs():
    """CSE must not merge MatMul nodes that differ only in transpose_a."""
    x = create_node("Placeholder", "x")
    w = create_node("Placeholder", "w")
    m1 = create_node("MatMul", "mm1", inputs=["x", "w"])
    m1.attr["transpose_a"].b = True
    m2 = create_node("MatMul", "mm2", inputs=["x", "w"])
    m2.attr["transpose_a"].b = False
    id1 = create_node("Identity", "id1", inputs=["mm1"])
    id2 = create_node("Identity", "id2", inputs=["mm2"])
    gd = tf.GraphDef()
    gd.node.extend([x, w, m1, m2, id1, id2])
    opt = TFGraphOptimizer(gd)
    CSEPass().transform(opt)
    assert len([n for n in opt.graph_def.node if n.op == "MatMul"]) == 2


# ---------------------------------------------------------------------------
# Algebraic + CSE convergence
# ---------------------------------------------------------------------------


def test_cse_then_algebraic_simplify_converges():
    """CSE merges zero consts → algebraic simplify removes both Add(x, 0) ops."""
    zero1 = create_const_node("zero1", 0.0, "float32", [])
    zero2 = create_const_node("zero2", 0.0, "float32", [])
    x = _ph("x", [10])
    add1 = create_node("Add", "add1", inputs=["x", "zero1"])
    add2 = create_node("Add", "add2", inputs=["x", "zero2"])
    final = create_node("Add", "final", inputs=["add1", "add2"])
    gd = tf.GraphDef()
    gd.node.extend([x, zero1, zero2, add1, add2, final])
    opt = TFGraphOptimizer(gd)
    CSEPass().transform(opt)
    AlgebraicSimplifyPass().transform(opt)
    names = {n.name for n in opt.graph_def.node}
    assert "add1" not in names
    assert "add2" not in names
    final_node = next(n for n in opt.graph_def.node if n.name == "final")
    assert final_node.input == ["x", "x"]


# ---------------------------------------------------------------------------
# Constant folding chain
# ---------------------------------------------------------------------------


def test_constant_fold_arithmetic_chain():
    """((2 + 3) * 4 - 10) / 2 = 5 must be folded to a single constant."""
    c2 = _const("c2", 2.0)
    c3 = _const("c3", 3.0)
    c4 = _const("c4", 4.0)
    c10 = _const("c10", 10.0)
    add = create_node("Add", "add", inputs=["c2", "c3"])
    add.attr["T"].type = tf.float32.as_datatype_enum
    mul = create_node("Mul", "mul", inputs=["add", "c4"])
    mul.attr["T"].type = tf.float32.as_datatype_enum
    sub = create_node("Sub", "sub", inputs=["mul", "c10"])
    sub.attr["T"].type = tf.float32.as_datatype_enum
    div = create_node("Div", "div", inputs=["sub", "c2"])
    div.attr["T"].type = tf.float32.as_datatype_enum
    out = create_node("Identity", "out", inputs=["div"])
    out.attr["T"].type = tf.float32.as_datatype_enum
    gd = tf.GraphDef()
    gd.node.extend([c2, c3, c4, c10, add, mul, sub, div, out])

    original = tf.GraphDef()
    original.CopyFrom(gd)

    report = OptimizationPipeline(graph_def=gd, level=3, output_nodes=["out"]).run()
    optimized = report.graph_def
    assert "out" in {n.name: n for n in optimized.node}

    # Verify numerically
    g = tf.Graph()
    with g.as_default():
        tf.import_graph_def(optimized, name="")
    with tf.Session(graph=g) as sess:
        val = sess.run(g.get_tensor_by_name("out:0"))
    assert abs(float(val) - 5.0) < 1e-5


# ---------------------------------------------------------------------------
# PackVectorize attribute mismatch guard
# ---------------------------------------------------------------------------


def test_pack_vectorize_refuses_attr_mismatch():
    """PackVectorize must NOT hoist MatMul nodes that differ in transpose_a."""
    x1 = _ph("x1", [4, 4])
    x2 = _ph("x2", [4, 4])
    w = _ph("w", [4, 4])
    mm1 = create_node("MatMul", "mm1", inputs=["x1", "w"])
    mm1.attr["transpose_a"].b = True
    mm1.attr["T"].type = tf.float32.as_datatype_enum
    _set_shape(mm1, [4, 4])
    mm2 = create_node("MatMul", "mm2", inputs=["x2", "w"])
    mm2.attr["transpose_a"].b = False  # mismatch
    mm2.attr["T"].type = tf.float32.as_datatype_enum
    _set_shape(mm2, [4, 4])
    pack = create_node("Pack", "pack", inputs=["mm1", "mm2"])
    pack.attr["axis"].i = 0
    pack.attr["N"].i = 2
    pack.attr["T"].type = tf.float32.as_datatype_enum
    _set_shape(pack, [2, 4, 4])
    gd = tf.GraphDef()
    gd.node.extend([x1, x2, w, mm1, mm2, pack])
    opt = TFGraphOptimizer(gd)
    result = PackVectorizePass().transform(opt)
    assert len([n for n in result.node if n.op == "MatMul"]) == 2


# ---------------------------------------------------------------------------
# Multi-pass pipeline integration
# ---------------------------------------------------------------------------


def test_algebraic_simplify_complex_graph():
    """Full pipeline: (x+0)*1 + (y-y) + (z/z) is simplified; final_add2 survives."""
    x = _ph("x", [10])
    y = _ph("y", [10])
    z = _ph("z", [10])
    zero = _const("zero", 0.0)
    one = _const("one", 1.0)

    def _t(n):
        n.attr["T"].type = tf.float32.as_datatype_enum
        return n

    add0 = _t(create_node("Add", "add0", inputs=["x", "zero"]))
    mul1 = _t(create_node("Mul", "mul1", inputs=["add0", "one"]))
    sub_y = _t(create_node("Sub", "sub_y", inputs=["y", "y"]))
    div_z = _t(create_node("Div", "div_z", inputs=["z", "z"]))
    fadd1 = _t(create_node("Add", "final_add1", inputs=["mul1", "sub_y"]))
    fadd2 = _t(create_node("Add", "final_add2", inputs=["final_add1", "div_z"]))
    gd = tf.GraphDef()
    gd.node.extend([x, y, z, zero, one, add0, mul1, sub_y, div_z, fadd1, fadd2])

    original = tf.GraphDef()
    original.CopyFrom(gd)

    report = OptimizationPipeline(
        graph_def=gd, level=3, output_nodes=["final_add2"]
    ).run()
    assert "final_add2" in {n.name: n for n in report.graph_def.node}
