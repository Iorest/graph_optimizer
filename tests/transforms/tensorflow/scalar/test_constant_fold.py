"""Tests for ConstantFoldPass (TensorFlow)."""

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import tensor_util
from graph_optimizer.transforms.tensorflow.scalar.constant_fold import ConstantFoldPass
from graph_optimizer.utils.tf.graph_utils import create_node, create_const_node
from ..conftest import make_graph

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(optimizer, **kwargs):
    ConstantFoldPass().transform(optimizer, auto_cleanup=True, **kwargs)


def _folded(optimizer, name):
    return [n for n in optimizer.graph_def.node if n.name == name]


def _val(node):
    return tensor_util.MakeNdarray(node.attr["value"].tensor)


# ---------------------------------------------------------------------------
# Basic folding — parametrized
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,a,b,a_dtype,b_dtype,expected",
    [
        ("Add", [2, 3], [4, 5], "int32", "int32", [6, 8]),
        ("Mul", [2, 3], [4, 5], "int32", "int32", [8, 15]),
        ("Maximum", [1, 5], [2, 4], "int32", "int32", [2, 5]),
        ("Minimum", [1, 5], [2, 4], "int32", "int32", [1, 4]),
        ("FloorDiv", [5, -5], [2, 2], "int32", "int32", [2, -3]),
        ("FloorMod", [5, -5], [2, 2], "int32", "int32", [1, 1]),
        (
            "RealDiv",
            [5.0, 6.0],
            [2.0, 3.0],
            "float32",
            "float32",
            [2.5, 2.0],
        ),
    ],
)
def test_fold_binary(op, a, b, a_dtype, b_dtype, expected):
    # Node name uses op.lower() so the pass creates "{op.lower()}_folded"
    node_name = op.lower()
    folded_name = node_name + "_folded"
    ca = create_const_node("a", value=a, dtype=a_dtype, shape=[len(a)])
    cb = create_const_node("b", value=b, dtype=b_dtype, shape=[len(b)])
    out = create_node(op, name=node_name, inputs=["a", "b"])
    opt = make_graph(ca, cb, out)
    _run(opt, protected_nodes=[folded_name])
    nodes = _folded(opt, folded_name)
    assert len(nodes) == 1
    assert _val(nodes[0]).tolist() == expected


# ---------------------------------------------------------------------------
# Edge cases (unique per test)
# ---------------------------------------------------------------------------


def test_div_zero_yields_inf():
    ca = create_const_node("a", value=[2.0], dtype="float32", shape=[1])
    cb = create_const_node("b", value=[0.0], dtype="float32", shape=[1])
    div = create_node("Div", name="div", inputs=["a", "b"])
    opt = make_graph(ca, cb, div)
    _run(opt, protected_nodes=["div_folded"])
    nodes = _folded(opt, "div_folded")
    assert len(nodes) == 1
    assert np.isinf(_val(nodes[0])[0])


def test_sqrt_negative_yields_nan():
    ca = create_const_node("a", value=[-1.0], dtype="float32", shape=[1])
    sqrt = create_node("Sqrt", name="sqrt", inputs=["a"])
    opt = make_graph(ca, sqrt)
    _run(opt, protected_nodes=["sqrt_folded"])
    nodes = _folded(opt, "sqrt_folded")
    assert len(nodes) == 1
    assert np.isnan(_val(nodes[0])[0])


def test_dtype_promotion():
    ca = create_const_node("a", value=[2], dtype="int32", shape=[1])
    cb = create_const_node("b", value=[4.5], dtype="float32", shape=[1])
    add = create_node("Add", name="add", inputs=["a", "b"])
    opt = make_graph(ca, cb, add)
    _run(opt, protected_nodes=["add_folded"])
    nodes = _folded(opt, "add_folded")
    assert len(nodes) == 1
    dtype = tf.as_dtype(nodes[0].attr["dtype"].type).name
    assert dtype in ("float32", "float64")


def test_unsupported_op_not_folded():
    ca = create_const_node("a", value=[1], dtype="int32", shape=[1])
    unk = create_node("UnknownOp", name="unk", inputs=["a"])
    opt = make_graph(ca, unk)
    _run(opt)
    assert any(n.name == "unk" for n in opt.graph_def.node)


def test_partial_non_const_not_folded():
    ca = create_const_node("a", value=[2], dtype="int32", shape=[1])
    ph = create_node("Placeholder", name="ph")
    add = create_node("Add", name="add", inputs=["a", "ph"])
    opt = make_graph(ca, ph, add)
    _run(opt)
    assert any(n.name == "add" for n in opt.graph_def.node)


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,val,expected",
    [
        ("Neg", [3.0, -1.0], [-3.0, 1.0]),
        ("Abs", [-2.0, 3.0], [2.0, 3.0]),
        ("Square", [2.0, 3.0], [4.0, 9.0]),
        ("Exp", [0.0], [1.0]),
        ("Sqrt", [4.0, 9.0], [2.0, 3.0]),
    ],
)
def test_fold_unary(op, val, expected):
    node_name = op.lower()
    folded_name = node_name + "_folded"
    ca = create_const_node("a", value=val, dtype="float32", shape=[len(val)])
    out = create_node(op, name=node_name, inputs=["a"])
    opt = make_graph(ca, out)
    _run(opt, protected_nodes=[folded_name])
    nodes = _folded(opt, folded_name)
    assert len(nodes) == 1
    result = _val(nodes[0]).tolist()
    assert all(abs(r - e) < 1e-5 for r, e in zip(result, expected))


def test_fold_pow():
    ca = create_const_node("base", value=[2.0, 3.0], dtype="float32", shape=[2])
    exp = create_const_node("exp", value=[3.0, 2.0], dtype="float32", shape=[2])
    pw = create_node("Pow", name="pow", inputs=["base", "exp"])
    opt = make_graph(ca, exp, pw)
    _run(opt, protected_nodes=["pow_folded"])
    nodes = _folded(opt, "pow_folded")
    assert len(nodes) == 1
    result = _val(nodes[0]).tolist()
    assert abs(result[0] - 8.0) < 1e-5  # 2^3
    assert abs(result[1] - 9.0) < 1e-5  # 3^2


# ---------------------------------------------------------------------------
# Comparison ops (result dtype = bool)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,a,b,expected",
    [
        ("Equal", [1, 2], [1, 3], [True, False]),
        ("NotEqual", [1, 2], [1, 3], [False, True]),
        ("Less", [1, 5], [2, 3], [True, False]),
        ("Greater", [3, 1], [2, 5], [True, False]),
        ("LessEqual", [1, 2], [2, 2], [True, True]),
        ("GreaterEqual", [2, 1], [2, 2], [True, False]),
    ],
)
def test_fold_comparison(op, a, b, expected):
    node_name = op.lower()
    folded_name = node_name + "_folded"
    ca = create_const_node("a", value=a, dtype="int32", shape=[len(a)])
    cb = create_const_node("b", value=b, dtype="int32", shape=[len(b)])
    out = create_node(op, name=node_name, inputs=["a", "b"])
    opt = make_graph(ca, cb, out)
    _run(opt, protected_nodes=[folded_name])
    nodes = _folded(opt, folded_name)
    assert len(nodes) == 1
    result = _val(nodes[0]).tolist()
    assert result == expected
