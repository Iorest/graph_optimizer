"""Tests for AlgebraicSimplifyPass (TensorFlow)."""

import pytest
import tensorflow.compat.v1 as tf
from graph_optimizer.transforms.tensorflow.scalar.algebraic_simplify import (
    AlgebraicSimplifyPass,
)
from graph_optimizer.utils.graph_utils import create_node, create_const_node
from ..conftest import make_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _names(optimizer):
    return {n.name for n in optimizer.graph_def.node}


def _run(optimizer, **kwargs):
    AlgebraicSimplifyPass().transform(optimizer, auto_cleanup=True, **kwargs)


# ---------------------------------------------------------------------------
# Identity elimination — parametrized
# Each tuple: (op, [input_names], c_name, c_value, c_dtype, kept, removed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,inputs,c_name,c_val,c_dtype,kept,removed",
    [
        ("Add", ["zero", "x"], "zero", 0, "float32", "x", "add"),
        ("Add", ["x", "zero"], "zero", 0, "float32", "x", "add"),
        ("Sub", ["x", "zero"], "zero", 0, "float32", "x", "sub"),
        ("Mul", ["one", "x"], "one", 1, "float32", "x", "mul"),
        ("Mul", ["x", "one"], "one", 1, "float32", "x", "mul"),
        ("Div", ["x", "one"], "one", 1, "float32", "x", "div"),
        ("Pow", ["x", "one"], "one", 1, "float32", "x", "pow"),
        ("FloorDiv", ["a", "one"], "one", 1, "int32", "a", "floor_div"),
        ("RealDiv", ["a", "one"], "one", 1.0, "float32", "a", "real_div"),
    ],
)
def test_identity_elimination(op, inputs, c_name, c_val, c_dtype, kept, removed):
    c = create_const_node(c_name, value=c_val, dtype=c_dtype, shape=[])
    ph_name = "a" if op in ("FloorDiv", "RealDiv") else "x"
    ph = create_node("Placeholder", name=ph_name)
    node = create_node(op, name=removed, inputs=inputs)
    optimizer = make_graph(ph, c, node)
    _run(optimizer)
    names = _names(optimizer)
    assert kept in names
    assert removed not in names


# ---------------------------------------------------------------------------
# Double-negation / involution rules
# ---------------------------------------------------------------------------


def test_neg_neg():
    x = create_node("Placeholder", name="x")
    neg1 = create_node("Neg", name="neg1", inputs=["x"])
    neg2 = create_node("Neg", name="neg2", inputs=["neg1"])
    opt = make_graph(x, neg1, neg2)
    _run(opt)
    names = _names(opt)
    assert "x" in names
    assert "neg1" not in names
    assert "neg2" not in names


def test_logical_not_not():
    x = create_node("Placeholder", name="x")
    not1 = create_node("LogicalNot", name="not1", inputs=["x"])
    not2 = create_node("LogicalNot", name="not2", inputs=["not1"])
    opt = make_graph(x, not1, not2)
    _run(opt)
    names = _names(opt)
    assert "not1" not in names
    assert "not2" not in names


# ---------------------------------------------------------------------------
# Self-operand rules (x OP x → const)
# ---------------------------------------------------------------------------


def _scalar_ph(name):
    n = create_node("Placeholder", name=name)
    n.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
    return n


def test_sub_same():
    x = _scalar_ph("x")
    sub = create_node("Sub", name="sub", inputs=["x", "x"])
    opt = make_graph(x, sub)
    _run(opt, protected_nodes=["sub_zero"])
    assert "sub_zero" in _names(opt)
    assert "sub" not in _names(opt)


def test_div_same():
    x = _scalar_ph("x")
    div = create_node("Div", name="div", inputs=["x", "x"])
    opt = make_graph(x, div)
    _run(opt, protected_nodes=["div_one"])
    assert "div_one" in _names(opt)
    assert "div" not in _names(opt)


def test_floor_div_same():
    a = _scalar_ph("a")
    fd = create_node("FloorDiv", name="floor_div", inputs=["a", "a"])
    opt = make_graph(a, fd)
    _run(opt, protected_nodes=["floor_div_one"])
    assert "floor_div_one" in _names(opt)
    assert "floor_div" not in _names(opt)


def test_floor_mod_one():
    a = _scalar_ph("a")
    one = create_const_node("one", value=1, dtype="int32", shape=[])
    fm = create_node("FloorMod", name="floor_mod", inputs=["a", "one"])
    opt = make_graph(a, one, fm)
    _run(opt, protected_nodes=["floor_mod_zero"])
    assert "floor_mod_zero" in _names(opt)
    assert "floor_mod" not in _names(opt)


def test_floor_mod_same():
    a = _scalar_ph("a")
    fm = create_node("FloorMod", name="floor_mod", inputs=["a", "a"])
    opt = make_graph(a, fm)
    _run(opt, protected_nodes=["floor_mod_zero"])
    assert "floor_mod_zero" in _names(opt)
    assert "floor_mod" not in _names(opt)


def test_add_neg():
    x = _scalar_ph("x")
    neg = create_node("Neg", name="neg", inputs=["x"])
    add = create_node("Add", name="add", inputs=["x", "neg"])
    opt = make_graph(x, neg, add)
    _run(opt, protected_nodes=["add_zero"])
    assert "add_zero" in _names(opt)
    assert "add" not in _names(opt)


def test_maximum_same():
    a = create_node("Placeholder", name="a")
    m = create_node("Maximum", name="max_node", inputs=["a", "a"])
    opt = make_graph(a, m)
    _run(opt)
    assert "a" in _names(opt)
    assert "max_node" not in _names(opt)


def test_minimum_same():
    a = create_node("Placeholder", name="a")
    m = create_node("Minimum", name="min_node", inputs=["a", "a"])
    opt = make_graph(a, m)
    _run(opt)
    assert "a" in _names(opt)
    assert "min_node" not in _names(opt)


# ---------------------------------------------------------------------------
# Mul(x, x) → Square(x)  and  Pow(x, 2) → Square(x)
# ---------------------------------------------------------------------------


def test_mul_same_becomes_square():
    x = create_node("Placeholder", name="x")
    mul = create_node("Mul", name="mul", inputs=["x", "x"])
    opt = make_graph(x, mul)
    _run(opt)
    ops = {n.op for n in opt.graph_def.node}
    assert "Square" in ops
    assert "Mul" not in ops


def test_pow_two_becomes_square():
    x = create_node("Placeholder", name="x")
    two = create_const_node("two", value=2, dtype="float32", shape=[])
    pw = create_node("Pow", name="pow", inputs=["x", "two"])
    opt = make_graph(x, two, pw)
    _run(opt)
    ops = {n.op for n in opt.graph_def.node}
    assert "Square" in ops
    assert "Pow" not in ops


# ---------------------------------------------------------------------------
# Mul(x, 0)  and  Div(0, x) → zero const  (shape-aware)
# ---------------------------------------------------------------------------


def test_mul_zero_scalar():
    x = _scalar_ph("x")
    zero = create_const_node("zero", value=0, dtype="float32", shape=[])
    mul = create_node("Mul", name="mul", inputs=["zero", "x"])
    opt = make_graph(x, zero, mul)
    _run(opt, protected_nodes=["mul_zero"])
    assert (
        len([n for n in opt.graph_def.node if n.op == "Const" and n.name == "mul_zero"])
        == 1
    )
    assert "mul" not in _names(opt)


def test_mul_zero_broadcast():
    """Mul([2,1], [1,2]) → output must be [2,2] zero, not a scalar."""
    x = create_node("Placeholder", name="x")
    x.attr["shape"].shape.CopyFrom(tf.TensorShape([2, 1]).as_proto())
    zero = create_const_node("zero", value=[[0, 0]], dtype="float32", shape=[1, 2])
    mul = create_node("Mul", name="mul", inputs=["x", "zero"])
    opt = make_graph(x, zero, mul)
    _run(opt, protected_nodes=["mul_zero"])
    folded = [n for n in opt.graph_def.node if n.name == "mul_zero"]
    assert len(folded) == 1
    shape = [d.size for d in folded[0].attr["value"].tensor.tensor_shape.dim]
    assert shape == [2, 2]


def test_div_zero_left():
    zero = create_const_node("zero", value=[0.0, 0.0], dtype="float32", shape=[2])
    b = create_node("Placeholder", name="b")
    b.attr["shape"].shape.CopyFrom(tf.TensorShape([2]).as_proto())
    div = create_node("Div", name="div_node", inputs=["zero", "b"])
    opt = make_graph(zero, b, div)
    _run(opt, protected_nodes=["div_node_zero"])
    assert "div_node_zero" in _names(opt)
    assert "div_node" not in _names(opt)


# ---------------------------------------------------------------------------
# Select / Equal identity rules
# ---------------------------------------------------------------------------


def test_equal_same():
    x = _scalar_ph("x")
    eq = create_node("Equal", name="eq", inputs=["x", "x"])
    opt = make_graph(x, eq)
    _run(opt, protected_nodes=["eq_bool"])
    trues = [n for n in opt.graph_def.node if n.op == "Const" and n.name == "eq_bool"]
    assert len(trues) == 1
    assert "eq" not in _names(opt)


def test_select_same_branch():
    cond = create_node("Placeholder", name="cond")
    x = create_node("Placeholder", name="x")
    sel = create_node("Select", name="sel", inputs=["cond", "x", "x"])
    opt = make_graph(cond, x, sel)
    _run(opt)
    assert "x" in _names(opt)
    assert "sel" not in _names(opt)


# ---------------------------------------------------------------------------
# Logical shortcuts
# ---------------------------------------------------------------------------


def test_logical_and_false_scalar():
    x = _scalar_ph("x")
    false = create_const_node("false", value=False, dtype="bool", shape=[])
    and_n = create_node("LogicalAnd", name="and_node", inputs=["x", "false"])
    opt = make_graph(x, false, and_n)
    _run(opt, protected_nodes=["and_node_bool"])
    consts = [n for n in opt.graph_def.node if n.op == "Const"]
    assert any(
        n.name == "and_node_bool" and not n.attr["value"].tensor.bool_val[0]
        for n in consts
    )


def test_logical_and_broadcast_shape():
    """And(scalar_x, [False, False]) → must keep output shape [2]."""
    x = _scalar_ph("x")
    false = create_const_node("false", value=[False, False], dtype="bool", shape=[2])
    and_n = create_node("LogicalAnd", name="and_node", inputs=["x", "false"])
    opt = make_graph(x, false, and_n)
    _run(opt, protected_nodes=["and_node_bool"])
    folded = [n for n in opt.graph_def.node if n.name == "and_node_bool"]
    assert len(folded) == 1
    shape = [d.size for d in folded[0].attr["value"].tensor.tensor_shape.dim]
    assert shape == [2]


def test_logical_or_true_scalar():
    x = _scalar_ph("x")
    true = create_const_node("true", value=True, dtype="bool", shape=[])
    or_n = create_node("LogicalOr", name="or_node", inputs=["x", "true"])
    opt = make_graph(x, true, or_n)
    _run(opt, protected_nodes=["or_node_bool"])
    consts = [n for n in opt.graph_def.node if n.op == "Const"]
    assert any(
        n.name == "or_node_bool" and n.attr["value"].tensor.bool_val[0] for n in consts
    )


# ---------------------------------------------------------------------------
# Shape-safety broadcasting guard
# ---------------------------------------------------------------------------


def test_add_zero_broadcast_safe():
    """Add([2,2]_x, scalar_0) → safe to simplify to x."""
    x = create_node("Placeholder", name="x")
    x.attr["shape"].shape.CopyFrom(tf.TensorShape([2, 2]).as_proto())
    zero = create_const_node("zero", value=0, dtype="float32", shape=[])
    add = create_node("Add", name="add", inputs=["x", "zero"])
    opt = make_graph(x, zero, add)
    _run(opt)
    assert "x" in _names(opt)
    assert "add" not in _names(opt)


def test_add_zero_broadcast_unsafe():
    """Add(scalar_x, [2,2]_0) → NOT safe; output shape differs from x."""
    x = _scalar_ph("x")
    zero = create_const_node(
        "zero", value=[[0, 0], [0, 0]], dtype="float32", shape=[2, 2]
    )
    add = create_node("Add", name="add", inputs=["x", "zero"])
    opt = make_graph(x, zero, add)
    _run(opt)
    assert "add" in _names(opt)


# ---------------------------------------------------------------------------
# No-op guard
# ---------------------------------------------------------------------------


def test_no_simplify_add_nonzero():
    x = create_node("Placeholder", name="x")
    y = create_node("Placeholder", name="y")
    add = create_node("Add", name="add", inputs=["x", "y"])
    opt = make_graph(x, y, add)
    _run(opt)
    assert "add" in _names(opt)


# ---------------------------------------------------------------------------
# Abs and Square/Sqrt inverse rules
# ---------------------------------------------------------------------------


def _scalar_ph(name):
    n = create_node("Placeholder", name=name)
    n.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
    return n


def test_abs_abs_idempotent():
    """Abs(Abs(x)) → Abs(x)."""
    x = _scalar_ph("x")
    ab1 = create_node("Abs", name="abs1", inputs=["x"])
    ab2 = create_node("Abs", name="abs2", inputs=["abs1"])
    opt = make_graph(x, ab1, ab2)
    _run(opt, protected_nodes=["abs2_abs"])
    # abs2 replaced by a new Abs node, original abs2 gone
    assert "abs1" not in _names(opt)
    ops = {n.op for n in opt.graph_def.node}
    assert "Abs" in ops


def test_square_sqrt_simplifies_to_x():
    """Square(Sqrt(x)) → x."""
    x = _scalar_ph("x")
    sq = create_node("Sqrt", name="sqrt_x", inputs=["x"])
    sqsq = create_node("Square", name="sqsq", inputs=["sqrt_x"])
    opt = make_graph(x, sq, sqsq)
    _run(opt)
    assert "sqsq" not in _names(opt)
    assert "x" in _names(opt)


def test_sqrt_square_simplifies_to_abs():
    """Sqrt(Square(x)) → Abs(x)."""
    x = _scalar_ph("x")
    sq = create_node("Square", name="sq", inputs=["x"])
    sqsq = create_node("Sqrt", name="sqrt_sq", inputs=["sq"])
    opt = make_graph(x, sq, sqsq)
    _run(opt, protected_nodes=["sqrt_sq_abs"])
    ops = {n.op for n in opt.graph_def.node}
    assert "Abs" in ops
    assert "Sqrt" not in ops or "sqrt_sq" not in _names(opt)


# ---------------------------------------------------------------------------
# Comparison same-operand rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,expected_val,result_name_suffix",
    [
        ("NotEqual", False, "_bool"),
        ("Less", False, "_bool"),
        ("Greater", False, "_bool"),
        ("LessEqual", True, "_bool"),
        ("GreaterEqual", True, "_bool"),
    ],
)
def test_comparison_same_operand(op, expected_val, result_name_suffix):
    x = _scalar_ph("x")
    nd = create_node(op, name="cmp", inputs=["x", "x"])
    opt = make_graph(x, nd)
    _run(opt, protected_nodes=["cmp" + result_name_suffix])
    consts = [n for n in opt.graph_def.node if n.op == "Const"]
    assert len(consts) == 1
    val = consts[0].attr["value"].tensor.bool_val[0]
    assert val == expected_val


# ---------------------------------------------------------------------------
# LogicalOr identity rules
# ---------------------------------------------------------------------------


def test_logical_or_false_identity():
    """Or(x, False) → x."""
    x = _scalar_ph("x")
    false = create_const_node("false", value=False, dtype="bool", shape=[])
    or_n = create_node("LogicalOr", name="or_node", inputs=["x", "false"])
    opt = make_graph(x, false, or_n)
    _run(opt)
    assert "x" in _names(opt)
    assert "or_node" not in _names(opt)


def test_logical_and_same_operand():
    """And(x, x) → x."""
    x = _scalar_ph("x")
    and_n = create_node("LogicalAnd", name="and_node", inputs=["x", "x"])
    opt = make_graph(x, and_n)
    _run(opt)
    assert "x" in _names(opt)
    assert "and_node" not in _names(opt)
