"""Tests for CSEPass (TensorFlow) — Common Subexpression Elimination."""

import pytest
import tensorflow.compat.v1 as tf
from graph_optimizer.core.tensorflow import TFGraphOptimizer
from graph_optimizer.transforms.tensorflow.scalar import CSEPass
from graph_optimizer.utils import create_node
from tensorflow.core.framework import attr_value_pb2

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_graph(*nodes):
    gd = tf.GraphDef()
    gd.node.extend(nodes)
    return TFGraphOptimizer(gd)


def _dtype_attr(dtype):
    return attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)


def _ph(name, dtype=tf.float32):
    return create_node("Placeholder", name, attr={"dtype": _dtype_attr(dtype)})


def _const(name, value, dtype=tf.float32):
    return create_node(
        "Const",
        name,
        attr={
            "dtype": _dtype_attr(dtype),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(value, dtype=dtype)
            ),
        },
    )


def _run(opt, **kwargs):
    CSEPass().transform(opt, **kwargs)


# ---------------------------------------------------------------------------
# Basic deduplication — parametrized for constant dtypes
# ---------------------------------------------------------------------------


def test_basic_duplicate_elimination():
    """Duplicate consts + downstream duplicate ops are both removed."""
    opt = _make_graph(
        _ph("input", tf.int32),
        _const("weights_1", 1.0),
        _const("weights_2", 1.0),  # dup
        create_node("Add", "add_1", inputs=["input", "weights_1"]),
        create_node("Add", "add_2", inputs=["input", "weights_2"]),  # will become dup
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before - 2
    assert "weights_1" in opt.nodes and "weights_2" not in opt.nodes
    assert "add_1" in opt.nodes and "add_2" not in opt.nodes


def test_const_different_dtypes_not_merged():
    opt = _make_graph(_const("ci", 1, tf.int32), _const("cf", 1.0, tf.float32))
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


def test_const_same_dtype_same_value_merged():
    opt = _make_graph(
        _const("c1", 3.14),
        _const("c2", 3.14),
        create_node("Add", "add", inputs=["c1", "c2"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before - 1
    assert "c1" in opt.nodes or "c2" in opt.nodes
    assert not ("c1" in opt.nodes and "c2" in opt.nodes)


def test_const_different_values_not_merged():
    opt = _make_graph(_const("c1", 1.0), _const("c2", 2.0), _const("c3", 3.14))
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


# ---------------------------------------------------------------------------
# Control dependency handling
# ---------------------------------------------------------------------------


def test_control_dep_prevents_merge():
    """Nodes with different control deps are NOT merged even if operands match."""
    opt = _make_graph(
        _ph("input"),
        _const("w1", 1.0),
        _const("w2", 1.0),
        create_node("NoOp", "ctrl_op"),
        create_node("Add", "add_1", inputs=["input", "w1", "^ctrl_op"]),
        create_node("Add", "add_2", inputs=["input", "w2"]),
    )
    before = len(opt.nodes)
    _run(opt)
    # w2 eliminated (1 node); add_1 and add_2 differ in ctrl dep
    assert len(opt.nodes) == before - 1
    add_1 = opt.nodes.get("add_1")
    assert add_1 is not None and "^ctrl_op" in add_1.input
    assert "add_2" in opt.nodes
    assert "w1" in opt.nodes and "w2" not in opt.nodes


def test_same_control_deps_merged():
    """Nodes with identical control deps ARE merged."""
    opt = _make_graph(
        _ph("input"),
        _const("w", 1.0),
        create_node("NoOp", "ctrl"),
        create_node("Add", "add_1", inputs=["input", "w", "^ctrl"]),
        create_node("Add", "add_2", inputs=["input", "w", "^ctrl"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before - 1
    assert ("add_1" in opt.nodes) != ("add_2" in opt.nodes)


def test_different_control_dep_counts_not_merged():
    """One ctrl dep vs two ctrl deps → different signatures → no merge."""
    opt = _make_graph(
        _ph("input"),
        _const("w", 1.0),
        create_node("NoOp", "ctrl_1"),
        create_node("NoOp", "ctrl_2"),
        create_node("Add", "add_1", inputs=["input", "w", "^ctrl_1"]),
        create_node("Add", "add_2", inputs=["input", "w", "^ctrl_1", "^ctrl_2"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before
    assert "add_1" in opt.nodes and "add_2" in opt.nodes


# ---------------------------------------------------------------------------
# Multi-port & iterative convergence
# ---------------------------------------------------------------------------


def test_multi_port_outputs():
    """Duplicate Split nodes + downstream Adds eliminated via cascading."""
    opt = _make_graph(
        _ph("input"),
        create_node(
            "Split",
            "split_1",
            inputs=["input"],
            attr={"num_split": attr_value_pb2.AttrValue(i=2)},
        ),
        create_node(
            "Split",
            "split_2",
            inputs=["input"],
            attr={"num_split": attr_value_pb2.AttrValue(i=2)},
        ),
        create_node("Add", "add_1", inputs=["split_1:0", "split_1:1"]),
        create_node("Add", "add_2", inputs=["split_2:0", "split_2:1"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before - 2
    assert "split_1" in opt.nodes and "split_2" not in opt.nodes
    assert "add_1" in opt.nodes and "add_2" not in opt.nodes


def test_iterative_convergence():
    """Eliminating c2 cascades to reveal add_2 and mul_2 as dups (3 removed)."""
    opt = _make_graph(
        _ph("x"),
        _ph("y"),
        _const("c1", 2.0),
        _const("c2", 2.0),
        create_node("Add", "add_1", inputs=["x", "c1"]),
        create_node("Add", "add_2", inputs=["x", "c2"]),
        create_node("Mul", "mul_1", inputs=["add_1", "y"]),
        create_node("Mul", "mul_2", inputs=["add_2", "y"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before - 3
    for kept in ("c1", "add_1", "mul_1"):
        assert kept in opt.nodes
    for removed in ("c2", "add_2", "mul_2"):
        assert removed not in opt.nodes


def test_deep_iterative_convergence():
    """4-level cascade: c2 + l1_b + l2_b + l3_b + l4_b = 5 removed."""
    opt = _make_graph(
        _ph("x"),
        _const("c1", 1.0),
        _const("c2", 1.0),
        create_node("Add", "l1_a", inputs=["x", "c1"]),
        create_node("Add", "l1_b", inputs=["x", "c2"]),
        create_node("Mul", "l2_a", inputs=["l1_a", "c1"]),
        create_node("Mul", "l2_b", inputs=["l1_b", "c2"]),
        create_node("Sub", "l3_a", inputs=["l2_a", "x"]),
        create_node("Sub", "l3_b", inputs=["l2_b", "x"]),
        create_node("Relu", "l4_a", inputs=["l3_a"]),
        create_node("Relu", "l4_b", inputs=["l3_b"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before - 5
    for removed in ("c2", "l1_b", "l2_b", "l3_b", "l4_b"):
        assert removed not in opt.nodes


def test_complex_graph_multi_duplicate():
    """c2, c4, add_2, mul_2, final_2 (5 nodes) removed."""
    opt = _make_graph(
        _ph("x"),
        _ph("y"),
        _const("c1", 1.0),
        _const("c2", 1.0),
        _const("c3", 2.0),
        _const("c4", 2.0),
        create_node("Add", "add_1", inputs=["x", "c1"]),
        create_node("Add", "add_2", inputs=["x", "c2"]),
        create_node("Mul", "mul_1", inputs=["y", "c3"]),
        create_node("Mul", "mul_2", inputs=["y", "c4"]),
        create_node("Add", "final_1", inputs=["add_1", "mul_1"]),
        create_node("Add", "final_2", inputs=["add_2", "mul_2"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before - 5
    for kept in ("c1", "c3", "add_1", "mul_1", "final_1"):
        assert kept in opt.nodes
    for removed in ("c2", "c4", "add_2", "mul_2", "final_2"):
        assert removed not in opt.nodes


def test_many_duplicates_performance():
    """50 duplicate consts → 1 surviving const; runs in < 5s."""
    import time

    nodes = [_ph("input")] + [_const(f"c_{i}", 3.14) for i in range(50)]
    opt = _make_graph(*nodes)
    t0 = time.time()
    _run(opt)
    assert len(opt.nodes) == 2
    assert time.time() - t0 < 5.0


# ---------------------------------------------------------------------------
# Skip / no-merge rules (ops that must never be CSE'd)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nodes,check",
    [
        # Placeholders with same dtype → never merged
        (
            [_ph("i1"), _ph("i2"), create_node("Add", "add", inputs=["i1", "i2"])],
            ["i1", "i2"],
        ),
        # Identity nodes → handled by a dedicated pass
        (
            [
                _ph("inp"),
                create_node("Identity", "id_1", inputs=["inp"]),
                create_node("Identity", "id_2", inputs=["inp"]),
            ],
            ["id_1", "id_2"],
        ),
        # NoOp → in skip_ops list
        (
            [
                create_node("NoOp", "n1"),
                create_node("NoOp", "n2"),
                create_node("NoOp", "n3"),
            ],
            ["n1", "n2", "n3"],
        ),
        # Assert (side effects)
        (
            [
                _ph("cond", tf.bool),
                _ph("data"),
                create_node("Assert", "a1", inputs=["cond", "data"]),
                create_node("Assert", "a2", inputs=["cond", "data"]),
            ],
            ["a1", "a2"],
        ),
    ],
)
def test_skip_ops_not_merged(nodes, check):
    opt = _make_graph(*nodes)
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before
    for name in check:
        assert name in opt.nodes


def test_skip_variable():
    shape_attr = attr_value_pb2.AttrValue(shape=tf.TensorShape([10]).as_proto())
    opt = _make_graph(
        create_node(
            "VariableV2",
            "var_1",
            attr={"dtype": _dtype_attr(tf.float32), "shape": shape_attr},
        ),
        create_node(
            "VariableV2",
            "var_2",
            attr={"dtype": _dtype_attr(tf.float32), "shape": shape_attr},
        ),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


def test_skip_stateful_ops():
    opt = _make_graph(
        _ph("input"),
        create_node(
            "RandomUniform",
            "rand_1",
            inputs=["input"],
            attr={
                "dtype": _dtype_attr(tf.float32),
                "seed": attr_value_pb2.AttrValue(i=42),
            },
        ),
        create_node(
            "RandomUniform",
            "rand_2",
            inputs=["input"],
            attr={
                "dtype": _dtype_attr(tf.float32),
                "seed": attr_value_pb2.AttrValue(i=42),
            },
        ),
        create_node("Print", "print_1", inputs=["input", "input"]),
        create_node("Print", "print_2", inputs=["input", "input"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


def test_skip_variable_read_ops():
    shape_attr = attr_value_pb2.AttrValue(shape=tf.TensorShape([10]).as_proto())
    opt = _make_graph(
        create_node(
            "VarHandleOp",
            "var",
            attr={"dtype": _dtype_attr(tf.float32), "shape": shape_attr},
        ),
        create_node(
            "ReadVariableOp",
            "read_1",
            inputs=["var"],
            attr={"dtype": _dtype_attr(tf.float32)},
        ),
        create_node(
            "ReadVariableOp",
            "read_2",
            inputs=["var"],
            attr={"dtype": _dtype_attr(tf.float32)},
        ),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


def test_skip_control_flow_ops():
    opt = _make_graph(
        _ph("pred", tf.bool),
        _ph("data"),
        create_node("Switch", "switch_1", inputs=["data", "pred"]),
        create_node("Switch", "switch_2", inputs=["data", "pred"]),
        create_node("Merge", "merge_1", inputs=["switch_1:0", "switch_1:1"]),
        create_node("Merge", "merge_2", inputs=["switch_2:0", "switch_2:1"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_graph():
    opt = _make_graph()
    _run(opt)
    assert len(opt.nodes) == 0


def test_single_node():
    opt = _make_graph(_ph("input"))
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


def test_no_duplicates_unchanged():
    opt = _make_graph(
        _ph("input"),
        _const("weights", 1.0),
        create_node("MatMul", "matmul", inputs=["input", "weights"]),
        create_node("Relu", "relu", inputs=["matmul"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


def test_different_inputs_same_op_not_merged():
    opt = _make_graph(
        _ph("x"),
        _ph("y"),
        _ph("z"),
        create_node("Add", "add_1", inputs=["x", "y"]),
        create_node("Add", "add_2", inputs=["x", "z"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


def test_different_attrs_same_inputs_not_merged():
    opt = _make_graph(
        _ph("input"),
        _const("ax0", 0, tf.int32),
        _const("ax1", 1, tf.int32),
        create_node(
            "Split",
            "s1",
            inputs=["ax0", "input"],
            attr={"num_split": attr_value_pb2.AttrValue(i=2)},
        ),
        create_node(
            "Split",
            "s2",
            inputs=["ax1", "input"],
            attr={"num_split": attr_value_pb2.AttrValue(i=2)},
        ),
    )
    _run(opt)
    assert "s1" in opt.nodes and "s2" in opt.nodes


def test_different_output_ports_not_merged():
    opt = _make_graph(
        _ph("input"),
        create_node(
            "Split",
            "split",
            inputs=["input"],
            attr={"num_split": attr_value_pb2.AttrValue(i=2)},
        ),
        create_node("Add", "add_1", inputs=["split:0", "split:0"]),
        create_node("Add", "add_2", inputs=["split:0", "split:1"]),
    )
    before = len(opt.nodes)
    _run(opt)
    assert len(opt.nodes) == before


# ---------------------------------------------------------------------------
# Canonical selection & protected nodes
# ---------------------------------------------------------------------------


def test_canonical_selection_shortest_name():
    opt = _make_graph(
        _ph("x"),
        _const("very_long_name_1", 1.0),
        _const("short", 1.0),
        _const("very_long_name_2", 1.0),
    )
    _run(opt)
    assert "short" in opt.nodes
    assert "very_long_name_1" not in opt.nodes
    assert "very_long_name_2" not in opt.nodes


def test_protected_nodes_kept_as_canonical():
    opt = _make_graph(
        _ph("input"),
        _const("weights_1", 1.0),
        _const("weights_2", 1.0),
        create_node("Add", "add_1", inputs=["input", "weights_1"]),
        create_node("Add", "add_2", inputs=["input", "weights_2"]),
    )
    before = len(opt.nodes)
    _run(opt, protected_nodes=["weights_2", "add_2"])
    assert "weights_2" in opt.nodes
    assert "add_2" in opt.nodes
    assert "weights_1" not in opt.nodes
    assert "add_1" not in opt.nodes
    assert len(opt.nodes) == before - 2


def test_protected_node_becomes_canonical():
    opt = _make_graph(
        _ph("input"),
        _const("const_a", 1.0),
        _const("const_b_protected", 1.0),
        _const("const_c", 1.0),
    )
    before = len(opt.nodes)
    _run(opt, protected_nodes=["const_b_protected"])
    assert "const_b_protected" in opt.nodes
    assert "const_a" not in opt.nodes
    assert "const_c" not in opt.nodes
    assert len(opt.nodes) == before - 2


def test_multiple_protected_nodes_same_sig_unprotected_removed():
    opt = _make_graph(
        _ph("input"),
        _const("const_long_protected_name", 1.0),
        _const("const_p", 1.0),
        _const("const_medium_protected", 1.0),
        _const("const_unprotected", 1.0),
    )
    _run(
        opt,
        protected_nodes=[
            "const_long_protected_name",
            "const_p",
            "const_medium_protected",
        ],
    )
    assert "const_p" in opt.nodes
    assert "const_long_protected_name" in opt.nodes
    assert "const_medium_protected" in opt.nodes
    assert "const_unprotected" not in opt.nodes


def test_protected_nodes_empty_list_same_as_none():
    opt = _make_graph(
        _ph("input"),
        _const("c1", 1.0),
        _const("c2", 1.0),
    )
    before = len(opt.nodes)
    _run(opt, protected_nodes=[])
    assert len(opt.nodes) == before - 1
