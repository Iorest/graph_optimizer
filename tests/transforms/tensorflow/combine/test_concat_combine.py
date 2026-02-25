"""Tests for ConcatCombinePass (TensorFlow)."""

import pytest
import tensorflow.compat.v1 as tf
from graph_optimizer.core.tensorflow import TFGraphOptimizer
from graph_optimizer.utils import create_node
from graph_optimizer.transforms.tensorflow.combine import ConcatCombinePass
from tensorflow.core.framework import attr_value_pb2

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# Fixture / helpers
# ---------------------------------------------------------------------------


def _axis_const(name="axis0", val=0):
    return create_node(
        "Const",
        name,
        attr={
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(val, dtype=tf.int32)
            )
        },
    )


def _run(graph_def, protected=("outer",)):
    opt = TFGraphOptimizer(graph_def)
    return ConcatCombinePass().transform(opt, protected_nodes=list(protected))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_concat_combine_basic():
    """inner = ConcatV2([a, b], axis) → outer = ConcatV2([a, b, c], axis)."""
    gd = tf.GraphDef()
    axis = _axis_const()
    a, b = create_node("Placeholder", "a"), create_node("Placeholder", "b")
    inner = create_node("ConcatV2", "inner", inputs=["a", "b", "axis0"])
    c = create_node("Placeholder", "c")
    outer = create_node("ConcatV2", "outer", inputs=["inner", "c", "axis0"])
    gd.node.extend([axis, a, b, inner, c, outer])

    result = _run(gd)
    node_map = {n.name: n for n in result.node}
    assert "outer" in node_map
    assert node_map["outer"].input == ["a", "b", "c", "axis0"]
    assert "inner" not in node_map


def test_concat_combine_hoists_inner_control_dep():
    """Control dep on inner node must be hoisted to outer."""
    gd = tf.GraphDef()
    axis = _axis_const()
    ctrl = create_node("NoOp", "ctrl_op")
    a, b = create_node("Placeholder", "a"), create_node("Placeholder", "b")
    inner = create_node("ConcatV2", "inner", inputs=["a", "b", "axis0", "^ctrl_op"])
    c = create_node("Placeholder", "c")
    outer = create_node("ConcatV2", "outer", inputs=["inner", "c", "axis0"])
    gd.node.extend([axis, ctrl, a, b, inner, c, outer])

    result = _run(gd)
    node_map = {n.name: n for n in result.node}
    outer_node = node_map["outer"]
    data_inputs = [i for i in outer_node.input if not i.startswith("^")]
    control_inputs = [i for i in outer_node.input if i.startswith("^")]
    assert data_inputs == ["a", "b", "c", "axis0"]
    assert "^ctrl_op" in control_inputs
    assert "inner" not in node_map


def test_concat_combine_preserves_outer_control_dep_n_attr():
    """Outer's own control deps and N attribute must be correct after fusion."""
    gd = tf.GraphDef()
    axis = _axis_const()
    ctrl = create_node("NoOp", "ctrl_op")
    a, b, c = [create_node("Placeholder", name) for name in ("a", "b", "c")]
    inner = create_node("ConcatV2", "inner", inputs=["a", "b", "axis0"])
    inner.attr["N"].i = 2
    outer = create_node("ConcatV2", "outer", inputs=["inner", "c", "axis0", "^ctrl_op"])
    outer.attr["N"].i = 2
    gd.node.extend([axis, ctrl, a, b, c, inner, outer])

    opt = TFGraphOptimizer(gd)
    ConcatCombinePass().transform(opt, protected_nodes=["outer"])
    new_outer = opt.nodes["outer"]
    assert new_outer.attr["N"].i == 3
    assert "^ctrl_op" in new_outer.input
