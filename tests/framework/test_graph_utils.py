"""
Unit tests for graph_utils and visualization utilities.

Covers: create_node, create_const_node, extract_base_name, canonicalize_axis,
compute_reference_counts, prune_dead_nodes, final_prune, build_consumer_index,
update_node_inputs, graph_to_mermaid, graph_to_dot.
"""

import pytest
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import attr_value_pb2

from graph_optimizer.utils.graph_utils import (
    create_node,
    create_const_node,
    extract_base_name,
    canonicalize_axis,
    compute_reference_counts,
    build_consumer_index,
    prune_dead_nodes,
    final_prune,
    update_node_inputs,
    remove_nodes,
    make_output_shapes_attr,
)
from graph_optimizer.utils.visualization import graph_to_mermaid, graph_to_dot

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# create_node / create_const_node
# ---------------------------------------------------------------------------


def test_create_node_basic():
    n = create_node("Add", "my_add", inputs=["a", "b"])
    assert n.op == "Add"
    assert n.name == "my_add"
    assert list(n.input) == ["a", "b"]


def test_create_node_no_inputs():
    n = create_node("Placeholder", "ph")
    assert list(n.input) == []


def test_create_node_with_attr():
    attr = {"dtype": attr_value_pb2.AttrValue(type=tf.float32.as_datatype_enum)}
    n = create_node("Const", "c", attr=attr)
    assert "dtype" in n.attr


def test_create_const_node_int():
    n = create_const_node("c_int", 42, dtype="int32", shape=[])
    assert n.name == "c_int"
    assert n.op == "Const"


def test_create_const_node_float_list():
    n = create_const_node("c_f", [1.0, 2.0, 3.0], dtype="float32", shape=[3])
    assert n.op == "Const"


# ---------------------------------------------------------------------------
# extract_base_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("node", "node"),
        ("node:0", "node"),
        ("node:1", "node"),
        ("^ctrl", "ctrl"),
        ("^ctrl:0", "ctrl"),  # unusual but should strip both
        ("a/b/c", "a/b/c"),  # slashes preserved
    ],
)
def test_extract_base_name(raw, expected):
    assert extract_base_name(raw) == expected


# ---------------------------------------------------------------------------
# canonicalize_axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis,rank,expected",
    [
        (0, 4, 0),
        (3, 4, 3),
        (-1, 4, 3),
        (-2, 4, 2),
        (None, 4, None),  # unknown rank passthrough
        (1, None, 1),  # unknown rank passthrough
    ],
)
def test_canonicalize_axis(axis, rank, expected):
    assert canonicalize_axis(axis, rank) == expected


# ---------------------------------------------------------------------------
# compute_reference_counts
# ---------------------------------------------------------------------------


def _gd(*nodes):
    gd = tf.GraphDef()
    gd.node.extend(nodes)
    return gd


def test_ref_counts_simple_chain():
    gd = _gd(
        create_node("Const", "a"),
        create_node("Add", "b", inputs=["a", "a"]),  # a used twice
        create_node("Mul", "c", inputs=["a", "b"]),  # a once, b once
    )
    refs = compute_reference_counts(gd)
    assert refs["a"] == 3
    assert refs["b"] == 1
    assert refs["c"] == 0


def test_ref_counts_control_dep_counted():
    gd = _gd(
        create_node("NoOp", "ctrl"),
        create_node("Const", "a", inputs=["^ctrl"]),
    )
    refs = compute_reference_counts(gd)
    assert refs["ctrl"] >= 1


def test_ref_counts_zero_for_orphan():
    gd = _gd(create_node("Const", "orphan"))
    refs = compute_reference_counts(gd)
    assert refs["orphan"] == 0


# ---------------------------------------------------------------------------
# build_consumer_index
# ---------------------------------------------------------------------------


def test_consumer_index_basic():
    gd = _gd(
        create_node("Const", "a"),
        create_node("Add", "b", inputs=["a", "a"]),
        create_node("Mul", "c", inputs=["a", "b"]),
    )
    idx = build_consumer_index(gd)
    assert "b" in idx["a"]
    assert "c" in idx["a"]
    assert "c" in idx["b"]


def test_consumer_index_control_dep_stripped():
    gd = _gd(
        create_node("NoOp", "ctrl"),
        create_node("Const", "a", inputs=["^ctrl"]),
    )
    idx = build_consumer_index(gd)
    assert "a" in idx["ctrl"]


# ---------------------------------------------------------------------------
# remove_nodes
# ---------------------------------------------------------------------------


def test_remove_nodes_removes_exactly():
    gd = _gd(
        create_node("Const", "a"),
        create_node("Const", "b"),
        create_node("Const", "c"),
    )
    result = remove_nodes(gd, {"b"})
    names = [n.name for n in result.node]
    assert "a" in names and "c" in names and "b" not in names


# ---------------------------------------------------------------------------
# prune_dead_nodes
# ---------------------------------------------------------------------------


def test_prune_dead_nodes_removes_unreferenced_const():
    gd = _gd(
        create_node(
            "Const",
            "live",
        ),
        create_node("Const", "dead"),  # no consumers
        create_node("Identity", "out", inputs=["live"]),
    )
    refs_before = compute_reference_counts(gd)
    # Make dead have refs_before > 0 to simulate it was just disconnected
    refs_before["dead"] = 1
    result = prune_dead_nodes(gd, refs_before=refs_before)
    names = [n.name for n in result.node]
    assert "live" in names
    assert "dead" not in names


def test_prune_dead_nodes_placeholder_preserved():
    gd = _gd(create_node("Placeholder", "ph"))  # no consumers
    result = prune_dead_nodes(gd)
    assert any(n.name == "ph" for n in result.node)


def test_prune_dead_nodes_protected_preserved():
    gd = _gd(
        create_node("Const", "protected"),
        create_node("Const", "unprotected"),
    )
    refs = {"protected": 0, "unprotected": 0}
    result = prune_dead_nodes(gd, refs_before=refs, protected_nodes={"protected"})
    names = [n.name for n in result.node]
    assert "protected" in names


# ---------------------------------------------------------------------------
# final_prune
# ---------------------------------------------------------------------------


def test_final_prune_removes_all_dead():
    # Chain: a → b → c (only c has no consumers, then b, then a)
    gd = _gd(
        create_node("Const", "a"),
        create_node("Identity", "b", inputs=["a"]),
        create_node("Identity", "c", inputs=["b"]),  # all dead
    )
    result = final_prune(gd)
    # No consumers for anything → all should be pruned (a is Const, b/c not Placeholder)
    names = [n.name for n in result.node]
    assert "b" not in names
    assert "c" not in names


def test_final_prune_respects_protected():
    gd = _gd(
        create_node("Const", "a"),
        create_node("Identity", "out", inputs=["a"]),
    )
    result = final_prune(gd, protected_nodes={"out"})
    names = [n.name for n in result.node]
    assert "out" in names
    assert "a" in names  # kept because out needs it


# ---------------------------------------------------------------------------
# update_node_inputs
# ---------------------------------------------------------------------------


def test_update_node_inputs_replaces_mapping():
    n = create_node("Add", "add", inputs=["old_a", "old_b"])
    update_node_inputs(n, node_mapping={"old_a": "new_a"})
    assert "new_a" in n.input
    assert "old_a" not in n.input
    assert "old_b" in n.input


def test_update_node_inputs_hoists_ctrl_dep():
    n = create_node("Add", "add", inputs=["x"])
    update_node_inputs(n, node_mapping={}, hoisted_controls=["^ctrl"])
    assert "^ctrl" in n.input


def test_update_node_inputs_no_self_ctrl_loop():
    """A hoisted ctrl dep must never create ^add on add itself."""
    n = create_node("Add", "add", inputs=["x"])
    update_node_inputs(n, node_mapping={}, hoisted_controls=["^add"])
    assert "^add" not in n.input


# ---------------------------------------------------------------------------
# make_output_shapes_attr
# ---------------------------------------------------------------------------


def test_make_output_shapes_attr_structure():
    attr = make_output_shapes_attr([[2, 4], [8]])
    # Should be a valid AttrValue
    assert attr is not None


# ---------------------------------------------------------------------------
# visualization: graph_to_mermaid
# ---------------------------------------------------------------------------


def test_mermaid_contains_all_node_names():
    gd = _gd(
        create_node("Const", "alpha"),
        create_node("Identity", "beta", inputs=["alpha"]),
    )
    out = graph_to_mermaid(gd)
    assert "alpha" in out
    assert "beta" in out


def test_mermaid_default_direction_td():
    gd = _gd(create_node("Const", "x"))
    assert "graph TD" in graph_to_mermaid(gd)


def test_mermaid_explicit_direction_lr():
    gd = _gd(create_node("Const", "x"))
    assert "graph LR" in graph_to_mermaid(gd, direction="LR")


def test_mermaid_control_dep_is_dotted():
    gd = _gd(
        create_node("NoOp", "ctrl"),
        create_node("Const", "a", inputs=["^ctrl"]),
    )
    out = graph_to_mermaid(gd)
    assert "control" in out


def test_mermaid_data_edge_uses_arrow():
    gd = _gd(
        create_node("Const", "a"),
        create_node("Identity", "b", inputs=["a"]),
    )
    out = graph_to_mermaid(gd)
    assert "-->" in out


# ---------------------------------------------------------------------------
# visualization: graph_to_dot
# ---------------------------------------------------------------------------


def test_dot_starts_with_digraph():
    gd = _gd(create_node("Const", "node1"))
    out = graph_to_dot(gd)
    assert out.strip().startswith("digraph G {")


def test_dot_ends_with_brace():
    gd = _gd(create_node("Const", "x"))
    assert graph_to_dot(gd).strip().endswith("}")


def test_dot_contains_node_label():
    gd = _gd(create_node("Placeholder", "inp"))
    out = graph_to_dot(gd)
    assert "inp" in out


def test_dot_control_dep_is_dotted():
    gd = _gd(
        create_node("NoOp", "ctrl"),
        create_node("Const", "a", inputs=["^ctrl"]),
    )
    out = graph_to_dot(gd)
    assert "dotted" in out


def test_dot_data_edge_plain_arrow():
    gd = _gd(
        create_node("Const", "a"),
        create_node("Identity", "b", inputs=["a"]),
    )
    out = graph_to_dot(gd)
    # plain arrow: "a" -> "b";
    assert "->" in out
