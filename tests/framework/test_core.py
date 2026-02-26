"""
Tests for the core TFGraphOptimizer engine.

Covers: optimizer construction, node lookup, graph execution lifecycle,
control dependency handling, and graph pruning.
"""

import tensorflow.compat.v1 as tf
from graph_optimizer.core.tensorflow import TFGraphOptimizer, Op
from graph_optimizer.utils.tf.graph_utils import create_node

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(*nodes):
    gd = tf.GraphDef()
    gd.node.extend(nodes)
    return TFGraphOptimizer(gd)


# ---------------------------------------------------------------------------
# Node tracking
# ---------------------------------------------------------------------------


def test_node_lookup():
    with tf.Graph().as_default():
        tf.placeholder(tf.float32, name="A")
        tf.constant(1.0, name="B")
        gd = tf.get_default_graph().as_graph_def()
    opt = TFGraphOptimizer(gd)
    assert "A" in opt.nodes
    assert "B" in opt.nodes


# ---------------------------------------------------------------------------
# Pattern primitives
# ---------------------------------------------------------------------------


def test_control_dependency_preserved_after_rewrite():
    """^control edges on removed nodes are hoisted to their replacement."""
    a = create_node("Placeholder", "a")
    c = create_node("Placeholder", "c")
    b = create_node("Identity", "b", inputs=["a", "^c"])
    opt = _make_graph(a, c, b)

    def rewriter(match, o):
        root = match.matched_nodes["root"]
        return [create_node("Identity", root.name, inputs=["a"])]

    opt.add_transformation(Op("Identity", alias="root"), rewriter)
    result = opt.optimize_patterns(auto_cleanup=False)
    new_b = next(n for n in result.node if n.name == "b")
    assert "^c" in new_b.input


# ---------------------------------------------------------------------------
# Graph pruning
# ---------------------------------------------------------------------------


def test_fundamental_pruning():
    """prune() keeps reachable nodes and removes dead ones."""
    with tf.Graph().as_default():
        tf.placeholder(tf.float32, name="A")
        tf.constant(1.0, name="B")
        add = tf.add(
            tf.placeholder(tf.float32, name="A_"),
            tf.constant(1.0, name="B_"),
            name="Add",
        )
        tf.identity(add, name="Y")
        tf.constant(2.0, name="Z")  # dead node
        gd = tf.get_default_graph().as_graph_def()
    opt = TFGraphOptimizer(gd)
    pruned = opt.prune(["Y"])
    names = [n.name for n in pruned.node]
    assert "Y" in names
    assert "Add" in names
    assert "Z" not in names


def test_preserve_placeholders():
    """final_prune() keeps Placeholder nodes even when they have no consumers."""
    gd = tf.GraphDef()
    gd.node.append(create_node("Placeholder", "unused"))
    opt = TFGraphOptimizer(gd)
    pruned = opt.final_prune(gd, "test")
    assert "unused" in [n.name for n in pruned.node]
