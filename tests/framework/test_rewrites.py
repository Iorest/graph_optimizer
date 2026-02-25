"""
Rewrite-engine correctness tests.

Covers: control dependency preservation, node mapping, self-loop prevention,
and regression tests for past bugs in the rewriter.
"""

import tensorflow.compat.v1 as tf
from graph_optimizer.core.tensorflow import (
    TFGraphOptimizer,
    Op,
    RewriteResult,
)
from graph_optimizer.utils import create_node

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# Control dependency hoisting
# ---------------------------------------------------------------------------


def test_internal_ctrl_dep_hoisted_to_replacement():
    """Control dep on an internal matched node must travel to the replacement."""
    gd = tf.GraphDef()
    gd.node.extend(
        [
            create_node("NoOp", "C"),
            create_node("Const", "A"),
            create_node("Identity", "inner", inputs=["A", "^C"]),
            create_node("Identity", "root", inputs=["inner"]),
        ]
    )
    opt = TFGraphOptimizer(gd)

    def rw(match, o):
        return [create_node("NoOp", match.matched_nodes["root"].name)]

    opt.add_transformation(
        Op("Identity", Op("Identity", alias="inner"), alias="root"), rw
    )
    result = opt.optimize_patterns(auto_cleanup=False)
    root_node = next(n for n in result.node if n.name == "root")
    assert "^C" in root_node.input, "Control dep from internal node was lost"


def test_ctrl_dep_preserved_via_node_mapping():
    """When a node is remapped, its control deps migrate to consumers."""
    gd = tf.GraphDef()
    gd.node.extend(
        [
            create_node("Const", "x"),
            create_node("Const", "zero"),
            create_node("NoOp", "trigger"),
            create_node("Add", "add", inputs=["x", "zero", "^trigger"]),
            create_node("Identity", "result", inputs=["add"]),
        ]
    )
    opt = TFGraphOptimizer(gd)

    def rw(match, o):
        return RewriteResult(
            new_nodes=[create_node("NoOp", "side")],
            replaced_nodes=[],
            node_mapping={"add": "x"},
        )

    opt.add_transformation(Op("Add", Op("Const"), Op("Const")), rw)
    result = opt.optimize_patterns(auto_cleanup=True, protected_nodes=["result"])
    nm = {n.name: n for n in result.node}
    assert "x" in nm["result"].input
    assert "^trigger" in nm["result"].input, "Control dep lost during node mapping"


def test_fallback_ctrl_dep_hoisted_to_first_new_node():
    """When a node is removed with a name-mapping, ^dep goes to the replacement."""
    gd = tf.GraphDef()
    gd.node.extend(
        [
            create_node("NoOp", "trig"),
            create_node("Const", "A", inputs=["^trig"]),
            create_node("Identity", "B", inputs=["A"]),
        ]
    )
    opt = TFGraphOptimizer(gd)

    def rw(match, o):
        return RewriteResult(
            new_nodes=[create_node("Const", "NewA")], node_mapping={"A": "NewA"}
        )

    opt.add_transformation(Op("Const", alias="root"), rw)
    result = opt.optimize_patterns(protected_nodes=["B"])
    nm = {n.name: n for n in result.node}
    assert "NewA" in nm
    assert "^trig" in nm["NewA"].input


# ---------------------------------------------------------------------------
# Self-loop / cycle prevention
# ---------------------------------------------------------------------------


def test_ctrl_dep_cycle_not_introduced():
    """Rewriter replacing A must not give its result node a self-loop (^A)."""
    gd = tf.GraphDef()
    gd.node.extend(
        [
            create_node("NoOp", "trig"),
            create_node("Const", "A", inputs=["^trig"]),
        ]
    )
    opt = TFGraphOptimizer(gd)

    def rw(match, o):
        return RewriteResult(new_nodes=[create_node("Const", "A")])

    opt.add_transformation(Op("Const", alias="root"), rw)
    result = opt.optimize_patterns(protected_nodes=["A"])
    node_a = next(n for n in result.node if n.name == "A")
    assert "^trig" in node_a.input
    assert "^A" not in node_a.input


def test_ctrl_dep_self_loop_not_propagated():
    """If A has ^B as a control dep and A is remapped to B, B must not get ^B."""
    gd = tf.GraphDef()
    a = create_node("Const", "A")
    b = create_node("Identity", "B", inputs=["A"])
    a.input.append("^B")  # A → ^B: would create a cycle if hoisted to B
    gd.node.extend([a, b])
    opt = TFGraphOptimizer(gd)

    def rw(match, o):
        return RewriteResult(new_nodes=[], node_mapping={"A": "B"})

    opt.add_transformation(Op("Const", alias="root"), rw)
    result = opt.optimize_patterns(protected_nodes=["B"])
    node_b = next(n for n in result.node if n.name == "B")
    assert "^B" not in node_b.input
