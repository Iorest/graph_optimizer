"""
Pattern-matching tests.

Covers: Op, CommutativeOp, Variadic, OptionalPattern, MultiOutputPattern,
cyclic graph safeguards, indexed pattern dispatch, and control-dep hoisting
during pattern rewrites.
"""

import tensorflow.compat.v1 as tf
from graph_optimizer.core.tensorflow import (
    TFGraphOptimizer,
    Op,
    Any,
    CommutativeOp,
    Variadic,
    OptionalPattern,
    MultiOutputPattern,
    RewriteResult,
)
from graph_optimizer.utils.tf.graph_utils import create_node

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# Pattern primitives — Op / CommutativeOp / Any / Variadic
# ---------------------------------------------------------------------------


def test_commutative_op_matches_reversed_inputs():
    """CommutativeOp(Add) must match even when inputs are written in reverse order."""
    pattern = CommutativeOp(
        "AddV2", Op("Const", alias="c"), Any(alias="x"), alias="root"
    )
    with tf.Graph().as_default():
        c = tf.constant(1.0, name="c")
        x = tf.placeholder(tf.float32, name="x")
        tf.add(x, c, name="add1")
        gd = tf.get_default_graph().as_graph_def()
    opt = TFGraphOptimizer(gd)
    assert pattern.match(opt.nodes["add1"], opt) is not None


def test_variadic_op_matches_multi_input():
    """Variadic matches a variable number of inputs on ConcatV2."""
    gd = tf.GraphDef()
    for i in range(1, 4):
        gd.node.append(create_node("Const", f"c{i}"))
    gd.node.append(create_node("Const", "axis"))
    concat = create_node("ConcatV2", "concat", inputs=["c1", "c2", "c3", "axis"])
    gd.node.append(concat)
    opt = TFGraphOptimizer(gd)
    pattern = Op("ConcatV2", Variadic(Op("Const")), Op("Const", alias="axis"))
    assert pattern.match(concat, opt) is not None


def test_control_dep_skipped_during_matching():
    """^ctrl edges are transparent to the matcher — Op(Identity, Ph) still matches."""
    a = create_node("Placeholder", "a")
    c = create_node("Placeholder", "c")
    b = create_node("Identity", "b", inputs=["a", "^c"])
    gd = tf.GraphDef()
    gd.node.extend([a, c, b])
    opt = TFGraphOptimizer(gd)
    assert Op("Identity", Op("Placeholder")).match(b, opt) is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opt(*nodes):
    gd = tf.GraphDef()
    gd.node.extend(nodes)
    return TFGraphOptimizer(gd)


# ---------------------------------------------------------------------------
# Variadic matching
# ---------------------------------------------------------------------------


def test_variadic_pattern_fires():
    """Variadic(min_count=1) matches ConcatV2 with 2 data inputs + 1 axis."""
    a, b, c = (
        create_node("Const", "A"),
        create_node("Const", "B"),
        create_node("Const", "C"),
    )
    concat = create_node("ConcatV2", "concat1", inputs=["A", "B", "C"])
    opt = _opt(a, b, c, concat)

    pattern = Op(
        "ConcatV2",
        Variadic(Op("Const"), min_count=1, alias="args"),
        Op("Const", alias="axis"),
        alias="root",
    )

    matched = []

    def rw(match, o):
        matched.append(match.matched_nodes["args"])
        return [create_node("NoOp", match.matched_nodes["root"].name)]

    opt.add_transformation(pattern, rw)
    result = opt.optimize_patterns(auto_cleanup=False)
    assert {n.name: n for n in result.node}["concat1"].op == "NoOp"
    assert len(matched) > 0


def test_variadic_alias_collects_list():
    """Variadic(..., alias='x') gives match.matched_nodes['x'] as a list."""
    c1, c2, c3 = [create_node("Const", f"c{i}") for i in range(1, 4)]
    root = create_node("ConcatV2", "root", inputs=["c1", "c2", "c3"])
    opt = _opt(c1, c2, c3, root)

    captured = []

    def rw(match, o):
        captured.append(match.matched_nodes["my_inputs"])
        return [match.matched_nodes["root"]]

    opt.add_transformation(
        Op("ConcatV2", Variadic(Op("Const"), alias="my_inputs"), alias="root"), rw
    )
    opt.optimize_patterns(auto_cleanup=False)
    assert len(captured) > 0
    assert len(captured[0]) == 3
    assert captured[0][0].name == "c1"


# ---------------------------------------------------------------------------
# Optional pattern
# ---------------------------------------------------------------------------


def test_optional_pattern_present():
    """Optional Cast is matched when it exists in the graph."""
    a = create_node("Const", "A")
    cast = create_node("Cast", "Cast", inputs=["A"])
    b = create_node("Identity", "B", inputs=["Cast"])
    opt = _opt(a, cast, b)

    pattern = Op(
        "Identity",
        OptionalPattern(Op("Cast", Op("Const", alias="const"), alias="opt")),
        alias="root",
    )

    saw_opt = []

    def rw(match, o):
        saw_opt.append("opt" in match.matched_nodes)
        return [create_node("NoOp", match.matched_nodes["root"].name)]

    opt.add_transformation(pattern, rw)
    opt.optimize_patterns(auto_cleanup=False)
    assert any(saw_opt)


def test_optional_pattern_absent():
    """Optional Cast is bypassed when not present; Const is matched directly."""
    a = create_node("Const", "A")
    b = create_node("Identity", "B", inputs=["A"])
    opt = _opt(a, b)

    pattern = Op(
        "Identity",
        OptionalPattern(Op("Cast", Op("Const", alias="const"), alias="opt")),
        alias="root",
    )

    saw_direct = []

    def rw(match, o):
        saw_direct.append(
            "opt" not in match.matched_nodes
            and match.matched_nodes["const"].name == "A"
        )
        return [create_node("NoOp", match.matched_nodes["root"].name)]

    opt.add_transformation(pattern, rw)
    opt.optimize_patterns(auto_cleanup=False)
    assert any(saw_direct)


# ---------------------------------------------------------------------------
# Commutative matching
# ---------------------------------------------------------------------------


def test_commutative_pattern():
    """commutative=True lets Add(Var, Const) match Op(Add, Const, Var)."""
    a = create_node("Const", "A")
    bv = create_node("Var", "B")
    add = create_node("Add", "Add", inputs=["B", "A"])
    opt = _opt(a, bv, add)

    pattern = Op(
        "Add",
        Op("Const", alias="c"),
        Op("Var", alias="v"),
        commutative=True,
        alias="root",
    )

    matched_names = []

    def rw(match, o):
        matched_names.append(
            (match.matched_nodes["c"].name, match.matched_nodes["v"].name)
        )
        return [create_node("NoOp", match.matched_nodes["root"].name)]

    opt.add_transformation(pattern, rw)
    opt.optimize_patterns(auto_cleanup=False)
    assert any(c == "A" and v == "B" for c, v in matched_names)


# ---------------------------------------------------------------------------
# MultiOutput pattern
# ---------------------------------------------------------------------------


def test_multi_output_pattern_matches_shared_root():
    """MultiOutputPattern matches two sinks sharing a common ancestor."""
    gd = tf.GraphDef()
    a = create_node("Const", "A")
    b = create_node("Identity", "B", inputs=["A"])
    c = create_node("Identity", "C", inputs=["B"])
    d = create_node("Identity", "D", inputs=["A"])
    e = create_node("Identity", "E", inputs=["D"])
    gd.node.extend([a, b, c, d, e])
    opt = TFGraphOptimizer(gd)

    pattern = MultiOutputPattern(
        [
            Op(
                "Identity",
                Op("Identity", Op("Const", alias="shared_root")),
                alias="out1",
            ),
            Op(
                "Identity",
                Op("Identity", Op("Const", alias="shared_root")),
                alias="out2",
            ),
        ]
    )
    match = pattern.match(opt.nodes["C"], opt)
    assert match is not None
    assert match.matched_nodes["out1"].name == "C"
    assert match.matched_nodes["out2"].name == "E"
    assert match.matched_nodes["shared_root"].name == "A"


def test_multi_output_pattern_rewrite():
    """MultiOutputPattern rewriter receives the shared node and replaces both sinks."""
    x = create_node("Const", "X")
    y1 = create_node("Relu", "Y1", inputs=["X"])
    y2 = create_node("Square", "Y2", inputs=["X"])
    opt = _opt(x, y1, y2)

    pattern = MultiOutputPattern(
        [
            Op("Relu", Op("Const", alias="shared"), alias="y1"),
            Op("Square", Op("Const", alias="shared"), alias="y2"),
        ],
        alias="subgraph",
    )

    def rw(match, o):
        assert match.matched_nodes["shared"].name == "X"
        return RewriteResult(
            new_nodes=[
                create_node("Identity", "Y1", inputs=["X"]),
                create_node("Identity", "Y2", inputs=["X"]),
            ],
            replaced_nodes=["Y1", "Y2"],
        )

    opt.add_transformation(pattern, rw)
    result = opt.optimize_patterns(auto_cleanup=False)
    nm = {n.name: n for n in result.node}
    assert nm["Y1"].op == "Identity"
    assert nm["Y2"].op == "Identity"


# ---------------------------------------------------------------------------
# Indexed pattern dispatch
# ---------------------------------------------------------------------------


def test_pattern_indexed_by_op_type():
    """Patterns are stored in pattern_index keyed by op type."""
    opt = TFGraphOptimizer(tf.GraphDef())
    opt.add_transformation(Op("Add"), lambda m, o: [])
    assert "Add" in opt.pattern_index
    assert len(opt.pattern_index["Add"]) == 1
    assert len(opt.wildcard_patterns) == 0


def test_wildcard_pattern_stored_separately():
    opt = TFGraphOptimizer(tf.GraphDef())
    opt.add_transformation(Op(None), lambda m, o: [])
    assert len(opt.wildcard_patterns) == 1


def test_wrong_op_pattern_never_fires():
    """A pattern registered for 'Mul' must NOT fire on an 'Add' node."""
    gd = tf.GraphDef()
    gd.node.append(tf.NodeDef(name="n1", op="Add"))
    opt = TFGraphOptimizer(gd)

    fired = []
    opt.add_transformation(Op("Mul"), lambda m, o: fired.append(True) or None)
    opt.optimize_patterns(auto_cleanup=False)
    assert not fired


# ---------------------------------------------------------------------------
# Safety: cyclic graph and loop detection
# ---------------------------------------------------------------------------


def test_cyclic_graph_does_not_hang():
    """Matcher must terminate on a cyclic graph (A→B→A)."""
    a = create_node("Identity", "A", inputs=["B"])
    b = create_node("Identity", "B", inputs=["A"])
    opt = _opt(a, b)

    pattern = Op("Identity", Op("Identity", alias="inner"), alias="root")
    opt.add_transformation(
        pattern, lambda m, o: [create_node("NoOp", m.matched_nodes["root"].name)]
    )
    result = opt.optimize_patterns(max_iterations=5, auto_cleanup=False)
    assert result is not None


def test_infinite_rewrite_loop_detection():
    """Rewriter that re-creates the origin node must not hang; terminates via max_iterations."""
    gd = tf.GraphDef()
    gd.node.append(create_node("Const", "A"))
    opt = TFGraphOptimizer(gd)
    opt.add_transformation(
        Op("Const"),
        lambda m, o: RewriteResult(new_nodes=[create_node("Const", "A")]),
    )
    result = opt.optimize_patterns(max_iterations=20, protected_nodes=["A"])
    assert len(result.node) == 1
