import tensorflow.compat.v1 as tf
from graph_optimizer.core import (
    GraphOptimizer,
    Op,
    RewriteResult,
    MultiOutputPattern,
    Any,
)
from graph_optimizer.utils.graph_utils import create_node


def test_infinite_loop_detection():
    """Verify that the optimizer detects and breaks infinite loops."""
    graph_def = tf.GraphDef()
    a = create_node("Const", "A")
    graph_def.node.append(a)

    optimizer = GraphOptimizer(graph_def)

    # A rewriter that replaces A with A (identity transform that always matches)
    def infinite_rewriter(match, opt):
        return RewriteResult(new_nodes=[create_node("Const", "A")])

    optimizer.add_transformation(Op("Const"), infinite_rewriter)

    # This should not hang; it should break after a few iterations due to loop detection
    optimized = optimizer.optimize(max_iterations=20, protected_nodes=["A"])
    assert len(optimized.node) == 1
    # Check if a log warning was issued (manual check easiest, but passing is proof it didn't hang)


def test_multi_output_localized_matching():
    """Verify MultiOutputPattern matches connected sinks efficiently."""
    graph_def = tf.GraphDef()
    # A -> B -> C (Output 1)
    # A -> D -> E (Output 2)
    a = create_node("Const", "A")
    b = create_node("Identity", "B", inputs=["A"])
    c = create_node("Identity", "C", inputs=["B"])
    d = create_node("Identity", "D", inputs=["A"])
    e = create_node("Identity", "E", inputs=["D"])
    graph_def.node.extend([a, b, c, d, e])

    optimizer = GraphOptimizer(graph_def)

    # Define a pattern matching both C and E outputs sharing ancestor A
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

    match = pattern.match(optimizer.nodes["C"], optimizer)
    assert match is not None
    assert match.matched_nodes["out1"].name == "C"
    assert match.matched_nodes["out2"].name == "E"
    assert match.matched_nodes["shared_root"].name == "A"


def test_control_dep_cycle_prevention():
    """Verify that rewriters don't introduce self-control-dependency cycles."""
    graph_def = tf.GraphDef()
    trig = create_node("NoOp", "trig")
    a = create_node("Const", "A", inputs=["^trig"])
    graph_def.node.extend([trig, a])

    optimizer = GraphOptimizer(graph_def)

    # Rewriter that replaces A and accidentally tries to keep control dep on its own new name
    def cycle_rewriter(match, opt):
        # Result node name matches the original node name
        new_a = create_node("Const", "A")
        # matcher.py will try to hoist ^trig to new_a
        return RewriteResult(new_nodes=[new_a])

    optimizer.add_transformation(Op("Const", alias="root"), cycle_rewriter)
    # Protect A to prevent pruning
    optimized = optimizer.optimize(protected_nodes=["A"])

    node_a = next(n for n in optimized.node if n.name == "A")
    assert "^trig" in node_a.input
    assert "^A" not in node_a.input


def test_control_dep_self_loop_prevention():
    graph_def = tf.GraphDef()
    a = create_node("Const", "A")
    b = create_node("Identity", "B", inputs=["A"])
    # A has control dep on B (potential cycle if hoisted)
    a.input.append("^B")
    graph_def.node.extend([a, b])

    optimizer = GraphOptimizer(graph_def)

    def rewriter(match, opt):
        # Replaces A with nothing, remaps A to B.
        # If A had a control input ^B, then B would get ^B.
        return RewriteResult(new_nodes=[], node_mapping={"A": "B"})

    optimizer.add_transformation(Op("Const", alias="root"), rewriter)
    # This should not result in B having ^B
    optimized = optimizer.optimize(protected_nodes=["B"])
    node_b = next(n for n in optimized.node if n.name == "B")
    assert "^B" not in node_b.input
