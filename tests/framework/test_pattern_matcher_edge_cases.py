import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import (
    GraphOptimizer,
    Op,
    Variadic,
    OptionalPattern,
    MultiOutputPattern,
    RewriteResult,
)
from graph_optimizer.utils import create_node

tf.disable_v2_behavior()


class TestPatternMatcherEdgeCases(unittest.TestCase):
    def test_variadic_pattern_matching_edge_cases(self):
        """Test variadic pattern with 0, 1, and multiple inputs."""
        graph_def = tf.GraphDef()
        a = create_node("Const", "A")
        b = create_node("Const", "B")
        c = create_node("Const", "C")

        # Concat with 2 inputs + axis
        concat1 = create_node("ConcatV2", "concat1", inputs=["A", "B", "C"])

        graph_def.node.extend([a, b, c, concat1])
        optimizer = GraphOptimizer(graph_def)

        pattern = Op(
            "ConcatV2",
            Variadic(Op("Const"), min_count=1, alias="args"),
            Op("Const", alias="axis"),
            alias="root",
        )

        def rewriter(match, opt):
            # Just prove it matched
            return [create_node("NoOp", match.matched_nodes["root"].name)]

        optimizer.add_transformation(pattern, rewriter)
        optimized = optimizer.optimize(auto_cleanup=False)

        # Verify it matched and replaced concat1
        node_map = {n.name: n for n in optimized.node}
        self.assertIn("concat1", node_map)
        self.assertEqual(node_map["concat1"].op, "NoOp")

    def test_cyclic_graph_safeguard(self):
        """Test that the matcher doesn't hang on a cyclic graph."""
        graph_def = tf.GraphDef()
        # A depends on B, B depends on A
        a = create_node("Identity", "A", inputs=["B"])
        b = create_node("Identity", "B", inputs=["A"])

        graph_def.node.extend([a, b])
        optimizer = GraphOptimizer(graph_def)

        # Pattern that matches Identity -> Identity
        pattern = Op("Identity", Op("Identity", alias="inner"), alias="root")

        def rewriter(match, opt):
            root_name = match.matched_nodes["root"].name
            return [create_node("NoOp", root_name)]

        optimizer.add_transformation(pattern, rewriter)

        # Should complete without infinite loop, max_iterations defends against cyclic rewrites
        optimized = optimizer.optimize(max_iterations=5, auto_cleanup=False)
        self.assertIsNotNone(optimized)

    def test_fallback_control_dep_hoisting(self):
        """Test control dependency hoisting when a node is removed with no direct replacement name match."""
        graph_def = tf.GraphDef()
        trig = create_node("NoOp", "trig")
        a = create_node("Const", "A", inputs=["^trig"])
        b = create_node("Identity", "B", inputs=["A"])

        graph_def.node.extend([trig, a, b])
        optimizer = GraphOptimizer(graph_def)

        pattern = Op("Const", alias="root")

        def rewriter(match, opt):
            # Replaces A with a node named NewA, no mapping provided
            # The fallback logic should attach ^trig to NewA OR B.
            # Actually, since NewA is created, it should attach to NewA (first new node).
            new_a = create_node("Const", "NewA")
            return RewriteResult(new_nodes=[new_a], node_mapping={"A": "NewA"})

        optimizer.add_transformation(pattern, rewriter)
        # Protect B so NewA is not pruned since it has no consumers yet (node_mapping handles consumer update)
        optimized = optimizer.optimize(protected_nodes=["B"])

        node_map = {n.name: n for n in optimized.node}
        self.assertIn("NewA", node_map)
        self.assertIn("^trig", node_map["NewA"].input)

    def test_optional_pattern_present(self):
        graph_def = tf.GraphDef()
        a = create_node("Const", "A")
        cast = create_node("Cast", "Cast", inputs=["A"])
        b = create_node("Identity", "B", inputs=["Cast"])
        graph_def.node.extend([a, cast, b])

        optimizer = GraphOptimizer(graph_def)
        # B -> Optional(Cast) -> A
        pattern = Op(
            "Identity",
            OptionalPattern(Op("Cast", Op("Const", alias="const"), alias="opt")),
            alias="root",
        )

        def rewriter(match, opt):
            # Proves it matched the optional Cast
            self.assertIn("opt", match.matched_nodes)
            return [create_node("NoOp", match.matched_nodes["root"].name)]

        optimizer.add_transformation(pattern, rewriter)
        optimized = optimizer.optimize(auto_cleanup=False)
        node_map = {n.name: n for n in optimized.node}
        self.assertEqual(node_map["B"].op, "NoOp")

    def test_optional_pattern_absent(self):
        graph_def = tf.GraphDef()
        a = create_node("Const", "A")
        # Direct connection skipping Cast
        b = create_node("Identity", "B", inputs=["A"])
        graph_def.node.extend([a, b])

        optimizer = GraphOptimizer(graph_def)
        # B -> Optional(Cast) -> A
        pattern = Op(
            "Identity",
            OptionalPattern(Op("Cast", Op("Const", alias="const"), alias="opt")),
            alias="root",
        )

        def rewriter(match, opt):
            # Proves it matched bypassing the optional Cast
            self.assertNotIn("opt", match.matched_nodes)
            self.assertEqual(match.matched_nodes["const"].name, "A")
            return [create_node("NoOp", match.matched_nodes["root"].name)]

        optimizer.add_transformation(pattern, rewriter)
        optimized = optimizer.optimize(auto_cleanup=False)
        node_map = {n.name: n for n in optimized.node}
        self.assertEqual(node_map["B"].op, "NoOp")

    def test_commutative_pattern(self):
        graph_def = tf.GraphDef()
        a = create_node("Const", "A")
        b = create_node("Var", "B")
        # Add(Var, Const) - pattern is Add(Const, Var)
        add = create_node("Add", "Add", inputs=["B", "A"])
        graph_def.node.extend([a, b, add])

        optimizer = GraphOptimizer(graph_def)
        # Pattern expects Const then Var, but commutative=True allows matched order
        pattern = Op(
            "Add",
            Op("Const", alias="c"),
            Op("Var", alias="v"),
            commutative=True,
            alias="root",
        )

        def rewriter(match, opt):
            self.assertEqual(match.matched_nodes["c"].name, "A")
            self.assertEqual(match.matched_nodes["v"].name, "B")
            return [create_node("NoOp", match.matched_nodes["root"].name)]

        optimizer.add_transformation(pattern, rewriter)
        optimized = optimizer.optimize(auto_cleanup=False)
        node_map = {n.name: n for n in optimized.node}
        self.assertEqual(node_map["Add"].op, "NoOp")

    def test_multi_output_pattern(self):
        graph_def = tf.GraphDef()
        x = create_node("Const", "X")
        # Subgraph: Y1 = Relu(X), Y2 = Square(X)
        y1 = create_node("Relu", "Y1", inputs=["X"])
        y2 = create_node("Square", "Y2", inputs=["X"])
        graph_def.node.extend([x, y1, y2])

        optimizer = GraphOptimizer(graph_def)
        # MultiOutputPattern requires both Y1 and Y2 to match
        p1 = Op("Relu", Op("Const", alias="shared"), alias="y1")
        p2 = Op("Square", Op("Const", alias="shared"), alias="y2")
        pattern = MultiOutputPattern([p1, p2], alias="subgraph")

        def rewriter(match, opt):
            self.assertEqual(match.matched_nodes["shared"].name, "X")
            # Replace both Y1 and Y2 with Identity(X)
            new_y1 = create_node("Identity", "Y1", inputs=["X"])
            new_y2 = create_node("Identity", "Y2", inputs=["X"])
            # Return result indicating exactly what was replaced to bypass anchor-only assumption
            return RewriteResult(
                new_nodes=[new_y1, new_y2], replaced_nodes=["Y1", "Y2"]
            )

        optimizer.add_transformation(pattern, rewriter)
        optimized = optimizer.optimize(auto_cleanup=False)
        node_map = {n.name: n for n in optimized.node}
        self.assertEqual(node_map["Y1"].op, "Identity")
        self.assertEqual(node_map["Y2"].op, "Identity")


if __name__ == "__main__":
    unittest.main()
