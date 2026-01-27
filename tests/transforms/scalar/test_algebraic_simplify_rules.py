"""
AlgebraicSimplifyPass Rules Tests
===================================

Tests for scalar algebraic simplification rules.
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import GraphOptimizer, PassRegistry
from graph_optimizer.utils.graph_utils import create_node, create_const_node, make_output_shapes_attr

tf.disable_v2_behavior()

class AlgebraicSimplifyRulesTest(unittest.TestCase):
    def create_graph(self, nodes):
        """Helper to create a GraphDef from node list."""
        graph_def = tf.GraphDef()
        graph_def.node.extend(nodes)
        return graph_def

    def run_pass(self, graph, pass_name, protected_nodes=None):
        """Helper to run a single registered pass."""
        optimizer = GraphOptimizer(graph)
        pass_instance = PassRegistry.get_pass(pass_name)
        pass_instance.transform(optimizer, auto_cleanup=True, protected_nodes=protected_nodes or [])
        return optimizer.graph_def

    def test_add_zero(self):
        x = create_node("Placeholder", name="x")
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        add = create_node("Add", name="add", inputs=["x", "zero"])
        graph = self.create_graph([x, zero, add])

        optimized_graph = self.run_pass(graph, "simplify_add")
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("add", names)

    def test_sub_zero(self):
        x = create_node("Placeholder", name="x")
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        sub = create_node("Sub", name="sub", inputs=["x", "zero"])
        graph = self.create_graph([x, zero, sub])

        optimized_graph = self.run_pass(graph, "simplify_sub")
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("sub", names)

    def test_mul_one(self):
        x = create_node("Placeholder", name="x")
        one = create_const_node("one", value=1, dtype="float32", shape=[])
        mul = create_node("Mul", name="mul", inputs=["x", "one"])
        graph = self.create_graph([x, one, mul])

        optimized_graph = self.run_pass(graph, "simplify_mul")
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("mul", names)

    def test_mul_zero(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        mul = create_node("Mul", name="mul", inputs=["x", "zero"])
        mul.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[]]))
        # Add a sink to ensure the output of the mul is protected
        sink = create_node("Identity", "sink", inputs=["mul"])
        graph = self.create_graph([x, zero, mul, sink])

        optimized_graph = self.run_pass(graph, "simplify_mul", protected_nodes=["sink"])

        # After optimization, the graph should contain a new const node, and the mul node should be gone
        optimized_nodes = {n.name: n for n in optimized_graph.node}
        self.assertNotIn("mul", optimized_nodes)

        # The sink node should now be connected to the new zero constant
        sink_node = optimized_nodes["sink"]
        self.assertTrue("mul_zero" in sink_node.input[0])

        # And the new zero constant should exist
        self.assertTrue(any(n.op == "Const" and "mul_zero" in n.name for n in optimized_graph.node))

    def test_div_one(self):
        x = create_node("Placeholder", name="x")
        one = create_const_node("one", value=1, dtype="float32", shape=[])
        div = create_node("Div", name="div", inputs=["x", "one"])
        graph = self.create_graph([x, one, div])

        optimized_graph = self.run_pass(graph, "simplify_div")
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("div", names)

    def test_neg_neg(self):
        x = create_node("Placeholder", name="x")
        neg1 = create_node("Neg", name="neg1", inputs=["x"])
        neg2 = create_node("Neg", name="neg2", inputs=["neg1"])
        graph = self.create_graph([x, neg1, neg2])

        optimized_graph = self.run_pass(graph, "simplify_neg")
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("neg1", names)
        self.assertNotIn("neg2", names)

    def test_logical_not_not(self):
        x = create_node("Placeholder", name="x")
        not1 = create_node("LogicalNot", name="not1", inputs=["x"])
        not2 = create_node("LogicalNot", name="not2", inputs=["not1"])
        graph = self.create_graph([x, not1, not2])

        optimized_graph = self.run_pass(graph, "simplify_logical_not")
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("not1", names)
        self.assertNotIn("not2", names)

    def test_equal_same(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        eq = create_node("Equal", name="eq", inputs=["x", "x"])
        graph = self.create_graph([x, eq])

        optimized_graph = self.run_pass(graph, "simplify_redundant_comparison", protected_nodes=["eq"])
        trues = [n for n in optimized_graph.node if n.op == "Const" and n.name == "eq"]
        self.assertEqual(len(trues), 1)

    def test_select_same_branch(self):
        cond = create_node("Placeholder", name="cond")
        x = create_node("Placeholder", name="x")
        sel = create_node("Select", name="sel", inputs=["cond", "x", "x"])
        graph = self.create_graph([cond, x, sel])

        optimized_graph = self.run_pass(graph, "simplify_select")
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("sel", names)

if __name__ == "__main__":
    unittest.main()
