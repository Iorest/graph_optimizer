"""
AlgebraicSimplifyPass Tests
===========================

Tests for the refactored, efficient scalar algebraic simplification pass.
"""

import unittest
import tensorflow.compat.v1 as tf
import numpy as np
from graph_optimizer.core import GraphOptimizer, PassRegistry
from graph_optimizer.utils.graph_utils import create_node, create_const_node

tf.disable_v2_behavior()

def _make_placeholder(name, dtype, shape):
    """Helper to create a placeholder node with dtype and shape attributes."""
    node = create_node("Placeholder", name=name)
    node.attr["dtype"].type = dtype.as_datatype_enum
    if shape is not None:
        node.attr["_output_shapes"].list.shape.extend([tf.TensorShape(shape).as_proto()])
    return node

class AlgebraicSimplifyPassTest(unittest.TestCase):
    def setUp(self):
        """Instantiate the pass from the registry."""
        self.simplify_pass = PassRegistry.get_pass("algebraic_simplify")
        self.assertIsNotNone(self.simplify_pass, "Pass not found in registry")

    def create_graph(self, nodes):
        """Helper to create a GraphDef from node list."""
        graph_def = tf.GraphDef()
        graph_def.node.extend(nodes)
        return graph_def

    def run_pass(self, graph, protected_nodes=None):
        """Helper to run the simplification pass on a graph."""
        optimizer = GraphOptimizer(graph)
        self.simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=protected_nodes
        )
        return optimizer.graph_def

    def test_add_zero(self):
        x = _make_placeholder("x", tf.float32, [])
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        add = create_node("Add", name="add", inputs=["x", "zero"])
        add.attr["T"].type = tf.float32.as_datatype_enum
        add.attr["_output_shapes"].list.shape.extend([tf.TensorShape([]).as_proto()])
        graph = self.create_graph([x, zero, add])

        optimized_graph = self.run_pass(graph)
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("add", names)

    def test_sub_zero(self):
        x = _make_placeholder("x", tf.float32, [])
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        sub = create_node("Sub", name="sub", inputs=["x", "zero"])
        sub.attr["T"].type = tf.float32.as_datatype_enum
        sub.attr["_output_shapes"].list.shape.extend([tf.TensorShape([]).as_proto()])
        graph = self.create_graph([x, zero, sub])

        optimized_graph = self.run_pass(graph)
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("sub", names)

    def test_sub_same(self):
        x = _make_placeholder("x", tf.float32, [2, 2])
        sub = create_node("Sub", name="sub", inputs=["x", "x"])
        sub.attr["T"].type = tf.float32.as_datatype_enum
        graph = self.create_graph([x, sub])
        optimized_graph = self.run_pass(graph)
        self.assertEqual(len(optimized_graph.node), 1)
        const_node = [n for n in optimized_graph.node if n.op == "Const"][0]
        self.assertEqual(const_node.name, "sub")

    def test_add_neg(self):
        x = _make_placeholder("x", tf.float32, [2, 2])
        neg = create_node("Neg", name="neg", inputs=[x.name])
        neg.attr["T"].type = tf.float32.as_datatype_enum
        add = create_node("Add", name="add", inputs=[x.name, neg.name])
        add.attr["T"].type = tf.float32.as_datatype_enum
        add.attr["_output_shapes"].list.shape.extend([tf.TensorShape([2, 2]).as_proto()])
        graph = self.create_graph([x, neg, add])
        optimized_graph = self.run_pass(graph)
        self.assertEqual(len(optimized_graph.node), 1)
        const_node = [n for n in optimized_graph.node if n.op == "Const"][0]
        self.assertEqual(const_node.name, "add")

    def test_mul_one(self):
        x = _make_placeholder("x", tf.float32, [])
        one = create_const_node("one", value=1, dtype="float32", shape=[])
        mul = create_node("Mul", name="mul", inputs=["x", "one"])
        mul.attr["T"].type = tf.float32.as_datatype_enum
        mul.attr["_output_shapes"].list.shape.extend([tf.TensorShape([]).as_proto()])
        graph = self.create_graph([x, one, mul])

        optimized_graph = self.run_pass(graph)
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("mul", names)

    def test_mul_zero(self):
        x = _make_placeholder("x", tf.float32, [2, 2])
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        mul = create_node("Mul", name="mul", inputs=[x.name, zero.name])
        mul.attr["T"].type = tf.float32.as_datatype_enum
        mul.attr["_output_shapes"].list.shape.extend([tf.TensorShape([2, 2]).as_proto()])
        graph = self.create_graph([x, zero, mul])
        optimized_graph = self.run_pass(graph)
        self.assertEqual(len(optimized_graph.node), 1)
        const_node = [n for n in optimized_graph.node if n.op == "Const"][0]
        self.assertEqual(const_node.name, "mul")

    def test_mul_same(self):
        x = _make_placeholder("x", tf.float32, [])
        mul = create_node("Mul", name="mul", inputs=["x", "x"])
        mul.attr["T"].type = tf.float32.as_datatype_enum
        graph = self.create_graph([x, mul])
        optimized_graph = self.run_pass(graph)
        self.assertTrue(any(n.op == "Square" for n in optimized_graph.node))
        self.assertFalse(any(n.op == "Mul" for n in optimized_graph.node))

    def test_div_one(self):
        x = _make_placeholder("x", tf.float32, [])
        one = create_const_node("one", value=1, dtype="float32", shape=[])
        div = create_node("Div", name="div", inputs=["x", "one"])
        div.attr["T"].type = tf.float32.as_datatype_enum
        div.attr["_output_shapes"].list.shape.extend([tf.TensorShape([]).as_proto()])
        graph = self.create_graph([x, one, div])

        optimized_graph = self.run_pass(graph)
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("div", names)

    def test_div_same(self):
        x = _make_placeholder("x", tf.float32, [2, 2])
        div = create_node("Div", name="div", inputs=["x", "x"])
        div.attr["T"].type = tf.float32.as_datatype_enum
        graph = self.create_graph([x, div])
        optimized_graph = self.run_pass(graph)
        self.assertEqual(len(optimized_graph.node), 1)
        const_node = [n for n in optimized_graph.node if n.op == "Const"][0]
        self.assertEqual(const_node.name, "div")

    def test_double_negation(self):
        x = _make_placeholder("x", tf.float32, [])
        neg1 = create_node("Neg", name="neg1", inputs=["x"])
        neg2 = create_node("Neg", name="neg2", inputs=["neg1"])
        graph = self.create_graph([x, neg1, neg2])

        optimized_graph = self.run_pass(graph)
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("neg1", names)
        self.assertNotIn("neg2", names)

    def test_identity_comparison(self):
        x = _make_placeholder("x", tf.float32, [])
        eq = create_node("Equal", name="eq", inputs=["x", "x"])
        eq.attr["T"].type = tf.float32.as_datatype_enum
        graph = self.create_graph([x, eq])

        optimized_graph = self.run_pass(graph)

        self.assertEqual(len(optimized_graph.node), 1)
        const_node = [n for n in optimized_graph.node if n.op == "Const"][0]
        self.assertEqual(const_node.attr['dtype'].type, tf.bool.as_datatype_enum)
        self.assertTrue(const_node.attr['value'].tensor.bool_val[0])

    def test_select_same_branches(self):
        cond = _make_placeholder("cond", tf.bool, [])
        x = _make_placeholder("x", tf.float32, [])
        sel = create_node("Select", name="sel", inputs=["cond", "x", "x"])
        graph = self.create_graph([cond, x, sel])

        optimized_graph = self.run_pass(graph)
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertIn("cond", names)
        self.assertNotIn("sel", names)

    def test_add_zero_broadcast_is_safe(self):
        x = _make_placeholder("x", tf.float32, [2, 2])
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        add = create_node("Add", name="add", inputs=["x", "zero"])
        add.attr["T"].type = tf.float32.as_datatype_enum
        add.attr["_output_shapes"].list.shape.extend([tf.TensorShape([2, 2]).as_proto()])
        graph = self.create_graph([x, zero, add])

        optimized_graph = self.run_pass(graph)
        names = {n.name for n in optimized_graph.node}
        self.assertIn("x", names)
        self.assertNotIn("add", names)

    def test_add_zero_broadcast_is_unsafe(self):
        x = _make_placeholder("x", tf.float32, [])
        zero = create_const_node("zero", value=np.zeros((2,2)), dtype="float32", shape=[2, 2])
        add = create_node("Add", name="add", inputs=["x", "zero"])
        add.attr["T"].type = tf.float32.as_datatype_enum
        add.attr["_output_shapes"].list.shape.extend([tf.TensorShape([2, 2]).as_proto()])
        graph = self.create_graph([x, zero, add])

        optimized_graph = self.run_pass(graph)
        names = {n.name for n in optimized_graph.node}
        self.assertIn("add", names)

    def test_logical_and_false(self):
        x = _make_placeholder("x", tf.bool, [2, 2])
        false_const = create_const_node("false", value=False, dtype="bool", shape=[])
        land = create_node("LogicalAnd", name="land", inputs=[x.name, false_const.name])
        land.attr["_output_shapes"].list.shape.extend([tf.TensorShape([2, 2]).as_proto()])
        graph = self.create_graph([x, false_const, land])
        optimized_graph = self.run_pass(graph)
        self.assertEqual(len(optimized_graph.node), 1)
        const_node = [n for n in optimized_graph.node if n.op == "Const"][0]
        self.assertEqual(const_node.name, "land")

    def test_logical_or_true(self):
        x = _make_placeholder("x", tf.bool, [2, 2])
        true_const = create_const_node("true", value=True, dtype="bool", shape=[])
        lor = create_node("LogicalOr", name="lor", inputs=[x.name, true_const.name])
        lor.attr["_output_shapes"].list.shape.extend([tf.TensorShape([2, 2]).as_proto()])
        graph = self.create_graph([x, true_const, lor])
        optimized_graph = self.run_pass(graph)
        self.assertEqual(len(optimized_graph.node), 1)
        const_node = [n for n in optimized_graph.node if n.op == "Const"][0]
        self.assertEqual(const_node.name, "lor")

if __name__ == "__main__":
    unittest.main()
