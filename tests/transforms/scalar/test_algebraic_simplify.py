"""
AlgebraicSimplifyPass Tests
===========================

Tests for scalar algebraic simplification pass.
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.transforms.scalar.algebraic_simplify import AlgebraicSimplifyPass
from graph_optimizer.utils.graph_utils import create_node, create_const_node


tf.disable_v2_behavior()


class AlgebraicSimplifyPassTest(unittest.TestCase):
    def create_graph(self, nodes):
        """Helper to create a GraphDef from node list."""
        graph_def = tf.GraphDef()
        graph_def.node.extend(nodes)
        return graph_def

    def test_add_zero_left(self):
        x = create_node("Placeholder", name="x")
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        add = create_node("Add", name="add", inputs=["zero", "x"])
        graph = self.create_graph([x, zero, add])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        # add should be replaced by x
        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("add", names)

    def test_add_zero_right(self):
        x = create_node("Placeholder", name="x")
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        add = create_node("Add", name="add", inputs=["x", "zero"])
        graph = self.create_graph([x, zero, add])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("add", names)

    def test_sub_zero(self):
        x = create_node("Placeholder", name="x")
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        sub = create_node("Sub", name="sub", inputs=["x", "zero"])
        graph = self.create_graph([x, zero, sub])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("sub", names)

    def test_mul_one_left(self):
        x = create_node("Placeholder", name="x")
        one = create_const_node("one", value=1, dtype="float32", shape=[])
        mul = create_node("Mul", name="mul", inputs=["one", "x"])
        graph = self.create_graph([x, one, mul])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("mul", names)

    def test_mul_one_right(self):
        x = create_node("Placeholder", name="x")
        one = create_const_node("one", value=1, dtype="float32", shape=[])
        mul = create_node("Mul", name="mul", inputs=["x", "one"])
        graph = self.create_graph([x, one, mul])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("mul", names)

    def test_mul_zero_left(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        mul = create_node("Mul", name="mul", inputs=["zero", "x"])
        graph = self.create_graph([x, zero, mul])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["mul_zero"]
        )

        # result should be a zero const
        zeros = [
            n
            for n in optimizer.graph_def.node
            if n.op == "Const" and n.name == "mul_zero"
        ]
        self.assertEqual(len(zeros), 1)
        self.assertNotIn("mul", {n.name for n in optimizer.graph_def.node})

    def test_div_one(self):
        x = create_node("Placeholder", name="x")
        one = create_const_node("one", value=1, dtype="float32", shape=[])
        div = create_node("Div", name="div", inputs=["x", "one"])
        graph = self.create_graph([x, one, div])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("div", names)

    def test_neg_neg(self):
        x = create_node("Placeholder", name="x")
        neg1 = create_node("Neg", name="neg1", inputs=["x"])
        neg2 = create_node("Neg", name="neg2", inputs=["neg1"])
        graph = self.create_graph([x, neg1, neg2])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("neg1", names)
        self.assertNotIn("neg2", names)

    def test_logical_not_not(self):
        x = create_node("Placeholder", name="x")
        not1 = create_node("LogicalNot", name="not1", inputs=["x"])
        not2 = create_node("LogicalNot", name="not2", inputs=["not1"])
        graph = self.create_graph([x, not1, not2])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("not1", names)
        self.assertNotIn("not2", names)

    def test_equal_same(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        eq = create_node("Equal", name="eq", inputs=["x", "x"])
        graph = self.create_graph([x, eq])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["eq_bool"]
        )

        trues = [
            n
            for n in optimizer.graph_def.node
            if n.op == "Const" and n.name == "eq_bool"
        ]
        self.assertEqual(len(trues), 1)
        self.assertNotIn("eq", {n.name for n in optimizer.graph_def.node})

    def test_select_same_branch(self):
        cond = create_node("Placeholder", name="cond")
        x = create_node("Placeholder", name="x")
        sel = create_node("Select", name="sel", inputs=["cond", "x", "x"])
        graph = self.create_graph([cond, x, sel])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("sel", names)

    def test_no_simplify_add_nonzero(self):
        x = create_node("Placeholder", name="x")
        y = create_node("Placeholder", name="y")
        add = create_node("Add", name="add", inputs=["x", "y"])
        graph = self.create_graph([x, y, add])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        # Should remain unchanged
        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("add", names)

    def test_sub_same(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        sub = create_node("Sub", name="sub", inputs=["x", "x"])
        graph = self.create_graph([x, sub])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["sub_zero"]
        )

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("sub_zero", names)
        self.assertNotIn("sub", names)

    def test_add_neg(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        neg = create_node("Neg", name="neg", inputs=["x"])
        add = create_node("Add", name="add", inputs=["x", "neg"])
        graph = self.create_graph([x, neg, add])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["add_zero"]
        )

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("add_zero", names)
        self.assertNotIn("add", names)

    def test_mul_same(self):
        x = create_node("Placeholder", name="x")
        mul = create_node("Mul", name="mul", inputs=["x", "x"])
        graph = self.create_graph([x, mul])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        ops = {n.op for n in optimizer.graph_def.node}
        self.assertIn("Square", ops)
        self.assertNotIn("Mul", ops)

    def test_div_same(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        div = create_node("Div", name="div", inputs=["x", "x"])
        graph = self.create_graph([x, div])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["div_one"]
        )

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("div_one", names)
        self.assertNotIn("div", names)

    def test_pow_one(self):
        x = create_node("Placeholder", name="x")
        one = create_const_node("one", value=1, dtype="float32", shape=[])
        pow_node = create_node("Pow", name="pow", inputs=["x", "one"])
        graph = self.create_graph([x, one, pow_node])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("pow", names)

    def test_pow_two(self):
        x = create_node("Placeholder", name="x")
        two = create_const_node("two", value=2, dtype="float32", shape=[])
        pow_node = create_node("Pow", name="pow", inputs=["x", "two"])
        graph = self.create_graph([x, two, pow_node])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        ops = {n.op for n in optimizer.graph_def.node}
        self.assertIn("Square", ops)
        self.assertNotIn("Pow", ops)

    def test_logical_and_false(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        false_node = create_const_node("false", value=False, dtype="bool", shape=[])
        and_node = create_node("LogicalAnd", name="and_node", inputs=["x", "false"])
        graph = self.create_graph([x, false_node, and_node])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["and_node_bool"]
        )

        consts = [n for n in optimizer.graph_def.node if n.op == "Const"]
        # Should have and_node_bool (False)
        has_false = any(
            n.name == "and_node_bool" and n.attr["value"].tensor.bool_val[0] == False
            for n in consts
        )
        self.assertTrue(has_false)

    def test_logical_or_true(self):
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        true_node = create_const_node("true", value=True, dtype="bool", shape=[])
        or_node = create_node("LogicalOr", name="or_node", inputs=["x", "true"])
        graph = self.create_graph([x, true_node, or_node])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["or_node_bool"]
        )

        consts = [n for n in optimizer.graph_def.node if n.op == "Const"]
        has_true = any(
            n.name == "or_node_bool" and n.attr["value"].tensor.bool_val[0] == True
            for n in consts
        )
        self.assertTrue(has_true)

    def test_add_zero_broadcast_positive(self):
        # x is [2, 2], zero is scalar []. Add(x, 0) -> x [2, 2]. Safe.
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([2, 2]).as_proto())
        zero = create_const_node("zero", value=0, dtype="float32", shape=[])
        add = create_node("Add", name="add", inputs=["x", "zero"])
        graph = self.create_graph([x, zero, add])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("x", names)
        self.assertNotIn("add", names)

    def test_add_zero_broadcast_negative(self):
        # x is scalar [], zero is [2, 2]. Add(x, zero) is [2, 2]. 
        # Simplifying to x would change shape to []. NOT SAFE.
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        zero = create_const_node("zero", value=[[0, 0], [0, 0]], dtype="float32", shape=[2, 2])
        add = create_node("Add", name="add", inputs=["x", "zero"])
        graph = self.create_graph([x, zero, add])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(optimizer, auto_cleanup=True)

        # Should NOT simplify
        names = {n.name for n in optimizer.graph_def.node}
        self.assertIn("add", names)

    def test_mul_zero_broadcast(self):
        # x is [2, 1], zero is [1, 2]. Mul(x, zero) is [2, 2].
        # Even if one is zero, we must create a [2, 2] zero.
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([2, 1]).as_proto())
        zero = create_const_node("zero", value=[[0, 0]], dtype="float32", shape=[1, 2])
        mul = create_node("Mul", name="mul", inputs=["x", "zero"])
        graph = self.create_graph([x, zero, mul])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["mul_zero"]
        )

        # Should simplify to a [2, 2] zero constant
        folded = [n for n in optimizer.graph_def.node if n.name == "mul_zero"]
        self.assertEqual(len(folded), 1)
        shape = [d.size for d in folded[0].attr["value"].tensor.tensor_shape.dim]
        self.assertEqual(shape, [2, 2])

    def test_logical_and_broadcast_negative(self):
        # x is [], False is [2]. And(x, False) is [2].
        # Simplifying to scalar False is NOT safe.
        x = create_node("Placeholder", name="x")
        x.attr["shape"].shape.CopyFrom(tf.TensorShape([]).as_proto())
        false_node = create_const_node("false", value=[False, False], dtype="bool", shape=[2])
        and_node = create_node("LogicalAnd", name="and_node", inputs=["x", "false"])
        graph = self.create_graph([x, false_node, and_node])

        optimizer = GraphOptimizer(graph)
        optimizer.load_state(graph)
        simplify_pass = AlgebraicSimplifyPass()
        simplify_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["and_node_bool"]
        )

        # Should simplify to a [2] False constant
        folded = [n for n in optimizer.graph_def.node if n.name == "and_node_bool"]
        self.assertEqual(len(folded), 1)
        shape = [d.size for d in folded[0].attr["value"].tensor.tensor_shape.dim]
        self.assertEqual(shape, [2])


if __name__ == "__main__":
    unittest.main()
