"""
ConstantFoldPass Tests
======================

Tests for scalar constant folding optimization pass.
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.transforms.scalar.constant_fold import ConstantFoldPass
from graph_optimizer.utils.graph_utils import create_node, create_const_node


tf.disable_v2_behavior()


class ConstantFoldPassTest(unittest.TestCase):
    def create_graph(self, nodes):
        """Helper to create a GraphDef from node list."""
        graph_def = tf.GraphDef()
        graph_def.node.extend(nodes)
        return graph_def

    def test_add_fold(self):
        const_a = create_const_node("a", value=[2, 3], dtype="int32", shape=[2])
        const_b = create_const_node("b", value=[4, 5], dtype="int32", shape=[2])
        add = create_node("Add", name="add", inputs=["a", "b"])
        graph = self.create_graph([const_a, const_b, add])

        optimizer = GraphOptimizer(graph)
        fold_pass = ConstantFoldPass()
        fold_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["add_folded"]
        )

        # Check the optimized graph
        const_nodes = [n for n in optimizer.graph_def.node if n.op == "Const"]
        self.assertGreaterEqual(len(const_nodes), 1)  # at least folded
        folded = [n for n in const_nodes if n.name == "add_folded"]
        self.assertEqual(len(folded), 1)
        # Check value - Make sure numpy conversion is handled properly
        from tensorflow.python.framework import tensor_util

        val = folded[0].attr.get("value", None)
        self.assertIsNotNone(val)
        arr = tensor_util.MakeNdarray(val.tensor).tolist()
        self.assertEqual(
            arr, [[6, 8]] if len(val.tensor.tensor_shape.dim) == 2 else [6, 8]
        )

    def test_mul_fold(self):
        const_a = create_const_node("a", value=[2, 3], dtype="int32", shape=[2])
        const_b = create_const_node("b", value=[4, 5], dtype="int32", shape=[2])
        mul = create_node("Mul", name="mul", inputs=["a", "b"])
        graph = self.create_graph([const_a, const_b, mul])

        optimizer = GraphOptimizer(graph)
        fold_pass = ConstantFoldPass()
        fold_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["mul_folded"]
        )

        folded = [n for n in optimizer.graph_def.node if n.name == "mul_folded"]
        self.assertEqual(len(folded), 1)
        from tensorflow.python.framework import tensor_util

        val = folded[0].attr.get("value", None)
        self.assertIsNotNone(val)
        arr = tensor_util.MakeNdarray(val.tensor).tolist()
        self.assertEqual(
            arr, [[8, 15]] if len(val.tensor.tensor_shape.dim) == 2 else [8, 15]
        )

    def test_unsupported_op_not_folded(self):
        const_a = create_const_node("a", value=[1], dtype="int32", shape=[1])
        unknown = create_node("UnknownOp", name="unk", inputs=["a"])
        graph = self.create_graph([const_a, unknown])

        optimizer = GraphOptimizer(graph)
        fold_pass = ConstantFoldPass()
        fold_pass.transform(optimizer, auto_cleanup=True)

        # UnknownOp should remain
        unk_nodes = [n for n in optimizer.graph_def.node if n.name == "unk"]
        self.assertEqual(len(unk_nodes), 1)

    def test_partial_non_const_not_folded(self):
        const_a = create_const_node("a", value=[2], dtype="int32", shape=[1])
        non_const = create_node("Placeholder", name="ph")
        add = create_node("Add", name="add", inputs=["a", "ph"])
        graph = self.create_graph([const_a, non_const, add])

        optimizer = GraphOptimizer(graph)
        fold_pass = ConstantFoldPass()
        fold_pass.transform(optimizer, auto_cleanup=True)

        # Add should remain because one input is not Const
        add_nodes = [n for n in optimizer.graph_def.node if n.name == "add"]
        self.assertEqual(len(add_nodes), 1)

    def test_dtype_promotion(self):
        const_a = create_const_node("a", value=[2], dtype="int32", shape=[1])
        const_b = create_const_node("b", value=[4.5], dtype="float32", shape=[1])
        add = create_node("Add", name="add", inputs=["a", "b"])
        graph = self.create_graph([const_a, const_b, add])

        optimizer = GraphOptimizer(graph)
        fold_pass = ConstantFoldPass()
        fold_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["add_folded"]
        )

        folded = [n for n in optimizer.graph_def.node if n.name == "add_folded"]
        self.assertEqual(len(folded), 1)
        # Result should be float32 or float64 (promoted)
        dtype = folded[0].attr["dtype"].type
        self.assertIn(tf.as_dtype(dtype).name, ["float32", "float64"])

    def test_div_zero_fold(self):
        const_a = create_const_node("a", value=[2.0], dtype="float32", shape=[1])
        const_b = create_const_node("b", value=[0.0], dtype="float32", shape=[1])
        div = create_node("Div", name="div", inputs=["a", "b"])
        graph = self.create_graph([const_a, const_b, div])

        optimizer = GraphOptimizer(graph)
        fold_pass = ConstantFoldPass()
        # Now it should fold to Inf instead of crashing or skipping
        fold_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["div_folded"]
        )

        folded = [n for n in optimizer.graph_def.node if n.name == "div_folded"]
        self.assertEqual(len(folded), 1)
        import numpy as np
        from tensorflow.python.framework import tensor_util

        arr = tensor_util.MakeNdarray(folded[0].attr["value"].tensor)
        self.assertTrue(np.isinf(arr[0]))

    def test_sqrt_negative(self):
        const_a = create_const_node("a", value=[-1.0], dtype="float32", shape=[1])
        sqrt = create_node("Sqrt", name="sqrt", inputs=["a"])
        graph = self.create_graph([const_a, sqrt])

        optimizer = GraphOptimizer(graph)
        fold_pass = ConstantFoldPass()
        # Should fold to NaN
        fold_pass.transform(
            optimizer, auto_cleanup=True, protected_nodes=["sqrt_folded"]
        )

        folded = [n for n in optimizer.graph_def.node if n.name == "sqrt_folded"]
        self.assertEqual(len(folded), 1)
        import numpy as np
        from tensorflow.python.framework import tensor_util

        arr = tensor_util.MakeNdarray(folded[0].attr["value"].tensor)
        self.assertTrue(np.isnan(arr[0]))


if __name__ == "__main__":
    unittest.main()
