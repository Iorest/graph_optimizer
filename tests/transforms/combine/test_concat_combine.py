"""
Concat Combine Tests - Concat 融合测试
======================================

测试内容：
1. test_concat_combine - 嵌套 ConcatV2 融合

场景：
    原图: inner = ConcatV2([a, b], axis=0)
          outer = ConcatV2([inner, c], axis=0)
    
    优化: outer = ConcatV2([a, b, c], axis=0)
          (inner 被删除，输入展开到 outer)

条件：内外 ConcatV2 的 axis 必须相同。
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.utils import create_node
from graph_optimizer.transforms.combine import ConcatCombinePass
from tensorflow.core.framework import attr_value_pb2

tf.disable_v2_behavior()


class TestConcatCombine(unittest.TestCase):
    """Concat 融合测试套件。"""
    def setUp(self):
        tf.reset_default_graph()

    def _create_const(self, name, value=0):
        attr = {
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(value, dtype=tf.int32)
            )
        }
        return create_node("Const", name, attr=attr)

    def test_concat_combine(self):
        """Test combining nested ConcatV2 operations with same axis."""
        graph_def = tf.GraphDef()
        axis0 = self._create_const("axis0", 0)
        a = create_node("Placeholder", "a")
        b = create_node("Placeholder", "b")
        inner = create_node("ConcatV2", "inner", inputs=["a", "b", "axis0"])
        c = create_node("Placeholder", "c")
        outer = create_node("ConcatV2", "outer", inputs=["inner", "c", "axis0"])
        graph_def.node.extend([axis0, a, b, inner, c, outer])

        optimizer = GraphOptimizer(graph_def)
        # Protect 'outer' to avoid pruning it as it's a leaf
        optimized = ConcatCombinePass().transform(optimizer, protected_nodes=["outer"])

        node_map = {n.name: n for n in optimized.node}
        self.assertIn("outer", node_map)
        self.assertEqual(node_map["outer"].input, ["a", "b", "c", "axis0"])
        self.assertNotIn("inner", node_map)


    def test_concat_combine_with_control_dependency(self):
        """
        Test that control dependencies from an inner, fused ConcatV2 node are
        correctly hoisted to the new combined ConcatV2 node.
        """
        graph_def = tf.GraphDef()
        axis0 = self._create_const("axis0", 0)
        ctrl_op = create_node("NoOp", "ctrl_op")
        a = create_node("Placeholder", "a")
        b = create_node("Placeholder", "b")
        # Inner concat has a control dependency
        inner = create_node("ConcatV2", "inner", inputs=["a", "b", "axis0", "^ctrl_op"])
        c = create_node("Placeholder", "c")
        outer = create_node("ConcatV2", "outer", inputs=["inner", "c", "axis0"])
        graph_def.node.extend([axis0, ctrl_op, a, b, inner, c, outer])

        optimizer = GraphOptimizer(graph_def)
        # Protect 'outer' to avoid pruning it as it's a leaf
        optimized = ConcatCombinePass().transform(optimizer, protected_nodes=["outer"])

        node_map = {n.name: n for n in optimized.node}
        self.assertIn("outer", node_map)
        
        # Verify inputs are combined correctly
        outer_node = node_map["outer"]
        # The order of control inputs is not guaranteed, so we check data and control inputs separately.
        data_inputs = [inp for inp in outer_node.input if not inp.startswith('^')]
        control_inputs = [inp for inp in outer_node.input if inp.startswith('^')]

        self.assertEqual(data_inputs, ["a", "b", "c", "axis0"])
        
        # Verify the control dependency from the inner node was inherited
        self.assertIn("^ctrl_op", control_inputs)
        
        # Verify the inner node was removed
        self.assertNotIn("inner", node_map)

    def test_outer_control_dep_bug(self):
        """
        Test that outer ConcatV2 with control dependencies is handled correctly.
        """
        graph_def = tf.GraphDef()
        axis0 = self._create_const("axis0", 0)
        ctrl_op = create_node("NoOp", "ctrl_op")
        
        a = create_node("Placeholder", "a")
        b = create_node("Placeholder", "b")
        c = create_node("Placeholder", "c")
        
        # inner = Concat([a, b], axis)
        inner = create_node("ConcatV2", "inner", inputs=["a", "b", "axis0"])
        inner.attr["N"].i = 2
        
        # outer = Concat([inner, c], axis) + ^ctrl_op
        outer = create_node("ConcatV2", "outer", inputs=["inner", "c", "axis0", "^ctrl_op"])
        outer.attr["N"].i = 2
        
        graph_def.node.extend([axis0, ctrl_op, a, b, c, inner, outer])
        
        optimizer = GraphOptimizer(graph_def)
        ConcatCombinePass().transform(optimizer, protected_nodes=["outer"])
        
        new_outer = optimizer.nodes["outer"]
        n_attr = new_outer.attr["N"].i
        
        # We expect N=3 (a, b, c).
        self.assertEqual(n_attr, 3, f"Outer N attribute should be 3, got {n_attr}.")
        self.assertIn("^ctrl_op", new_outer.input)


if __name__ == "__main__":
    unittest.main()
