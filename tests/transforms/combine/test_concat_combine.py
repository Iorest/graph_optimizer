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


if __name__ == "__main__":
    unittest.main()
