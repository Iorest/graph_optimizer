"""
Indexed Matching Tests - 索引化模式匹配测试
============================================

测试内容：
1. test_indexing_logic        - 模式按 op_type 正确索引
2. test_indexed_optimization  - 优化时只检查相关模式

优化原理：
- 模式按 op_type 分组存储到 pattern_index
- 优化时，只检查与当前节点 op_type 匹配的模式
- Op(None) 是通配符，存储到 wildcard_patterns

性能影响：
- 避免 O(N*M) 的全模式遍历
- 实现 O(N*K) 其中 K << M（K 是同类型模式数）
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import (
    GraphOptimizer,
    Op,
)

tf.disable_v2_behavior()


class TestIndexedMatching(unittest.TestCase):
    """索引化模式匹配测试套件。"""
    def test_indexing_logic(self):
        """Test that patterns are correctly indexed by op_type."""
        optimizer = GraphOptimizer(tf.GraphDef())

        # Specific op pattern
        p1 = Op("Add")

        def r1(m, o):
            return []

        optimizer.add_transformation(p1, r1)
        self.assertIn("Add", optimizer.pattern_index)
        self.assertEqual(len(optimizer.pattern_index["Add"]), 1)
        self.assertEqual(len(optimizer.wildcard_patterns), 0)

        # Null op pattern (wildcard)
        p2 = Op(None)

        def r2(m, o):
            return []

        optimizer.add_transformation(p2, r2)
        self.assertEqual(len(optimizer.wildcard_patterns), 1)

    def test_indexed_optimization(self):
        """Test that only relevant patterns are checked during optimization."""
        graph_def = tf.GraphDef()
        graph_def.node.append(tf.NodeDef(name="n1", op="Add"))

        optimizer = GraphOptimizer(graph_def)

        # Pattern that should NOT match because it's for 'Mul'
        matched_wrong = False

        def r_wrong(m, o):
            nonlocal matched_wrong
            matched_wrong = True
            return None

        optimizer.add_transformation(Op("Mul"), r_wrong)
        optimizer.optimize(auto_cleanup=False)
        self.assertFalse(matched_wrong)


if __name__ == "__main__":
    unittest.main()
