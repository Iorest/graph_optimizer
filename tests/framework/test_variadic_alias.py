"""
Variadic Alias Tests - Variadic 模式别名测试
=============================================

测试内容：
1. test_variadic_alias_collection - Variadic 匹配结果收集到别名

Variadic 模式用于匹配可变数量的输入：
    模式: Op("ConcatV2", Variadic(Op("Const"), alias="my_inputs"), alias="root")
    节点: ConcatV2(c1, c2, c3)
    
    匹配后: match.matched_nodes["my_inputs"] = [c1, c2, c3]

这对于处理 ConcatV2、Pack 等可变输入数量的操作非常重要。
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import (
    GraphOptimizer,
    Op,
    Variadic,
)
from graph_optimizer.utils import create_node

tf.disable_v2_behavior()


class TestVariadicAlias(unittest.TestCase):
    """Variadic 别名收集测试套件。"""
    def test_variadic_alias_collection(self):
        """Test that Variadic(..., alias="name") collects a list of nodes."""
        graph_def = tf.GraphDef()
        graph_def.node.append(create_node("Const", "c1"))
        graph_def.node.append(create_node("Const", "c2"))
        graph_def.node.append(create_node("Const", "c3"))
        graph_def.node.append(
            create_node("ConcatV2", "root", inputs=["c1", "c2", "c3"])
        )

        optimizer = GraphOptimizer(graph_def)

        captured_inputs = None

        def rewriter(match, opt):
            nonlocal captured_inputs
            captured_inputs = match.matched_nodes["my_inputs"]
            return [match.matched_nodes["root"]]

        # Variadic matches everything before the last input (which isn't really a thing here,
        # but the pattern will match c1, c2 as variadic if followed by something)
        # However, ConcatV2 has axis as last input. Let's simplify.

        # Match Any node with some variadic inputs
        pattern = Op("ConcatV2", Variadic(Op("Const"), alias="my_inputs"), alias="root")
        optimizer.add_transformation(pattern, rewriter)

        optimizer.optimize(auto_cleanup=False)

        self.assertIsNotNone(captured_inputs)
        self.assertEqual(len(captured_inputs), 3)
        self.assertEqual(captured_inputs[0].name, "c1")
        self.assertEqual(captured_inputs[1].name, "c2")
        self.assertEqual(captured_inputs[2].name, "c3")


if __name__ == "__main__":
    unittest.main()
