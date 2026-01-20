"""
Core Engine Tests - 核心引擎基础功能测试
=========================================

测试内容：
1. test_node_lookup          - 节点查找和追踪
2. test_commutative_matching - 交换律操作符匹配（Add(x,c) == Add(c,x)）
3. test_variadic_matching    - 可变参数输入匹配（ConcatV2 多输入）
4. test_control_dependency_handling - 控制依赖处理（^node 跳过匹配但保留）

依赖：
- GraphOptimizer : 核心优化器
- Op, Any, CommutativeOp, Variadic : 模式匹配原语
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import (
    GraphOptimizer,
    Op,
    Any,
    CommutativeOp,
    Variadic,
)
from graph_optimizer.utils import create_node

tf.disable_v2_behavior()


class TestCoreEngine(unittest.TestCase):
    """GraphOptimizer 核心引擎基础功能测试套件。"""
    def setUp(self):
        tf.reset_default_graph()

    def test_node_lookup(self):
        """Test fundamental node tracking."""
        with tf.Graph().as_default():
            tf.placeholder(tf.float32, name="A")
            tf.constant(1.0, name="B")
            graph_def = tf.get_default_graph().as_graph_def()

        optimizer = GraphOptimizer(graph_def)
        self.assertIn("A", optimizer.nodes)
        self.assertIn("B", optimizer.nodes)

    def test_commutative_matching(self):
        """Test matching logic for commutative operators."""
        pattern = CommutativeOp(
            "AddV2", Op("Const", alias="c"), Any(alias="x"), alias="root"
        )

        with tf.Graph().as_default():
            c = tf.constant(1.0, name="c")
            x = tf.placeholder(tf.float32, name="x")
            tf.add(x, c, name="add1")
            graph_def = tf.get_default_graph().as_graph_def()

        optimizer = GraphOptimizer(graph_def)
        match = pattern.match(optimizer.nodes["add1"], optimizer)
        self.assertIsNotNone(match)

    def test_variadic_matching(self):
        """Test matching variable number of inputs."""
        graph_def = tf.GraphDef()
        for i in range(1, 4):
            graph_def.node.append(create_node("Const", f"c{i}"))
        graph_def.node.append(create_node("Const", "axis"))

        concat = create_node("ConcatV2", "concat", inputs=["c1", "c2", "c3", "axis"])
        graph_def.node.append(concat)
        optimizer = GraphOptimizer(graph_def)

        pattern = Op("ConcatV2", Variadic(Op("Const")), Op("Const", alias="axis"))
        match = pattern.match(concat, optimizer)
        self.assertIsNotNone(match)

    def test_control_dependency_handling(self):
        """Test that ^control dependencies are ignored during matching but preserved during rewrites."""
        graph_def = tf.GraphDef()
        a = create_node("Placeholder", "a")
        c = create_node("Placeholder", "c")
        b = create_node("Identity", "b", inputs=["a", "^c"])
        graph_def.node.extend([a, c, b])
        optimizer = GraphOptimizer(graph_def)

        # Matching should skip ^c
        pattern = Op("Identity", Op("Placeholder"))
        self.assertIsNotNone(pattern.match(b, optimizer))

        # Preservation (tested via optimize simulation)
        def rewriter(match, opt):
            root = match.matched_nodes["root"]
            return [create_node("Identity", root.name, inputs=["a"])]

        optimizer.add_transformation(Op("Identity", alias="root"), rewriter)
        opt_graph = optimizer.optimize(auto_cleanup=False)
        new_b = next(n for n in opt_graph.node if n.name == "b")
        self.assertIn("^c", new_b.input)


if __name__ == "__main__":
    unittest.main()
