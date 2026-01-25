"""
Control Dependency Loss Tests - 控制依赖丢失测试
=================================================

测试内容：
1. test_internal_control_dep_preservation - 内部节点控制依赖保留

场景：
    原图: A -> Identity(B, ^C)
    模式: Op("Identity", Op("Identity", alias="inner"), alias="root")
    
    当 rewriter 替换 root 时，inner 上的 ^C 必须被保留到新节点。
    
这是回归测试，确保 framework 在 rewrite 时正确收集和应用控制依赖。
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import (
    GraphOptimizer,
    Op,
    PatternRewritePass,
)
from graph_optimizer.utils import create_node

tf.disable_v2_behavior()


class TestControlDepLoss(unittest.TestCase):
    """控制依赖保留测试套件。"""
    def test_internal_control_dep_preservation(self):
        """
        Test that control dependencies on internal nodes of a match are preserved.
        Graph: A -> Identity(B, ^C)
        Pattern: Op("Identity", Op("Identity", alias="inner"), alias="root")
        If we replace root, we must keep ^C even if it was on 'inner'.
        """
        graph_def = tf.GraphDef()
        c = create_node("NoOp", "C")
        a = create_node("Const", "A")
        # inner has a control dep on C
        inner = create_node("Identity", "inner", inputs=["A", "^C"])
        # root points to inner
        root = create_node("Identity", "root", inputs=["inner"])

        graph_def.node.extend([c, a, inner, root])

        optimizer = GraphOptimizer(graph_def)

        # Define a rewriter that replaces root with a NoOp
        def rewriter(match, opt):
            # The framework SHOULD collect ^C from 'inner' and apply it to the result
            return [create_node("NoOp", match.matched_nodes["root"].name)]

        pattern = Op("Identity", Op("Identity", alias="inner"), alias="root")
        optimizer.add_transformation(pattern, rewriter)

        # Optimize
        optimized = optimizer.optimize(auto_cleanup=False)

        root_optimized = next(n for n in optimized.node if n.name == "root")
        self.assertIn(
            "^C",
            root_optimized.input,
            "Control dependency from internal node was lost!",
        )

    def test_control_dep_preservation_with_mapping(self):
        """
        Test that control dependencies are preserved when using node_mapping.
        Scenario: trigger -> add(x, zero) -> result
        Rewrite: add -> x (bypass), result should now depend on trigger.
        """
        from graph_optimizer.core import RewriteResult

        graph_def = tf.GraphDef()
        x = create_node("Const", "x")
        zero = create_node("Const", "zero")
        trigger = create_node("NoOp", "trigger")
        add = create_node("Add", "add", inputs=["x", "zero", "^trigger"])
        result = create_node("Identity", "result", inputs=["add"])

        graph_def.node.extend([x, zero, trigger, add, result])

        optimizer = GraphOptimizer(graph_def)

        # Define a rewriter that remaps add -> x and adds a side node
        def rewriter(match, opt):
            side = create_node("NoOp", "side")
            return RewriteResult(
                new_nodes=[side],
                replaced_nodes=[],
                node_mapping={"add": "x"}
            )

        pattern = Op("Add", Op("Const"), Op("Const"))
        optimizer.add_transformation(pattern, rewriter)

        # Optimize
        optimized = optimizer.optimize(auto_cleanup=True, protected_nodes=["result"])
        node_map = {n.name: n for n in optimized.node}
        
        result_node = node_map["result"]
        self.assertIn("x", result_node.input)
        self.assertIn("^trigger", result_node.input, 
                      "Control dependency was lost during node remapping!")


if __name__ == "__main__":
    unittest.main()
