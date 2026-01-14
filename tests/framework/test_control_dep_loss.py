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


if __name__ == "__main__":
    unittest.main()
