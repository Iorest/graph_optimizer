import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.utils import create_node, export_to_dot


class TestVisualize(unittest.TestCase):
    def test_export_to_dot(self):
        graph_def = tf.GraphDef()
        graph_def.node.append(create_node("Const", "a"))
        graph_def.node.append(create_node("Const", "b"))
        graph_def.node.append(create_node("Add", "c", inputs=["a", "b"]))

        dot = export_to_dot(graph_def, highlight_nodes={"c"})
        self.assertIn("digraph G {", dot)
        self.assertIn('"a" [label="a\\n(Const)"', dot)
        self.assertIn('"c" [label="c\\n(Add)", fillcolor="lightblue"', dot)
        self.assertIn('"a" -> "c"', dot)
        self.assertIn('"b" -> "c"', dot)


if __name__ == "__main__":
    unittest.main()
