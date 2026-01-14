import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.utils import create_node

tf.disable_v2_behavior()


class TestPruning(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_fundamental_pruning(self):
        """Test fundamental graph pruning."""
        with tf.Graph().as_default():
            tf.placeholder(tf.float32, name="A")
            tf.constant(1.0, name="B")
            add = tf.add(
                tf.placeholder(tf.float32, name="A_"),
                tf.constant(1.0, name="B_"),
                name="Add",
            )
            tf.identity(add, name="Y")
            tf.constant(2.0, name="Z")
            graph_def = tf.get_default_graph().as_graph_def()

        optimizer = GraphOptimizer(graph_def)
        # Test pruning
        pruned = optimizer.prune(["Y"])
        node_names = [n.name for n in pruned.node]
        self.assertIn("Y", node_names)
        self.assertIn("Add", node_names)
        self.assertNotIn("Z", node_names)

    def test_reference_counts(self):
        graph_def = tf.GraphDef()
        graph_def.node.append(create_node("Const", "a"))
        graph_def.node.append(create_node("Add", "b", inputs=["a", "a"]))
        graph_def.node.append(create_node("Mul", "c", inputs=["a", "b"]))

        optimizer = GraphOptimizer(graph_def)
        refs = optimizer._compute_reference_counts(graph_def)
        self.assertEqual(refs["a"], 3)
        self.assertEqual(refs["b"], 1)

    def test_preserve_placeholders(self):
        graph_def = tf.GraphDef()
        graph_def.node.append(create_node("Placeholder", "unused"))
        optimizer = GraphOptimizer(graph_def)
        pruned = optimizer._final_prune(graph_def, "test")
        self.assertIn("unused", [n.name for n in pruned.node])


if __name__ == "__main__":
    unittest.main()
