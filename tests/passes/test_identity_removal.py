import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.utils import create_node
from graph_optimizer.optimizers.identity_removal import IdentityRemovalPass

tf.disable_v2_behavior()


class TestIdentityRemoval(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_identity_removal(self):
        """Test removal of dual Identity nodes."""
        graph_def = tf.GraphDef()
        x = create_node("Placeholder", "x")
        id1 = create_node("Identity", "id1", inputs=["x"])
        id2 = create_node("Identity", "id2", inputs=["id1"])
        graph_def.node.extend([x, id1, id2])

        optimizer = GraphOptimizer(graph_def)
        # Protect 'id2' to allow id1 to be pruned while id2 remains
        optimized = IdentityRemovalPass().transform(optimizer, protected_nodes=["id2"])

        node_names = [n.name for n in optimized.node]
        self.assertIn("x", node_names)
        self.assertIn("id2", node_names)
        self.assertNotIn("id1", node_names)

        new_id2 = next(n for n in optimized.node if n.name == "id2")
        self.assertEqual(new_id2.input, ["x"])


if __name__ == "__main__":
    unittest.main()
