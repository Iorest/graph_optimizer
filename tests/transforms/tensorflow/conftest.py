"""Shared pytest fixtures for TensorFlow transform tests."""

import pytest
import tensorflow.compat.v1 as tf
from graph_optimizer.core.tensorflow import TFGraphOptimizer

tf.disable_v2_behavior()


def make_graph(*nodes):
    """Build a TFGraphOptimizer from a list of NodeDef objects."""
    graph_def = tf.GraphDef()
    graph_def.node.extend(nodes)
    return TFGraphOptimizer(graph_def)
