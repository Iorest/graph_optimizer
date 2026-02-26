"""Tests for ConstantFoldPass extra features (Bitwise dtypes and Cast expansion)."""

import tensorflow.compat.v1 as tf
import pytest
import numpy as np
from graph_optimizer.core.tensorflow import TFGraphOptimizer
from graph_optimizer.transforms.tensorflow.scalar.constant_fold import ConstantFoldPass


def test_bitwise_fold_dtype_preservation():
    """Verify bitwise folding preserves integer dtypes and doesn't force int64."""
    with tf.Graph().as_default() as g:
        # Use int32
        c1 = tf.constant([1, 2, 3], dtype=tf.int32, name="c1")
        c2 = tf.constant([4, 5, 6], dtype=tf.int32, name="c2")
        and_ = tf.bitwise.bitwise_and(c1, c2, name="and")
        out = tf.identity(and_, name="out")
        graph_def = g.as_graph_def()

    optimizer = TFGraphOptimizer(graph_def)
    pass_ = ConstantFoldPass()
    pass_.transform(optimizer, protected_nodes={"out"})

    # Check result
    folded_nodes = [
        n for n in optimizer.graph_def.node if n.op == "Const" and "folded" in n.name
    ]
    assert len(folded_nodes) == 1
    node = folded_nodes[0]

    # Verify dtype is int32, not int64 (previous bug forced int64)
    dtype_attr = node.attr.get("dtype")
    assert dtype_attr.type == tf.int32.as_datatype_enum, (
        f"Expected int32, got {dtype_attr.type}"
    )


def test_cast_fold_expansion():
    """Verify expanded support for Cast folding (uint8, int16, int8)."""
    with tf.Graph().as_default() as g:
        # Cast float32 to uint8
        c = tf.constant([1.0, 2.5, 255.0], dtype=tf.float32, name="c")
        cast = tf.cast(c, tf.uint8, name="cast")
        out = tf.identity(cast, name="out")
        graph_def = g.as_graph_def()

    optimizer = TFGraphOptimizer(graph_def)
    pass_ = ConstantFoldPass()
    pass_.transform(optimizer, protected_nodes={"out"})

    # Check result
    folded_nodes = [
        n for n in optimizer.graph_def.node if n.op == "Const" and "folded" in n.name
    ]
    assert len(folded_nodes) == 1
    node = folded_nodes[0]

    # Verify dtype is uint8
    dtype_attr = node.attr.get("dtype")
    assert dtype_attr.type == tf.uint8.as_datatype_enum, (
        f"Expected uint8, got {dtype_attr.type}"
    )

    # Verify values
    from tensorflow.python.framework import tensor_util

    val = tensor_util.MakeNdarray(node.attr.get("value").tensor)
    assert val.dtype == np.uint8
    np.testing.assert_array_equal(val, [1, 2, 255])
