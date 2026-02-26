"""Regression tests for AlgebraicSimplifyPass bug fixes."""

import tensorflow.compat.v1 as tf
import pytest
from graph_optimizer.core.tensorflow import TFGraphOptimizer
from graph_optimizer.transforms.tensorflow.scalar.algebraic_simplify import (
    AlgebraicSimplifyPass,
)


def test_read_variable_op_safety():
    """Verify that ReadVariableOp is NOT simplified/bypassed by CSE or AlgebraicSimplify."""
    with tf.Graph().as_default() as g:
        # Manually create ReadVariableOp nodes to avoid API issues
        try:
            v = tf.get_variable("v", shape=[1], dtype=tf.float32)
        except:
            v = tf.placeholder(tf.float32, shape=[1], name="v")

        # We'll use Identity strings for names
        r1 = tf.identity(v, name="ReadVariableOp_1")
        r2 = tf.identity(v, name="ReadVariableOp_2")
        add = tf.add(r1, r2, name="add")
        # Ensure we have a protected output
        y = tf.identity(add, name="out")
        graph_def = g.as_graph_def()

    # Manually change op type to ReadVariableOp for the test
    for node in graph_def.node:
        if "ReadVariableOp" in node.name:
            node.op = "ReadVariableOp"

    optimizer = TFGraphOptimizer(graph_def)
    pass_ = AlgebraicSimplifyPass()

    # Apply pass, protecting the output
    pass_.transform(optimizer, protected_nodes={"out"})

    # Check ReadVariableOp nodes
    read_nodes = [n for n in optimizer.graph_def.node if n.op == "ReadVariableOp"]
    assert len(read_nodes) == 2, (
        f"ReadVariableOps should not be merged or removed, found {len(read_nodes)}"
    )


def test_numeric_const_dtype_robustness():
    """Verify _numeric_const handles dtypes correctly using get_node_attr."""
    # Create an Add(x, x_neg) -> 0.0 with specific dtype
    # Use DT_DOUBLE and explicit shape
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float64, shape=[2, 2], name="x")
        x_neg = tf.negative(x, name="x_neg")
        add = tf.add(x, x_neg, name="add")
        # Add output to prevent pruning
        out = tf.identity(add, name="out")
        graph_def = g.as_graph_def()

    optimizer = TFGraphOptimizer(graph_def)
    pass_ = AlgebraicSimplifyPass()

    # Should simplify Add(x, Neg(x)) to Const(0.0)
    # We use transform to pass protected_nodes
    pass_.transform(optimizer, protected_nodes={"out"})

    # Check the created zero constant
    zero_nodes = [n for n in optimizer.graph_def.node if n.op == "Const"]
    assert len(zero_nodes) >= 1, "Should have created a Const node for zero"

    # One of the constants should be our zero
    zero_node = None
    for n in zero_nodes:
        if "zero" in n.name or "add" in n.name:
            zero_node = n
            break
    assert zero_node is not None, "Zero constant node not found"

    # Verify dtype
    dtype_attr = zero_node.attr.get("dtype")
    assert dtype_attr.type == tf.float64.as_datatype_enum, (
        f"Folded zero should maintain float64 dtype, got {dtype_attr.type}"
    )

    # Verify shape
    shape_attr = zero_node.attr.get("value").tensor.tensor_shape
    dims = [d.size for d in shape_attr.dim]
    assert dims == [2, 2], f"Expected shape [2, 2], got {dims}"
