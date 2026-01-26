"""
Optimized Algebraic Simplification Pass
=======================================

This module implements a single, efficient pass for algebraic simplification.
It replaces the legacy implementation that used a generic `Any()` pattern, which
was inefficient.

This pass inherits from `BasePass` and, during its execution, registers multiple
specialized `OpPattern`s with the graph optimizer. By using specific patterns,
it leverages the framework's O(1) op-type index for fast matching, while still
presenting a single, unified 'algebraic_simplify' pass to the user.

Key Improvements:
-----------------
- **Performance**: Uses specific `OpPattern`s for O(1) matching.
- **Maintainability**: Consolidates all algebraic rules in one place but uses
  separate patterns and rewrite methods for clarity.
- **Safety**: Includes shape-preservation checks to prevent unsafe optimizations
  where broadcasting could change tensor shapes.
"""

from __future__ import annotations
import numpy as np
import tensorflow.compat.v1 as tf

from graph_optimizer.core import (
    Op,
    CommutativeOp,
    PassRegistry,
    BasePass,
    Any,
    RewriteResult,
)
from graph_optimizer.utils.graph_utils import create_const_node, create_node

# ==============================================================================
# Helper functions (used by rewrite methods)
# ==============================================================================

def _get_node(optimizer, name):
    real_name = name.split(":")[0]
    return optimizer.nodes.get(real_name)

def _is_const_value(optimizer, node_name, value):
    node = _get_node(optimizer, node_name)
    if not node or node.op != "Const":
        return False
    val = optimizer.get_node_attr(node, "value")
    return np.all(np.equal(val, value))

def _get_shape(optimizer, node_name):
    node = _get_node(optimizer, node_name)
    if not node:
        return None
    return optimizer.get_node_shape(node)

def _get_broadcast_shape(s1, s2):
    if s1 is None or s2 is None: return None
    if s1 == s2: return s1
    try:
        return tf.broadcast_static_shape(tf.TensorShape(s1), tf.TensorShape(s2)).as_list()
    except (ValueError, tf.errors.OpError):
        return None # Incompatible shapes

def _check_shape_preservation(optimizer, op_node, keep_input_name, other_input_name):
    shape_op = optimizer.get_node_shape(op_node)
    shape_keep = _get_shape(optimizer, keep_input_name)
    if shape_op is None or shape_keep is None:
        shape_other = _get_shape(optimizer, other_input_name)
        return shape_other == []
    return shape_op == shape_keep

# ==============================================================================
# The Unified Algebraic Simplify Pass
# ==============================================================================

@PassRegistry.register("algebraic_simplify", opt_level=1, priority=7)
class AlgebraicSimplifyPass(BasePass):
    def __init__(self):
        super().__init__(name="AlgebraicSimplify", iterative=True)

    def transform_once(self, optimizer, auto_cleanup=True, protected_nodes=None):
        optimizer.clear_transformations()
        self._register_patterns(optimizer)
        new_graph_def, changes = optimizer.match_patterns_once(
            pass_name=self.name,
            auto_cleanup=auto_cleanup,
            protected_nodes=protected_nodes
        )
        if changes > 0:
            optimizer.load_state(new_graph_def)
        return changes

    def _register_patterns(self, optimizer):
        # Arithmetic patterns
        optimizer.add_transformation(
            CommutativeOp("Add", Any("x"), Op("Const", alias="c"), alias="op"), self._rewrite_add
        )
        optimizer.add_transformation(
            CommutativeOp("Add", Any("x"), Op("Neg", Any("y"), alias="neg"), alias="op"), self._rewrite_add_neg
        )
        optimizer.add_transformation(
            Op("Sub", Any("x"), Any("y"), alias="op"), self._rewrite_sub
        )
        optimizer.add_transformation(
            CommutativeOp("Mul", Any("x"), Any("y"), alias="op"), self._rewrite_mul
        )
        optimizer.add_transformation(
            Op("Div", Any("x"), Any("y"), alias="op"), self._rewrite_div
        )
        optimizer.add_transformation(
            Op("Neg", Op("Neg", Any("x"), alias="inner"), alias="op"), self._rewrite_double_inverse
        )
        # Logical patterns
        optimizer.add_transformation(
            Op("LogicalNot", Op("LogicalNot", Any("x"), alias="inner"), alias="op"), self._rewrite_double_inverse
        )
        for op_type in ["Equal", "NotEqual", "Less", "Greater", "LessEqual", "GreaterEqual"]:
            optimizer.add_transformation(Op(op_type, Any("x"), Any("y"), alias="op"), self._rewrite_identity_comparison)
        optimizer.add_transformation(
            CommutativeOp("LogicalAnd", Any("x"), Op("Const", alias="c"), alias="op"), self._rewrite_logical_and
        )
        optimizer.add_transformation(
            CommutativeOp("LogicalOr", Any("x"), Op("Const", alias="c"), alias="op"), self._rewrite_logical_or
        )
        # Other patterns
        optimizer.add_transformation(Op("Select", Any("c"), Any("x"), Any("y"), alias="op"), self._rewrite_select)
        optimizer.add_transformation(Op("Identity", Any("x"), alias="op"), self._rewrite_identity)

    # --- Rewrite Methods ---

    def _rewrite_add(self, match, optimizer):
        op, x, c = [match.matched_nodes[n] for n in ["op", "x", "c"]]
        if not _is_const_value(optimizer, c.name, 0): return None
        if _check_shape_preservation(optimizer, op, x.name, c.name):
            return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})
        return None

    def _rewrite_add_neg(self, match, optimizer):
        op, x, neg = [match.matched_nodes[n] for n in ["op", "x", "neg"]]
        if x.name != neg.input[0]: return None
        shape = _get_shape(optimizer, op.name)
        if shape is None: return None
        dtype = tf.DType(op.attr["T"].type)
        return [create_const_node(op.name, 0, dtype.name, shape)]

    def _rewrite_sub(self, match, optimizer):
        op, x, y = [match.matched_nodes[n] for n in ["op", "x", "y"]]
        y_node = _get_node(optimizer, y.name)
        if y_node and y_node.op == "Const" and _is_const_value(optimizer, y.name, 0):
             if _check_shape_preservation(optimizer, op, x.name, y.name):
                return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})
        elif x.name == y.name:
            shape = _get_shape(optimizer, op.name)
            if shape is None: return None
            dtype = tf.DType(op.attr["T"].type)
            return [create_const_node(op.name, 0, dtype.name, shape)]
        return None

    def _rewrite_mul(self, match, optimizer):
        op, x, y = [match.matched_nodes[n] for n in ["op", "x", "y"]]
        y_node = _get_node(optimizer, y.name)

        if y_node and y_node.op == "Const":
            if _is_const_value(optimizer, y.name, 1) and _check_shape_preservation(optimizer, op, x.name, y.name):
                return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})
            elif _is_const_value(optimizer, y.name, 0):
                shape = _get_shape(optimizer, op.name)
                if shape is None: return None
                dtype = tf.DType(op.attr["T"].type)
                return [create_const_node(op.name, 0, dtype.name, shape)]
        elif x.name == y.name:
            return [create_node("Square", op.name, [x.name], attr=op.attr)]
        return None

    def _rewrite_div(self, match, optimizer):
        op, x, y = [match.matched_nodes[n] for n in ["op", "x", "y"]]
        y_node = _get_node(optimizer, y.name)
        if y_node and y_node.op == "Const" and _is_const_value(optimizer, y.name, 1):
            if _check_shape_preservation(optimizer, op, x.name, y.name):
                return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})
        elif x.name == y.name:
            shape = _get_shape(optimizer, op.name)
            if shape is None: return None
            dtype = tf.DType(op.attr["T"].type)
            return [create_const_node(op.name, 1, dtype.name, shape)]
        return None

    def _rewrite_double_inverse(self, match, optimizer):
        op, inner = match.matched_nodes["op"], match.matched_nodes["inner"]
        return RewriteResult(new_nodes=[], node_mapping={op.name: inner.input[0]})

    def _rewrite_identity_comparison(self, match, optimizer):
        op = match.matched_nodes["op"]
        if len(op.input) < 2 or op.input[0] != op.input[1]: return None

        result_val = None
        if op.op in ("Equal", "LessEqual", "GreaterEqual"): result_val = True
        elif op.op in ("NotEqual", "Less", "Greater"): result_val = False

        if result_val is not None:
            shape = _get_shape(optimizer, op.input[0])
            if shape is None: return None
            return [create_const_node(op.name, value=result_val, dtype="bool", shape=shape)]
        return None

    def _rewrite_logical_and(self, match, optimizer):
        op, x, c = [match.matched_nodes[n] for n in ["op", "x", "c"]]
        if _is_const_value(optimizer, c.name, True):
            if _check_shape_preservation(optimizer, op, x.name, c.name):
                return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})
        elif _is_const_value(optimizer, c.name, False):
            shape = _get_shape(optimizer, op.name)
            if shape is None: return None
            return [create_const_node(op.name, False, "bool", shape)]
        return None

    def _rewrite_logical_or(self, match, optimizer):
        op, x, c = [match.matched_nodes[n] for n in ["op", "x", "c"]]
        if _is_const_value(optimizer, c.name, False):
            if _check_shape_preservation(optimizer, op, x.name, c.name):
                return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})
        elif _is_const_value(optimizer, c.name, True):
            shape = _get_shape(optimizer, op.name)
            if shape is None: return None
            return [create_const_node(op.name, True, "bool", shape)]
        return None

    def _rewrite_select(self, match, optimizer):
        op = match.matched_nodes["op"]
        if len(op.input) >= 3 and op.input[1] == op.input[2]:
            return RewriteResult(new_nodes=[], node_mapping={op.name: op.input[1]})
        return None

    def _rewrite_identity(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        if op.name in optimizer.protected_nodes: return None
        if "_class" in op.attr: return None
        return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})
