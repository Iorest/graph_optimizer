"""
Algebraic Simplify Pass
=======================

Purpose:
--------
Performs algebraic simplification by applying identity laws, zero-element elimination,
and inverse operation cancellation on graph operations. This includes transforming
operations like `Add(x, 0) → x`, `Mul(x, 1) → x`, `Neg(Neg(x)) → x`, etc.

This pass generalizes `IdentityEliminationPass` by covering arithmetic, logical, and
comparison identities beyond pure Identity nodes.

Algorithm:
----------
1. Define patterns for common algebraic identities where one or more inputs are
   constants or repeated variables.
2. Match these patterns in the graph.
3. Replace matched subgraphs with simplified expressions according to algebra rules.
4. Run iteratively until no more simplifications apply (convergence).

Supported identities include:
- Add(x, 0) → x ; Add(0, x) → x
- Sub(x, 0) → x
- Mul(x, 1) → x ; Mul(1, x) → x
- Mul(x, 0) → 0 (with care for broadcasting)
- Div(x, 1) → x
- Neg(Neg(x)) → x
- LogicalNot(LogicalNot(x)) → x
- Abs(Abs(x)) → Abs(x)
- Square(Sqrt(x)) → x  (for nonnegative x, in practice applied if domain not violated)
- Sqrt(Square(x)) → Abs(x)
- Equal(x, x) → True
- NotEqual(x, x) → False
- Less(x, x) → False
- Greater(x, x) → False
- LessEqual(x, x) → True
- GreaterEqual(x, x) → True
- And(x, True) → x ; And(True, x) → x
- Or(x, False) → x ; Or(False, x) → x
- Select(cond, x, x) → x

Complexity:
-----------
- Time: O(N) per iteration for N nodes, typically converges in few iterations.
- Space: O(1) auxiliary space per pattern match.

Example:
--------
Example 1 - Add zero:
    Original: y = Add(x, Const(0))
    Optimized: y = x

Example 2 - Double negation:
    Original: y = Neg(Neg(x))
    Optimized: y = x

Example 3 - Compare equal:
    Original: y = Equal(a, a)
    Optimized: y = Const(True)

Relationships:
--------------
- Runs after `ConstantFoldPass` (to fold constants before simplifying forms).
- Runs before `IdentityEliminationPass` (to reduce cases like Identity(Add(x,0))).
- Helps `CSEPass` by producing simpler, more canonical expressions.
"""

from __future__ import annotations
import numpy as np
from graph_optimizer.core import (
    Any,
    BasePass,
    CommutativeOp,
    ConstValue,
    Op,
    PassRegistry,
    RewriteResult,
)
from graph_optimizer.utils.graph_utils import create_const_node, create_node
from graph_optimizer.utils.logger import logger as logging

@PassRegistry.register("algebraic_simplify", opt_level=1, priority=7)
class AlgebraicSimplifyPass(BasePass):
    def __init__(self):
        super().__init__(name="AlgebraicSimplify", iterative=True)
        self.rewrite_rules = [
            (CommutativeOp("Add", Any(alias="x"), ConstValue(0, alias="const"), alias="op"), self._rewrite_add_zero),
            (CommutativeOp("Add", Any(alias="x"), Op("Neg", Any(alias="y")), alias="op"), self._rewrite_add_neg),
            (CommutativeOp("Mul", Any(alias="x"), ConstValue(1, alias="const"), alias="op"), self._rewrite_mul_one),
            (CommutativeOp("Mul", Any(alias="x"), ConstValue(0, alias="const"), alias="op"), self._rewrite_mul_zero),
            (Op("Mul", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_mul_self_to_square),
            (CommutativeOp("LogicalAnd", Any(alias="x"), ConstValue(True, alias="const"), alias="op"), self._rewrite_logicaland_true),
            (CommutativeOp("LogicalAnd", Any(alias="x"), ConstValue(False, alias="const"), alias="op"), self._rewrite_logicaland_false),
            (Op("LogicalAnd", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_logicaland_self),
            (CommutativeOp("LogicalOr", Any(alias="x"), ConstValue(False, alias="const"), alias="op"), self._rewrite_logicalor_false),
            (CommutativeOp("LogicalOr", Any(alias="x"), ConstValue(True, alias="const"), alias="op"), self._rewrite_logicalor_true),
            (Op("LogicalOr", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_logicalor_self),
            (Op("Sub", Any(alias="x"), ConstValue(0, alias="const"), alias="op"), self._rewrite_sub_zero),
            (Op("Sub", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_sub_self),
            (Op("Div", Any(alias="x"), ConstValue(1, alias="const"), alias="op"), self._rewrite_div_one),
            (Op("Div", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_div_self),
            (Op("Neg", Op("Neg", Any(alias="x")), alias="op"), self._rewrite_double_neg),
            (Op("LogicalNot", Op("LogicalNot", Any(alias="x")), alias="op"), self._rewrite_double_logical_not),
            (Op("Abs", Op("Abs", Any(alias="x")), alias="op"), self._rewrite_double_abs),
            (Op("Square", Op("Sqrt", Any(alias="x")), alias="op"), self._rewrite_square_sqrt),
            (Op("Sqrt", Op("Square", Any(alias="x")), alias="op"), self._rewrite_sqrt_square),
            (Op("Pow", Any(alias="x"), ConstValue(1, alias="const"), alias="op"), self._rewrite_pow_one),
            (Op("Pow", Any(alias="x"), ConstValue(2, alias="const"), alias="op"), self._rewrite_pow_two),
            (Op("Equal", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_equal_self),
            (Op("NotEqual", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_not_equal_self),
            (Op("Less", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_less_self),
            (Op("Greater", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_greater_self),
            (Op("LessEqual", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_less_equal_self),
            (Op("GreaterEqual", Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_greater_equal_self),
            (Op("Select", Any(), Any(alias="x"), Any(alias="y"), alias="op"), self._rewrite_select_self),
            (Op("Identity", Any(alias="x"), alias="op"), self._rewrite_identity),
        ]

    def _get_shape_safe(self, optimizer, node_or_name):
        return optimizer.get_node_shape(node_or_name)

    def _get_broadcast_shape(self, s1, s2):
        if s1 is None or s2 is None:
            return None
        if s1 == s2:
            return s1
        if not s1:
            return s2
        if not s2:
            return s1
        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)
        result = []
        for i in range(max_len):
            d1 = s1[len1 - 1 - i] if i < len1 else 1
            d2 = s2[len2 - 1 - i] if i < len2 else 1
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                return None
        return result[::-1]

    def _rewrite_add_zero(self, match, optimizer):
        op, x, const = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["const"]
        s_x = self._get_shape_safe(optimizer, x)
        s_const = self._get_shape_safe(optimizer, const)
        s_res = self._get_broadcast_shape(s_x, s_const)
        if s_res == s_x:
            return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_add_neg(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            s = self._get_shape_safe(optimizer, x)
            dtype = x.attr.get("dtype", "float32")
            new_node = create_const_node(op.name + "_zero", value=0, dtype=dtype, shape=s)
            return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_mul_one(self, match, optimizer):
        op, x, const = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["const"]
        s_x = self._get_shape_safe(optimizer, x)
        s_const = self._get_shape_safe(optimizer, const)
        s_res = self._get_broadcast_shape(s_x, s_const)
        if s_res == s_x:
            return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_mul_zero(self, match, optimizer):
        op, x, const = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["const"]
        s_x = self._get_shape_safe(optimizer, x)
        s_const = self._get_shape_safe(optimizer, const)
        s_res = self._get_broadcast_shape(s_x, s_const)
        if s_res is not None:
            dtype = x.attr.get("dtype", "float32")
            new_node = create_const_node(op.name + "_zero", value=0, dtype=dtype, shape=s_res)
            return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_mul_self_to_square(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            new_node = create_node("Square", op.name + "_sq", inputs=[x.name])
            return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_logicaland_true(self, match, optimizer):
        op, x, const = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["const"]
        s_x = self._get_shape_safe(optimizer, x)
        s_const = self._get_shape_safe(optimizer, const)
        s_res = self._get_broadcast_shape(s_x, s_const)
        if s_res == s_x:
            return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_logicaland_false(self, match, optimizer):
        op, x, const = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["const"]
        s_x = self._get_shape_safe(optimizer, x)
        s_const = self._get_shape_safe(optimizer, const)
        s_res = self._get_broadcast_shape(s_x, s_const)
        if s_res is not None:
            new_node = create_const_node(op.name + "_bool", value=False, dtype="bool", shape=s_res)
            return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_logicaland_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_logicalor_false(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        s_op = self._get_shape_safe(optimizer, op)
        s_x = self._get_shape_safe(optimizer, x)
        if s_op == s_x:
            return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_logicalor_true(self, match, optimizer):
        op = match.matched_nodes["op"]
        s_op = self._get_shape_safe(optimizer, op)
        new_node = create_const_node(op.name + "_bool", value=True, dtype="bool", shape=s_op)
        return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_logicalor_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_sub_zero(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_sub_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            s = self._get_shape_safe(optimizer, x)
            dtype = x.attr.get("dtype", "float32")
            new_node = create_const_node(op.name + "_zero", value=0, dtype=dtype, shape=s)
            return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_div_one(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_div_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            s = self._get_shape_safe(optimizer, x)
            dtype = x.attr.get("dtype", "float32")
            new_node = create_const_node(op.name + "_one", value=1, dtype=dtype, shape=s)
            return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_double_neg(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_double_logical_not(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_double_abs(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        new_node = create_node("Abs", op.name + "_abs", inputs=[x.name])
        return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_square_sqrt(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_sqrt_square(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        new_node = create_node("Abs", op.name + "_abs", inputs=[x.name])
        return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_pow_one(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_pow_two(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        new_node = create_node("Square", op.name + "_sq", inputs=[x.name])
        return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _comparison_const(self, op, x, value, optimizer):
        s = self._get_shape_safe(optimizer, x)
        new_node = create_const_node(op.name + "_bool", value=value, dtype="bool", shape=s)
        return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})

    def _rewrite_equal_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return self._comparison_const(op, x, True, optimizer)

    def _rewrite_not_equal_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return self._comparison_const(op, x, False, optimizer)

    def _rewrite_less_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return self._comparison_const(op, x, False, optimizer)

    def _rewrite_greater_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return self._comparison_const(op, x, False, optimizer)

    def _rewrite_less_equal_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return self._comparison_const(op, x, True, optimizer)

    def _rewrite_greater_equal_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return self._comparison_const(op, x, True, optimizer)

    def _rewrite_select_self(self, match, optimizer):
        op, x, y = match.matched_nodes["op"], match.matched_nodes["x"], match.matched_nodes["y"]
        if x.name == y.name:
            return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def _rewrite_identity(self, match, optimizer):
        op, x = match.matched_nodes["op"], match.matched_nodes["x"]
        if op.name in optimizer.protected_nodes or "ReadVariableOp" in op.name or "_class" in op.attr:
            return
        if x.op == "Identity":
            new_node = create_node("Identity", op.name + "_collapsed", inputs=[x.input[0]])
            return RewriteResult(new_nodes=[new_node], node_mapping={op.name: new_node.name})
        return RewriteResult(new_nodes=[], node_mapping={op.name: x.name})

    def transform_once(self, optimizer, auto_cleanup=True, protected_nodes=None):
        optimizer.clear_transformations()
        for pattern, rewriter in self.rewrite_rules:
            optimizer.add_transformation(pattern, rewriter)

        new_graph_def, changes = optimizer.match_patterns_once(
            pass_name=self.name,
            auto_cleanup=auto_cleanup,
            protected_nodes=protected_nodes,
        )

        if changes > 0:
            optimizer.load_state(new_graph_def)

        return changes
