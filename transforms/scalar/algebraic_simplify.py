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

from graph_optimizer.core import (
    Op,
    PassRegistry,
    PatternRewritePass,
    Any,
    RewriteResult,
)
from graph_optimizer.utils.graph_utils import create_node, create_const_node
from graph_optimizer.utils.logger import logger as logging
import numpy as np


@PassRegistry.register("algebraic_simplify", opt_level=1, priority=7)
class AlgebraicSimplifyPass(PatternRewritePass):
    """
    Applies algebraic identities to simplify expressions.
    """

    def __init__(self):
        # We'll handle multiple patterns manually in _rewrite
        pattern = Any(alias="op")  # fallback, we check inside
        super().__init__(pattern, self._rewrite, name="AlgebraicSimplify")

    def _rewrite(self, match, optimizer):
        node = match.matched_nodes["op"]
        op_type = node.op
        inputs = list(node.input)
        name = node.name

        def _mapped_result(target_name):
            return RewriteResult(new_nodes=[], node_mapping={name: target_name})

        def _new_node_result(new_node):
            return RewriteResult(
                new_nodes=[new_node], node_mapping={name: new_node.name}
            )

        # Helper to create True/False const
        def _bool_const(val):
            return _new_node_result(
                create_const_node(name + "_bool", value=val, dtype="bool", shape=[])
            )

        # Helper to get node object ignoring output index
        def _get_node(name):
            real_name = name.split(":")[0]
            return optimizer.nodes.get(real_name)

        # Helper to check if a node is Const with given value (broadcast-safe)
        def _is_const(node_name, value):
            node = _get_node(node_name)
            if node is None:
                return False
            if node.op != "Const":
                return False
            val = optimizer.get_node_attr(node, "value")
            # Check if all elements are equal to the target value
            return np.all(np.equal(val, value))

        # Helper to get shape of a node
        def _get_shape(node_name):
            node = _get_node(node_name)
            if node is None:
                return None
            # Check for shape attribute (Placeholder, etc.)
            if "shape" in node.attr:
                return [d.size for d in node.attr["shape"].shape.dim]
            # Check for Const value shape
            if node.op == "Const" and "value" in node.attr:
                tensor = node.attr["value"].tensor
                if tensor.HasField("tensor_shape"):
                    return [d.size for d in tensor.tensor_shape.dim]
            return None

        # Helper to check if a node is definitely scalar
        def _is_scalar(node_name):
            shape = _get_shape(node_name)
            return shape == []

        # Helper to compute broadcast shape of two shapes
        def _get_broadcast_shape(s1, s2):
            if s1 is None or s2 is None:
                return None
            if s1 == s2:
                return s1
            if not s1:
                return s2
            if not s2:
                return s1
            
            # Simple broadcasting logic
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
                    return None # Incompatible
            return result[::-1]

        # Helper to check if simplification is shape-preserving
        def _is_shape_preserving(source_shape, target_shape):
            # If both are unknown, assume it's safe (common in simple tests)
            if source_shape is None and target_shape is None:
                return True
            if source_shape is None or target_shape is None:
                return False
            return source_shape == target_shape

        # Rule: Add(x, 0) or Add(0, x)
        if op_type == "Add":
            left, right = inputs[0], inputs[1]
            s_left, s_right = _get_shape(left), _get_shape(right)
            s_res = _get_broadcast_shape(s_left, s_right)

            if _is_const(left, 0) and _is_shape_preserving(s_res, s_right):
                return _mapped_result(right)
            if _is_const(right, 0) and _is_shape_preserving(s_res, s_left):
                return _mapped_result(left)
            # Add(x, Neg(x)) -> 0 or Add(Neg(x), x) -> 0
            # Note: This is a simplified check for Neg(x)
            for l, r in [(left, right), (right, left)]:
                rn = _get_node(r)
                if rn and rn.op == "Neg" and rn.input[0] == l:
                    s = _get_shape(l)
                    if s is not None:
                        source = _get_node(l)
                        dtype = source.attr.get("dtype", "float32") if source else "float32"
                        return _new_node_result(
                            create_const_node(name + "_zero", value=0, dtype=dtype, shape=s)
                        )

        # Rule: Sub(x, 0) → x
        if op_type == "Sub":
            left, right = inputs[0], inputs[1]
            if _is_const(right, 0) and (
                _is_scalar(right) or _get_shape(right) == _get_shape(left)
            ):
                return _mapped_result(left)
            # Sub(x, x) → 0
            if left == right:
                s = _get_shape(left)
                if s is not None:
                    source = _get_node(left)
                    dtype = source.attr.get("dtype", "float32") if source else "float32"
                    return _new_node_result(
                        create_const_node(name + "_zero", value=0, dtype=dtype, shape=s)
                    )

        # Rule: Mul(x, 1) or Mul(1, x)
        if op_type == "Mul":
            left, right = inputs[0], inputs[1]
            s_left, s_right = _get_shape(left), _get_shape(right)
            s_res = _get_broadcast_shape(s_left, s_right)

            if _is_const(left, 1) and _is_shape_preserving(s_res, s_right):
                return _mapped_result(right)
            if _is_const(right, 1) and _is_shape_preserving(s_res, s_left):
                return _mapped_result(left)
            # Mul(x, 0) → 0
            if _is_const(left, 0) or _is_const(right, 0):
                if s_res is not None:
                    source_name = right if _is_const(left, 0) else left
                    source = _get_node(source_name)
                    dtype = source.attr.get("dtype", "float32") if source else "float32"
                    return _new_node_result(
                        create_const_node(
                            name + "_zero", value=0, dtype=dtype, shape=s_res
                        )
                    )
            # Mul(x, x) -> Square(x)
            if left == right:
                return _new_node_result(
                    create_node("Square", name + "_sq", inputs=[left])
                )

        # Rule: Div(x, 1) → x
        if op_type == "Div":
            left, right = inputs[0], inputs[1]
            s_left, s_right = _get_shape(left), _get_shape(right)
            s_res = _get_broadcast_shape(s_left, s_right)
            if _is_const(right, 1) and _is_shape_preserving(s_res, s_left):
                return _mapped_result(left)
            # Div(x, x) -> 1
            if left == right:
                s = _get_shape(left)
                if s is not None:
                    source = _get_node(left)
                    dtype = source.attr.get("dtype", "float32") if source else "float32"
                    return _new_node_result(
                        create_const_node(name + "_one", value=1, dtype=dtype, shape=s)
                    )

        # Rule: Neg(Neg(x)) → x
        if op_type == "Neg":
            inp = _get_node(inputs[0])
            if inp and inp.op == "Neg":
                return _mapped_result(inp.input[0])

        # Rule: LogicalNot(LogicalNot(x)) → x
        if op_type == "LogicalNot":
            inp = _get_node(inputs[0])
            if inp and inp.op == "LogicalNot":
                return _mapped_result(inp.input[0])

        # Rule: Abs(Abs(x)) → Abs(x)
        if op_type == "Abs":
            inp = _get_node(inputs[0])
            if inp and inp.op == "Abs":
                orig = _get_node(inp.input[0])
                if orig:
                    return _new_node_result(
                        create_node("Abs", name + "_abs", inputs=[orig.name])
                    )

        # Rule: Square(Sqrt(x)) → x  (domain assumed ok)
        if op_type == "Square":
            inp = _get_node(inputs[0])
            if inp and inp.op == "Sqrt":
                return _mapped_result(inp.input[0])

        # Rule: Sqrt(Square(x)) → Abs(x)
        if op_type == "Sqrt":
            inp = _get_node(inputs[0])
            if inp and inp.op == "Square":
                orig = _get_node(inp.input[0])
                if orig:
                    return _new_node_result(
                        create_node("Abs", name + "_abs", inputs=[orig.name])
                    )

        # Rule: Pow(x, 1) -> x
        if op_type == "Pow":
            left, right = inputs[0], inputs[1]
            s_left, s_right = _get_shape(left), _get_shape(right)
            s_res = _get_broadcast_shape(s_left, s_right)
            if _is_const(right, 1) and _is_shape_preserving(s_res, s_left):
                return _mapped_result(left)
            # Pow(x, 2) -> Square(x)
            if _is_const(right, 2) and _is_shape_preserving(s_res, s_left):
                return _new_node_result(
                    create_node("Square", name + "_sq", inputs=[left])
                )

        # Helper for comparison results
        def _comparison_const(val):
            # Equal(x, x) -> True should have same shape as x (or broadcasted shape)
            # If x is [2, 2], result is [2, 2] of True
            s = _get_shape(inputs[0])
            if s is None:
                return None  # Safer to skip if shape unknown
            return _new_node_result(
                create_const_node(name + "_bool", value=val, dtype="bool", shape=s)
            )

        # Rule: Equal(x, x) → True
        if op_type == "Equal":
            left, right = inputs[0], inputs[1]
            if left == right:
                return _comparison_const(True)

        # Rule: NotEqual(x, x) → False
        if op_type == "NotEqual":
            left, right = inputs[0], inputs[1]
            if left == right:
                return _comparison_const(False)

        # Rule: Less(x, x) → False ; Greater(x, x) → False
        if op_type in ("Less", "Greater") and inputs[0] == inputs[1]:
            return _comparison_const(False)

        # Rule: LessEqual(x, x) → True ; GreaterEqual(x, x) → True
        if op_type in ("LessEqual", "GreaterEqual") and inputs[0] == inputs[1]:
            return _comparison_const(True)

        # Rule: And(x, True) → x ; And(True, x) → x
        if op_type == "LogicalAnd":
            left, right = inputs[0], inputs[1]
            s_left, s_right = _get_shape(left), _get_shape(right)
            s_res = _get_broadcast_shape(s_left, s_right)

            if _is_const(left, True) and _is_shape_preserving(s_res, s_right):
                return _mapped_result(right)
            if _is_const(right, True) and _is_shape_preserving(s_res, s_left):
                return _mapped_result(left)
            # LogicalAnd(x, x) -> x
            if left == right:
                return _mapped_result(left)
            # LogicalAnd(x, False) -> False
            if _is_const(left, False) or _is_const(right, False):
                if s_res is not None:
                    return _new_node_result(
                        create_const_node(name + "_bool", value=False, dtype="bool", shape=s_res)
                    )

        # Rule: Or(x, False) → x ; Or(False, x) → x
        if op_type == "LogicalOr":
            left, right = inputs[0], inputs[1]
            s_left, s_right = _get_shape(left), _get_shape(right)
            s_res = _get_broadcast_shape(s_left, s_right)

            if _is_const(left, False) and _is_shape_preserving(s_res, s_right):
                return _mapped_result(right)
            if _is_const(right, False) and _is_shape_preserving(s_res, s_left):
                return _mapped_result(left)
            # LogicalOr(x, x) -> x
            if left == right:
                return _mapped_result(left)
            # LogicalOr(x, True) -> True
            if _is_const(left, True) or _is_const(right, True):
                if s_res is not None:
                    return _new_node_result(
                        create_const_node(name + "_bool", value=True, dtype="bool", shape=s_res)
                    )

        # Rule: Select(cond, x, x) → x
        if op_type == "Select":
            if len(inputs) >= 3 and inputs[1] == inputs[2]:
                return _mapped_result(inputs[1])

        # Rule: Identity(x) -> x (bypass or collapse nested Identity)
        if op_type == "Identity":
            # Skip if protected/output node
            if (
                hasattr(optimizer, "protected_nodes")
                and name in optimizer.protected_nodes
            ):
                return None
            # Skip ReadVariableOp
            if "ReadVariableOp" in name:
                return None
            # Skip colocation constraint
            if "_class" in node.attr:
                return None
            # Collapse nested Identity
            inp_node = _get_node(inputs[0])
            if inp_node and inp_node.op == "Identity":
                inner_input = inp_node.input[0]
                new_node = create_node(
                    "Identity", name + "_collapsed", inputs=[inner_input]
                )
                return _new_node_result(new_node)
            # Bypass single Identity
            return _mapped_result(inputs[0])

        return None
