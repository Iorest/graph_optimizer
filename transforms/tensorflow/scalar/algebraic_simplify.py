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

from graph_optimizer.core import PassRegistry
from graph_optimizer.core.tensorflow import (
    PatternRewritePass,
    Any,
    RewriteResult,
)
from graph_optimizer.utils.tf.graph_utils import create_node, create_const_node
import numpy as np


@PassRegistry.register(
    "algebraic_simplify", backend="tensorflow", opt_level=1, priority=7
)
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

        # Rule: Maximum(x, x) -> x, Minimum(x, x) -> x
        if op_type in ("Maximum", "Minimum"):
            if inputs[0] == inputs[1]:
                return self._mapped_result(name, inputs[0])

        if op_type in ("Add", "AddV2"):
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                self._get_shape(left, optimizer),
                self._get_shape(right, optimizer),
            )
            s_res = self._get_broadcast_shape(s_left, s_right)

            if self._is_const(left, 0, optimizer) and self._is_shape_preserving(
                s_res, s_right
            ):
                return self._mapped_result(name, right)
            if self._is_const(right, 0, optimizer) and self._is_shape_preserving(
                s_res, s_left
            ):
                return self._mapped_result(name, left)
            for lop, rop in [(left, right), (right, left)]:
                rn = self._get_node(rop, optimizer)
                if rn and rn.op == "Neg" and rn.input[0] == lop:
                    return self._numeric_const(0, lop, name, optimizer)

        if op_type == "Sub":
            left, right = inputs[0], inputs[1]
            if self._is_const(right, 0, optimizer) and (
                self._is_scalar(right, optimizer)
                or self._get_shape(right, optimizer) == self._get_shape(left, optimizer)
            ):
                return self._mapped_result(name, left)
            if left == right:
                return self._numeric_const(0, left, name, optimizer)

        if op_type == "Mul":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                self._get_shape(left, optimizer),
                self._get_shape(right, optimizer),
            )
            s_res = self._get_broadcast_shape(s_left, s_right)

            if self._is_const(left, 1, optimizer) and self._is_shape_preserving(
                s_res, s_right
            ):
                return self._mapped_result(name, right)
            if self._is_const(right, 1, optimizer) and self._is_shape_preserving(
                s_res, s_left
            ):
                return self._mapped_result(name, left)
            if self._is_const(left, 0, optimizer) or self._is_const(
                right, 0, optimizer
            ):
                if s_res is not None:
                    source_name = right if self._is_const(left, 0, optimizer) else left
                    return self._numeric_const(
                        0, source_name, name, optimizer, custom_shape=s_res
                    )
            if left == right:
                return self._new_node_result(
                    name, create_node("Square", name + "_sq", inputs=[left])
                )

        if op_type in ("Div", "RealDiv", "FloorDiv"):
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                self._get_shape(left, optimizer),
                self._get_shape(right, optimizer),
            )
            s_res = self._get_broadcast_shape(s_left, s_right)

            if self._is_const(right, 1, optimizer) and self._is_shape_preserving(
                s_res, s_left
            ):
                return self._mapped_result(name, left)
            if left == right:
                return self._numeric_const(1, left, name, optimizer)
            if self._is_const(left, 0, optimizer):
                if s_res is not None:
                    return self._numeric_const(
                        0, right, name, optimizer, custom_shape=s_res
                    )

        if op_type == "FloorMod":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                self._get_shape(left, optimizer),
                self._get_shape(right, optimizer),
            )
            s_res = self._get_broadcast_shape(s_left, s_right)
            if self._is_const(right, 1, optimizer) or left == right:
                if s_res is not None:
                    return self._numeric_const(
                        0, left, name, optimizer, custom_shape=s_res
                    )

        if op_type in ("Neg", "LogicalNot"):
            inp = self._get_node(inputs[0], optimizer)
            if inp and inp.op == op_type:
                return self._mapped_result(name, inp.input[0])

        if op_type == "Abs":
            inp = self._get_node(inputs[0], optimizer)
            if inp and inp.op == "Abs":
                orig = self._get_node(inp.input[0], optimizer)
                if orig:
                    return self._new_node_result(
                        name, create_node("Abs", name + "_abs", inputs=[orig.name])
                    )

        if op_type == "Square":
            inp = self._get_node(inputs[0], optimizer)
            if inp and inp.op == "Sqrt":
                return self._mapped_result(name, inp.input[0])
        if op_type == "Sqrt":
            inp = self._get_node(inputs[0], optimizer)
            if inp and inp.op == "Square":
                orig = self._get_node(inp.input[0], optimizer)
                if orig:
                    return self._new_node_result(
                        name, create_node("Abs", name + "_abs", inputs=[orig.name])
                    )

        if op_type == "Pow":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                self._get_shape(left, optimizer),
                self._get_shape(right, optimizer),
            )
            s_res = self._get_broadcast_shape(s_left, s_right)
            if self._is_const(right, 1, optimizer) and self._is_shape_preserving(
                s_res, s_left
            ):
                return self._mapped_result(name, left)
            if self._is_const(right, 2, optimizer) and self._is_shape_preserving(
                s_res, s_left
            ):
                return self._new_node_result(
                    name, create_node("Square", name + "_sq", inputs=[left])
                )

        if op_type in (
            "Equal",
            "NotEqual",
            "Less",
            "Greater",
            "LessEqual",
            "GreaterEqual",
        ):
            if inputs[0] == inputs[1]:
                val = op_type in ("Equal", "LessEqual", "GreaterEqual")
                res = self._comparison_const(val, name, inputs[0], optimizer)
                return res if res else None

        if op_type == "LogicalAnd":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                self._get_shape(left, optimizer),
                self._get_shape(right, optimizer),
            )
            s_res = self._get_broadcast_shape(s_left, s_right)
            if self._is_const(left, True, optimizer) and self._is_shape_preserving(
                s_res, s_right
            ):
                return self._mapped_result(name, right)
            if self._is_const(right, True, optimizer) and self._is_shape_preserving(
                s_res, s_left
            ):
                return self._mapped_result(name, left)
            if left == right:
                return self._mapped_result(name, left)
            if self._is_const(left, False, optimizer) or self._is_const(
                right, False, optimizer
            ):
                if s_res is not None:
                    return self._new_node_result(
                        name,
                        create_const_node(
                            name + "_bool", value=False, dtype="bool", shape=s_res
                        ),
                    )

        if op_type == "LogicalOr":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                self._get_shape(left, optimizer),
                self._get_shape(right, optimizer),
            )
            s_res = self._get_broadcast_shape(s_left, s_right)
            if self._is_const(left, False, optimizer) and self._is_shape_preserving(
                s_res, s_right
            ):
                return self._mapped_result(name, right)
            if self._is_const(right, False, optimizer) and self._is_shape_preserving(
                s_res, s_left
            ):
                return self._mapped_result(name, left)
            if left == right:
                return self._mapped_result(name, left)
            if self._is_const(left, True, optimizer) or self._is_const(
                right, True, optimizer
            ):
                if s_res is not None:
                    return self._new_node_result(
                        name,
                        create_const_node(
                            name + "_bool", value=True, dtype="bool", shape=s_res
                        ),
                    )

        if op_type == "Select":
            if len(inputs) >= 3 and inputs[1] == inputs[2]:
                return self._mapped_result(name, inputs[1])

        # Rule: Identity(x) -> x
        if op_type == "Identity":
            if (
                hasattr(optimizer, "protected_nodes")
                and name in optimizer.protected_nodes
            ):
                return None
            if node.op == "ReadVariableOp" or "_class" in node.attr:
                return None
            inp_node = self._get_node(inputs[0], optimizer)
            if inp_node and inp_node.op == "Identity":
                inner_input = inp_node.input[0]
                return self._new_node_result(
                    name,
                    create_node("Identity", name + "_collapsed", inputs=[inner_input]),
                )
            return self._mapped_result(name, inputs[0])

        return None

    # === Helper Methods (Refactored out of _rewrite for performance) ===

    def _mapped_result(self, name, target_name):
        return RewriteResult(new_nodes=[], node_mapping={name: target_name})

    def _new_node_result(self, name, new_node):
        return RewriteResult(new_nodes=[new_node], node_mapping={name: new_node.name})

    def _get_node(self, name, optimizer):
        real_name = name.split(":")[0]
        return optimizer.nodes.get(real_name)

    def _is_const(self, node_name, value, optimizer):
        node = self._get_node(node_name, optimizer)
        if node is None or node.op != "Const":
            return False
        val = optimizer.get_node_attr(node, "value")
        return np.all(np.equal(val, value))

    def _get_shape(self, node_name, optimizer):
        node = self._get_node(node_name, optimizer)
        if node is None:
            return None
        if "shape" in node.attr:
            return [d.size for d in node.attr["shape"].shape.dim]
        if node.op == "Const" and "value" in node.attr:
            tensor = node.attr["value"].tensor
            if tensor.HasField("tensor_shape"):
                return [d.size for d in tensor.tensor_shape.dim]
        return None

    def _is_scalar(self, node_name, optimizer):
        return self._get_shape(node_name, optimizer) == []

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

    def _is_shape_preserving(self, source_shape, target_shape):
        if source_shape is None and target_shape is None:
            return True
        return (
            source_shape == target_shape
            if source_shape is not None and target_shape is not None
            else False
        )

    def _numeric_const(
        self, val, ref_name, original_name, optimizer, custom_shape=None
    ):
        s = (
            custom_shape
            if custom_shape is not None
            else self._get_shape(ref_name, optimizer)
        )
        if s is None:
            return None
        source = self._get_node(ref_name, optimizer)
        dtype = (
            optimizer.get_node_attr(source, "dtype", "float32") if source else "float32"
        )
        suffix = (
            "zero" if val == 0 else "one" if val == 1 else str(val).replace(".", "_")
        )
        return self._new_node_result(
            original_name,
            create_const_node(
                original_name + "_" + suffix, value=val, dtype=dtype, shape=s
            ),
        )

    def _comparison_const(self, val, original_name, ref_input, optimizer):
        s = self._get_shape(ref_input, optimizer)
        if s is None:
            return None
        return self._new_node_result(
            original_name,
            create_const_node(
                original_name + "_bool", value=val, dtype="bool", shape=s
            ),
        )

        return None
