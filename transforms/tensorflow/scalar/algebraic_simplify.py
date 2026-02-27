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

import tensorflow.compat.v1 as tf
from graph_optimizer.core import PassRegistry
from graph_optimizer.core.tensorflow import (
    PatternRewritePass,
    Any,
    RewriteResult,
)
from graph_optimizer.utils.tf.graph_utils import (
    create_node,
    create_const_node,
    get_node_shape,
    get_node_dtype,
    is_scalar,
    is_const,
    get_broadcast_shape,
    extract_base_name,
    make_type_attr,
    make_output_shapes_attr,
)


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

        if op_type in ("Identity", "IdentityN"):
            # Treated as opaque/stateful in this pass if it's a ReadVariableOp
            inp_node = self._get_node(inputs[0], optimizer)
            if inp_node and inp_node.op == "ReadVariableOp":
                return None
            return self._mapped_result(name, inputs[0])

        # Rule: Maximum(x, x) -> x, Minimum(x, x) -> x
        if op_type in ("Maximum", "Minimum"):
            if extract_base_name(inputs[0]) == extract_base_name(inputs[1]):
                return self._mapped_result(name, inputs[0])

        if op_type in ("Add", "AddV2"):
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                get_node_shape(self._get_node(left, optimizer)),
                get_node_shape(self._get_node(right, optimizer)),
            )
            s_res = get_broadcast_shape(s_left, s_right)

            if is_const(
                self._get_node(left, optimizer), 0
            ) and self._is_shape_preserving(s_res, s_right):
                return self._mapped_result(name, right)
            if is_const(
                self._get_node(right, optimizer), 0
            ) and self._is_shape_preserving(s_res, s_left):
                return self._mapped_result(name, left)
            for lop, rop in [(left, right), (right, left)]:
                rn = self._get_node(rop, optimizer)
                if (
                    rn
                    and rn.op == "Neg"
                    and extract_base_name(rn.input[0]) == extract_base_name(lop)
                ):
                    return self._numeric_const(0, lop, name, optimizer)

        if op_type == "Sub":
            left, right = inputs[0], inputs[1]
            if is_const(self._get_node(right, optimizer), 0) and (
                is_scalar(self._get_node(right, optimizer))
                or get_node_shape(self._get_node(right, optimizer))
                == get_node_shape(self._get_node(left, optimizer))
            ):
                return self._mapped_result(name, left)
            if extract_base_name(left) == extract_base_name(right):
                return self._numeric_const(0, left, name, optimizer)

        if op_type == "Mul":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                get_node_shape(self._get_node(left, optimizer)),
                get_node_shape(self._get_node(right, optimizer)),
            )
            s_res = get_broadcast_shape(s_left, s_right)

            if is_const(
                self._get_node(left, optimizer), 1
            ) and self._is_shape_preserving(s_res, s_right):
                return self._mapped_result(name, right)
            if is_const(
                self._get_node(right, optimizer), 1
            ) and self._is_shape_preserving(s_res, s_left):
                return self._mapped_result(name, left)
            if is_const(self._get_node(left, optimizer), 0) or is_const(
                self._get_node(right, optimizer), 0
            ):
                if s_res is not None:
                    source_name = (
                        right if is_const(self._get_node(left, optimizer), 0) else left
                    )
                    return self._numeric_const(
                        0, source_name, name, optimizer, custom_shape=s_res
                    )
            if extract_base_name(left) == extract_base_name(right):
                return self._new_node_result(
                    name, create_node("Square", name + "_sq", inputs=[left])
                )

        if op_type in ("Div", "RealDiv", "FloorDiv"):
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                get_node_shape(self._get_node(left, optimizer)),
                get_node_shape(self._get_node(right, optimizer)),
            )
            s_res = get_broadcast_shape(s_left, s_right)

            if is_const(
                self._get_node(right, optimizer), 1
            ) and self._is_shape_preserving(s_res, s_left):
                return self._mapped_result(name, left)
            if extract_base_name(left) == extract_base_name(right):
                return self._numeric_const(1, left, name, optimizer)
            if is_const(self._get_node(left, optimizer), 0):
                if s_res is not None:
                    return self._numeric_const(
                        0, right, name, optimizer, custom_shape=s_res
                    )

        if op_type == "FloorMod":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                get_node_shape(self._get_node(left, optimizer)),
                get_node_shape(self._get_node(right, optimizer)),
            )
            s_res = get_broadcast_shape(s_left, s_right)
            if is_const(self._get_node(right, optimizer), 1) or extract_base_name(
                left
            ) == extract_base_name(right):
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
                # Abs(Abs(x)) -> Abs(x)
                dtype = get_node_dtype(self._get_node(inputs[0], optimizer))
                s = get_node_shape(node)
                attr = {"T": make_type_attr(dtype)}
                if s is not None:
                    attr["_output_shapes"] = make_output_shapes_attr([s])
                return self._new_node_result(
                    name,
                    create_node(
                        "Abs", name=name + "_abs", inputs=[inp.input[0]], attr=attr
                    ),
                )
        if op_type == "Square":
            inp = self._get_node(inputs[0], optimizer)
            if inp and inp.op == "Sqrt":
                return self._mapped_result(name, inp.input[0])
        if op_type == "Sqrt":
            inp_node = self._get_node(inputs[0], optimizer)
            if inp_node and inp_node.op == "Square":
                # Sqrt(Square(x)) -> Abs(x)
                dtype = get_node_dtype(self._get_node(inp_node.input[0], optimizer))
                s = get_node_shape(inp_node)
                attr = {"T": make_type_attr(dtype)}
                if s is not None:
                    attr["_output_shapes"] = make_output_shapes_attr([s])
                return self._new_node_result(
                    name,
                    create_node(
                        "Abs", name=name + "_abs", inputs=[inp_node.input[0]], attr=attr
                    ),
                )

        if op_type == "Pow":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                get_node_shape(self._get_node(left, optimizer)),
                get_node_shape(self._get_node(right, optimizer)),
            )
            s_res = get_broadcast_shape(s_left, s_right)
            if is_const(
                self._get_node(right, optimizer), 1
            ) and self._is_shape_preserving(s_res, s_left):
                return self._mapped_result(name, left)
            if is_const(
                self._get_node(right, optimizer), 2
            ) and self._is_shape_preserving(s_res, s_left):
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
            if extract_base_name(inputs[0]) == extract_base_name(inputs[1]):
                val = op_type in ("Equal", "LessEqual", "GreaterEqual")
                res = self._comparison_const(val, name, inputs[0], optimizer)
                return res if res else None

        if op_type == "LogicalAnd":
            left, right = inputs[0], inputs[1]
            s_left, s_right = (
                get_node_shape(self._get_node(left, optimizer)),
                get_node_shape(self._get_node(right, optimizer)),
            )
            s_res = get_broadcast_shape(s_left, s_right)
            if is_const(
                self._get_node(left, optimizer), True
            ) and self._is_shape_preserving(s_res, s_right):
                return self._mapped_result(name, right)
            if is_const(
                self._get_node(right, optimizer), True
            ) and self._is_shape_preserving(s_res, s_left):
                return self._mapped_result(name, left)
            if extract_base_name(left) == extract_base_name(right):
                return self._mapped_result(name, left)
            if is_const(self._get_node(left, optimizer), False) or is_const(
                self._get_node(right, optimizer), False
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
                get_node_shape(self._get_node(left, optimizer)),
                get_node_shape(self._get_node(right, optimizer)),
            )
            s_res = get_broadcast_shape(s_left, s_right)
            if is_const(
                self._get_node(left, optimizer), False
            ) and self._is_shape_preserving(s_res, s_right):
                return self._mapped_result(name, right)
            if is_const(
                self._get_node(right, optimizer), False
            ) and self._is_shape_preserving(s_res, s_left):
                return self._mapped_result(name, left)
            if extract_base_name(left) == extract_base_name(right):
                return self._mapped_result(name, left)
            if is_const(self._get_node(left, optimizer), True) or is_const(
                self._get_node(right, optimizer), True
            ):
                if s_res is not None:
                    return self._new_node_result(
                        name,
                        create_const_node(
                            name + "_bool", value=True, dtype="bool", shape=s_res
                        ),
                    )

        if op_type == "Select":
            if len(inputs) >= 3 and extract_base_name(inputs[1]) == extract_base_name(
                inputs[2]
            ):
                return self._mapped_result(name, inputs[1])

        return None

    # === Helper Methods (Refactored out of _rewrite for performance) ===

    def _get_node(self, name, optimizer):
        return optimizer.nodes.get(extract_base_name(name))

    def _mapped_result(self, name, target_name):
        return RewriteResult(new_nodes=[], node_mapping={name: target_name})

    def _new_node_result(self, name, new_node):
        return RewriteResult(new_nodes=[new_node], node_mapping={name: new_node.name})

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
        ref_node = self._get_node(ref_name, optimizer)
        s = custom_shape if custom_shape is not None else get_node_shape(ref_node)
        if s is None and ref_node is not None:
            # Try to infer from other inputs or fallback
            for inp_name in ref_node.input:
                s = get_node_shape(self._get_node(inp_name, optimizer))
                if s is not None:
                    break
        if s is None:
            return None
        dtype = get_node_dtype(ref_node)
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
        s = get_node_shape(self._get_node(ref_input, optimizer))
        if s is None:
            return None
        return self._new_node_result(
            original_name,
            create_const_node(
                original_name + "_bool", value=val, dtype=tf.bool, shape=s
            ),
        )
