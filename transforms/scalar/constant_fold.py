"""
Constant Folding Pass
=====================

Purpose:
--------
Performs constant folding by evaluating operations whose inputs are all constants
at compile time, replacing them with a single constant node containing the computed result.
This reduces runtime computation and enables further optimizations such as CSE.

Algorithm:
----------
1. Match patterns where an operation node has all inputs as `Const` nodes.
2. Extract the constant values and operation type.
3. Compute the result using NumPy (or Python fallback for simple ops).
4. Create a new `Const` node with the result and replace the original subgraph.
5. Repeat until no more constant-only ops remain.

Complexity:
-----------
- Time: O(N) for N constant-only nodes (each evaluated once)
- Space: O(1) auxiliary space per evaluation

Example:
--------
Input graph fragment:
  const_a = Const(value=[2, 3])
  const_b = Const(value=[4, 5])
  add = Add(const_a, const_b)

After ConstantFoldPass:
  folded = Const(value=[6, 8])

Relationships:
--------------
- Should run **before** or **between** CSE passes to allow folded constants to be eliminated as duplicates.
- Works well with `IdentityEliminationPass` (removes identity wrappers around constants).
- Can feed into `PackVectorizePass` if folded constants enable pack hoisting.
"""

from __future__ import annotations

import numpy as np
from graph_optimizer.core import (
    Op,
    PassRegistry,
    PatternRewritePass,
    Any,
    RewriteResult,
)
from graph_optimizer.utils.graph_utils import create_node, create_const_node


@PassRegistry.register("constant_fold", opt_level=1, priority=5)
class ConstantFoldPass(PatternRewritePass):
    """
    Performs constant folding on eligible operation nodes.
    """

    def __init__(self):
        # Matches any operation with all inputs as Const
        pattern = Any(alias="op")
        super().__init__(pattern, self._rewrite_constant_op, name="ConstantFold")

    def _is_all_const(self, inputs, optimizer):
        """Check if all inputs are Const nodes.

        Args:
            inputs: List of input node names (strings)
            optimizer: GraphOptimizer instance to lookup nodes
        """
        from tensorflow.python.framework import tensor_util

        for inp_name in inputs:
            if inp_name not in optimizer.nodes:
                return False
            inp_node = optimizer.nodes[inp_name]
            if inp_node.op != "Const":
                return False
            # Check if value attribute exists (basic check)
            if "value" not in inp_node.attr:
                return False
        return True

    def _rewrite_constant_op(self, match, optimizer):
        op_node = match.matched_nodes["op"]
        if op_node.op == "Const":
            return None

        inputs = list(op_node.input)
        if not inputs:
            return None

        if not self._is_all_const(inputs, optimizer):
            return None

        try:
            from tensorflow.python.framework import tensor_util

            dtypes = []
            for inp_name in inputs:
                # Look up node
                inp = optimizer.nodes[inp_name]
                dtype_attr = inp.attr.get("dtype", None)
                if dtype_attr is None:
                    return None
                tf_dtype = dtype_attr.type
                dtypes.append(tf_dtype)
            dtype = dtypes[0]
            arrays = []
            for inp_name in inputs:
                inp = optimizer.nodes[inp_name]
                value_attr = inp.attr.get("value", None)
                if value_attr is None or not value_attr.HasField("tensor"):
                    return None
                tensor = value_attr.tensor
                arr = tensor_util.MakeNdarray(tensor)
                arrays.append(arr)

            op_type = op_node.op

            # Define supported ops
            def _add(x, y):
                return np.add(x, y)

            def _mul(x, y):
                return np.multiply(x, y)

            def _sub(x, y):
                return np.subtract(x, y)

            def _div(x, y):
                # Safety: check for division by zero to avoid inf/nan nodes
                if np.any(y == 0):
                    raise ValueError("Division by zero in constant folding")
                return np.divide(x, y)

            def _neg(x):
                return np.negative(x)

            def _equal(x, y):
                return np.equal(x, y)

            def _not_equal(x, y):
                return np.not_equal(x, y)

            def _less(x, y):
                return np.less(x, y)

            def _greater(x, y):
                return np.greater(x, y)

            def _less_equal(x, y):
                return np.less_equal(x, y)

            def _greater_equal(x, y):
                return np.greater_equal(x, y)

            def _logical_and(x, y):
                return np.logical_and(x, y)

            def _logical_or(x, y):
                return np.logical_or(x, y)

            def _logical_not(x):
                return np.logical_not(x)

            def _bitwise_and(x, y):
                return np.bitwise_and(x.astype(np.int64), y.astype(np.int64))

            def _bitwise_or(x, y):
                return np.bitwise_or(x.astype(np.int64), y.astype(np.int64))

            def _bitwise_xor(x, y):
                return np.bitwise_xor(x.astype(np.int64), y.astype(np.int64))

            def _abs(x):
                return np.abs(x)

            def _exp(x):
                return np.exp(x)

            def _expm1(x):
                return np.expm1(x)

            def _log(x):
                return np.log(x)

            def _log1p(x):
                return np.log1p(x)

            def _sqrt(x):
                return np.sqrt(x)

            def _pow(x, y):
                return np.power(x, y)

            def _rsqrt(x):
                return 1.0 / np.sqrt(x)

            def _square(x):
                return np.square(x)

            def _sin(x):
                return np.sin(x)

            def _cos(x):
                return np.cos(x)

            def _tan(x):
                return np.tan(x)

            def _asin(x):
                return np.arcsin(x)

            def _acos(x):
                return np.arccos(x)

            def _atan(x):
                return np.arctan(x)

            def _atan2(y, x):
                return np.arctan2(y, x)

            def _floor(x):
                return np.floor(x)

            def _ceil(x):
                return np.ceil(x)

            def _round(x):
                return np.round(x)

            def _sign(x):
                return np.sign(x)

            def _reshape(x, shape):
                return np.reshape(x, shape)

            def _transpose(x, axes):
                return np.transpose(x, axes)

            def _concatenate(x_list, axis=0):
                return np.concatenate(x_list, axis=axis)

            def _select(cond, x, y):
                return np.where(cond, x, y)

            ops_map = {
                "Add": lambda: _add(*arrays[:2]),
                "Mul": lambda: _mul(*arrays[:2]),
                "Sub": lambda: _sub(*arrays[:2]),
                "Div": lambda: _div(*arrays[:2]),
                "Neg": lambda: _neg(arrays[0]),
                "Equal": lambda: _equal(*arrays[:2]),
                "NotEqual": lambda: _not_equal(*arrays[:2]),
                "Less": lambda: _less(*arrays[:2]),
                "Greater": lambda: _greater(*arrays[:2]),
                "LessEqual": lambda: _less_equal(*arrays[:2]),
                "GreaterEqual": lambda: _greater_equal(*arrays[:2]),
                "LogicalAnd": lambda: _logical_and(*arrays[:2]),
                "LogicalOr": lambda: _logical_or(*arrays[:2]),
                "LogicalNot": lambda: _logical_not(arrays[0]),
                "BitwiseAnd": lambda: _bitwise_and(*arrays[:2]),
                "BitwiseOr": lambda: _bitwise_or(*arrays[:2]),
                "BitwiseXor": lambda: _bitwise_xor(*arrays[:2]),
                "Abs": lambda: _abs(arrays[0]),
                "Exp": lambda: _exp(arrays[0]),
                "Expm1": lambda: _expm1(arrays[0]),
                "Log": lambda: _log(arrays[0]),
                "Log1p": lambda: _log1p(arrays[0]),
                "Sqrt": lambda: _sqrt(arrays[0]),
                "Pow": lambda: _pow(*arrays[:2]),
                "Rsqrt": lambda: _rsqrt(arrays[0]),
                "Square": lambda: _square(arrays[0]),
                "Sin": lambda: _sin(arrays[0]),
                "Cos": lambda: _cos(arrays[0]),
                "Tan": lambda: _tan(arrays[0]),
                "Asin": lambda: _asin(arrays[0]),
                "Acos": lambda: _acos(arrays[0]),
                "Atan": lambda: _atan(arrays[0]),
                "Atan2": lambda: _atan2(*arrays[:2]),
                "Floor": lambda: _floor(arrays[0]),
                "Ceil": lambda: _ceil(arrays[0]),
                "Round": lambda: _round(arrays[0]),
                "Sign": lambda: _sign(arrays[0]),
            }

            # Handle special cases requiring extra attrs
            if op_type == "Reshape":
                shape_arr = arrays[1]
                if shape_arr.ndim != 1:
                    return None
                result = _reshape(arrays[0], tuple(shape_arr.astype(int)))
            elif op_type == "Transpose":
                axes_arr = arrays[1]
                if axes_arr.ndim != 1:
                    return None
                result = _transpose(arrays[0], tuple(axes_arr.astype(int)))
            elif op_type == "ConcatV2":
                axis_val = int(arrays[-1])
                result = _concatenate(arrays[:-1], axis=axis_val)
            elif op_type == "Select":
                result = _select(*arrays[:3])
            elif op_type == "Cast":
                # Cast to target dtype
                dst_t_attr = op_node.attr.get("DstT", None)
                if dst_t_attr is None:
                    return None
                dst_dtype = np.dtype(dst_t_attr.type)
                result = arrays[0].astype(dst_dtype)
            else:
                if op_type in ops_map:
                    result = ops_map[op_type]()
                else:
                    return None

            new_const = create_const_node(
                name=f"{op_node.name}_folded",
                value=result.tolist(),
                dtype=str(result.dtype),
                shape=list(result.shape),
            )
            return RewriteResult(
                new_nodes=[new_const], node_mapping={op_node.name: new_const.name}
            )
        except Exception:
            return None
