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
        # Define supported ops
        supported_ops = [
            "Add", "Mul", "Sub", "Div", "Neg", "Equal", "NotEqual", "Less", "Greater",
            "LessEqual", "GreaterEqual", "LogicalAnd", "LogicalOr", "LogicalNot",
            "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Abs", "Exp", "Expm1", "Log",
            "Log1p", "Sqrt", "Pow", "Rsqrt", "Square", "Sin", "Cos", "Tan", "Asin",
            "Acos", "Atan", "Atan2", "Floor", "Ceil", "Round", "Sign", "Reshape",
            "Transpose", "ConcatV2", "Select", "Cast"
        ]
        # Register specific Op patterns to leverage indexed matching
        patterns = [(Op(op, alias="op"), self._rewrite_constant_op) for op in supported_ops]
        super().__init__(patterns=patterns, name="ConstantFold")

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

            # Extract arrays and their dtypes
            arrays = []
            input_dtypes = []
            for inp_name in inputs:
                inp = optimizer.nodes[inp_name]
                value_attr = inp.attr.get("value", None)
                if value_attr is None or not value_attr.HasField("tensor"):
                    return None
                tensor = value_attr.tensor
                arr = tensor_util.MakeNdarray(tensor)
                arrays.append(arr)
                input_dtypes.append(arr.dtype)

            # Determine result dtype using numpy promotion rules
            try:
                if len(input_dtypes) > 1:
                    res_dtype = np.result_type(*input_dtypes)
                elif len(input_dtypes) == 1:
                    res_dtype = input_dtypes[0]
                else:
                    return None
            except Exception:
                res_dtype = input_dtypes[0]

            op_type = op_node.op

            # Define operations map (using static definitions to avoid re-creation)
            ops_map = {
                "Add": lambda: np.add(arrays[0], arrays[1]),
                "Mul": lambda: np.multiply(arrays[0], arrays[1]),
                "Sub": lambda: np.subtract(arrays[0], arrays[1]),
                "Div": lambda: np.divide(arrays[0], arrays[1]),
                "Neg": lambda: np.negative(arrays[0]),
                "Equal": lambda: np.equal(arrays[0], arrays[1]),
                "NotEqual": lambda: np.not_equal(arrays[0], arrays[1]),
                "Less": lambda: np.less(arrays[0], arrays[1]),
                "Greater": lambda: np.greater(arrays[0], arrays[1]),
                "LessEqual": lambda: np.less_equal(arrays[0], arrays[1]),
                "GreaterEqual": lambda: np.greater_equal(arrays[0], arrays[1]),
                "LogicalAnd": lambda: np.logical_and(arrays[0], arrays[1]),
                "LogicalOr": lambda: np.logical_or(arrays[0], arrays[1]),
                "LogicalNot": lambda: np.logical_not(arrays[0]),
                "BitwiseAnd": lambda: np.bitwise_and(arrays[0].astype(np.int64), arrays[1].astype(np.int64)),
                "BitwiseOr": lambda: np.bitwise_or(arrays[0].astype(np.int64), arrays[1].astype(np.int64)),
                "BitwiseXor": lambda: np.bitwise_xor(arrays[0].astype(np.int64), arrays[1].astype(np.int64)),
                "Abs": lambda: np.abs(arrays[0]),
                "Exp": lambda: np.exp(arrays[0]),
                "Expm1": lambda: np.expm1(arrays[0]),
                "Log": lambda: np.log(arrays[0]),
                "Log1p": lambda: np.log1p(arrays[0]),
                "Sqrt": lambda: np.sqrt(arrays[0]),
                "Pow": lambda: np.power(arrays[0], arrays[1]),
                "Rsqrt": lambda: 1.0 / np.sqrt(arrays[0]),
                "Square": lambda: np.square(arrays[0]),
                "Sin": lambda: np.sin(arrays[0]),
                "Cos": lambda: np.cos(arrays[0]),
                "Tan": lambda: np.tan(arrays[0]),
                "Asin": lambda: np.arcsin(arrays[0]),
                "Acos": lambda: np.arccos(arrays[0]),
                "Atan": lambda: np.arctan(arrays[0]),
                "Atan2": lambda: np.arctan2(arrays[0], arrays[1]),
                "Floor": lambda: np.floor(arrays[0]),
                "Ceil": lambda: np.ceil(arrays[0]),
                "Round": lambda: np.round(arrays[0]),
                "Sign": lambda: np.sign(arrays[0]),
            }

            # Handle special cases requiring extra attrs
            if op_type == "Reshape":
                shape_arr = arrays[1]
                if shape_arr.ndim != 1:
                    return None
                result = np.reshape(arrays[0], tuple(shape_arr.astype(int)))
            elif op_type == "Transpose":
                axes_arr = arrays[1]
                if axes_arr.ndim != 1:
                    return None
                result = np.transpose(arrays[0], tuple(axes_arr.astype(int)))
            elif op_type == "ConcatV2":
                axis_val = int(arrays[-1])
                result = np.concatenate(arrays[:-1], axis=axis_val)
            elif op_type == "Select":
                result = np.where(arrays[0], arrays[1], arrays[2])
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

            # Ensure result has the expected promoted dtype if necessary
            # For some ops like Equal, the result is always bool, we don't force res_dtype.
            if op_type not in (
                "Equal",
                "NotEqual",
                "Less",
                "Greater",
                "LessEqual",
                "GreaterEqual",
                "LogicalAnd",
                "LogicalOr",
                "LogicalNot",
            ):
                if result.dtype != res_dtype:
                    result = result.astype(res_dtype)

            new_const = create_const_node(
                name=f"{op_node.name}_folded",
                value=result.tolist(),
                dtype=str(result.dtype),
                shape=list(result.shape),
            )
            return RewriteResult(
                new_nodes=[new_const], node_mapping={op_node.name: new_const.name}
            )
        except Exception as e:
            import logging as py_logging

            py_logging.error(f"Error folding {op_node.name}: {e}")
            return None
