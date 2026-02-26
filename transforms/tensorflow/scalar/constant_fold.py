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
import logging as py_logging
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import types_pb2
from graph_optimizer.core import PassRegistry
from graph_optimizer.core.tensorflow import (
    PatternRewritePass,
    Any,
    RewriteResult,
)
from graph_optimizer.utils.tf.graph_utils import create_const_node


@PassRegistry.register("constant_fold", backend="tensorflow", opt_level=1, priority=5)
class ConstantFoldPass(PatternRewritePass):
    """
    Performs constant folding on eligible operation nodes.
    """

    # Map TF DT enum to numpy dtype
    TF_TO_NP = {
        types_pb2.DT_FLOAT: np.float32,
        types_pb2.DT_DOUBLE: np.float64,
        types_pb2.DT_INT32: np.int32,
        types_pb2.DT_INT64: np.int64,
        types_pb2.DT_BOOL: np.bool_,
        types_pb2.DT_UINT8: np.uint8,
        types_pb2.DT_INT16: np.int16,
        types_pb2.DT_INT8: np.int8,
    }

    def __init__(self):
        # Matches any operation with all inputs as Const
        pattern = Any(alias="op")
        super().__init__(pattern, self._rewrite_constant_op, name="ConstantFold")

        # Define supported ops mapping
        self._ops_map = {
            "Add": np.add,
            "Mul": np.multiply,
            "Sub": np.subtract,
            "Div": self._safe_div,
            "RealDiv": self._safe_div,
            "FloorDiv": self._safe_floor_div,
            "FloorMod": self._safe_mod,
            "Maximum": np.maximum,
            "Minimum": np.minimum,
            "Neg": np.negative,
            "Equal": np.equal,
            "NotEqual": np.not_equal,
            "Less": np.less,
            "Greater": np.greater,
            "LessEqual": np.less_equal,
            "GreaterEqual": np.greater_equal,
            "LogicalAnd": np.logical_and,
            "LogicalOr": np.logical_or,
            "LogicalNot": np.logical_not,
            "BitwiseAnd": np.bitwise_and,
            "BitwiseOr": np.bitwise_or,
            "BitwiseXor": np.bitwise_xor,
            "Abs": np.abs,
            "Exp": np.exp,
            "Expm1": np.expm1,
            "Log": self._safe_log,
            "Log1p": np.log1p,
            "Sqrt": self._safe_sqrt,
            "Pow": np.power,
            "Rsqrt": self._safe_rsqrt,
            "Square": np.square,
            "Sin": np.sin,
            "Cos": np.cos,
            "Tan": np.tan,
            "Asin": np.arcsin,
            "Acos": np.arccos,
            "Atan": np.arctan,
            "Atan2": np.arctan2,
            "Floor": np.floor,
            "Ceil": np.ceil,
            "Round": np.round,
            "Sign": np.sign,
        }

    @staticmethod
    def _safe_div(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(x, y)

    @staticmethod
    def _safe_floor_div(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.floor_divide(x, y)

    @staticmethod
    def _safe_mod(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.mod(x, y)

    @staticmethod
    def _safe_log(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(x)

    @staticmethod
    def _safe_sqrt(x):
        with np.errstate(invalid="ignore"):
            return np.sqrt(x)

    @staticmethod
    def _safe_rsqrt(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.sqrt(x)

    def _is_all_const(self, inputs, optimizer):
        for inp_name in inputs:
            if inp_name not in optimizer.nodes:
                return False
            inp_node = optimizer.nodes[inp_name]
            if inp_node.op != "Const" or "value" not in inp_node.attr:
                return False
        return True

    def _rewrite_constant_op(self, match, optimizer):
        op_node = match.matched_nodes["op"]
        if op_node.op == "Const":
            return None

        inputs = list(op_node.input)
        if not inputs or not self._is_all_const(inputs, optimizer):
            return None

        try:
            arrays = []
            input_dtypes = []
            for inp_name in inputs:
                inp = optimizer.nodes[inp_name]
                value_attr = inp.attr.get("value", None)
                if value_attr is None or not value_attr.HasField("tensor"):
                    return None
                arr = tensor_util.MakeNdarray(value_attr.tensor)
                arrays.append(arr)
                input_dtypes.append(arr.dtype)

            res_dtype = (
                np.result_type(*input_dtypes)
                if len(input_dtypes) > 1
                else input_dtypes[0]
            )
            op_type = op_node.op

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
                dst_t_attr = op_node.attr.get("DstT")
                if not dst_t_attr or dst_t_attr.type not in self.TF_TO_NP:
                    return None
                result = arrays[0].astype(self.TF_TO_NP[dst_t_attr.type])
            elif op_type in self._ops_map:
                f = self._ops_map[op_type]
                result = f(*arrays[:2]) if len(arrays) >= 2 else f(arrays[0])
            else:
                return None

            # Promote dtype if necessary
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
                "Cast",
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
            py_logging.error(f"Error folding {op_node.name}: {e}")
            return None
