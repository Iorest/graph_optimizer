"""
Specific algebraic simplification rules implemented as individual passes.
"""
from __future__ import annotations
from graph_optimizer.core import (
    Op,
    PassRegistry,
    PatternRewritePass,
    Any,
    RewriteResult,
    CommutativeOp,
)
from graph_optimizer.utils.graph_utils import create_node, create_const_node
import numpy as np

# Helper function to check if a node is a constant with a specific value
def is_const_value(node, optimizer, value):
    if node is None or node.op != "Const":
        return False
    val = optimizer.get_node_attr(node, "value")
    return np.all(np.equal(val, value))

@PassRegistry.register("simplify_add", opt_level=1, priority=7)
class SimplifyAddPass(PatternRewritePass):
    def __init__(self):
        pattern = CommutativeOp(
            "Add",
            Any(alias="x"),
            Op("Const", alias="c"),
            alias="root"
        )
        super().__init__(pattern, self._rewrite, name="SimplifyAdd")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        c = match.matched_nodes["c"]

        if is_const_value(c, optimizer, 0):
            # Add(x, 0) -> x
            # Shape preservation check
            s_x = optimizer.get_node_shape(x)
            s_root = optimizer.get_node_shape(root)
            if s_x == s_root:
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

@PassRegistry.register("simplify_sub", opt_level=1, priority=7)
class SimplifySubPass(PatternRewritePass):
    def __init__(self):
        pattern = Op(
            "Sub",
            Any(alias="x"),
            Op("Const", alias="c"),
            alias="root"
        )
        super().__init__(pattern, self._rewrite, name="SimplifySub")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        c = match.matched_nodes["c"]

        if is_const_value(c, optimizer, 0):
            # Sub(x, 0) -> x
            # Shape preservation check
            s_x = optimizer.get_node_shape(x)
            s_root = optimizer.get_node_shape(root)
            if s_x == s_root:
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

@PassRegistry.register("simplify_mul", opt_level=1, priority=7)
class SimplifyMulPass(PatternRewritePass):
    def __init__(self):
        pattern = CommutativeOp(
            "Mul",
            Any(alias="x"),
            Op("Const", alias="c"),
            alias="root"
        )
        super().__init__(pattern, self._rewrite, name="SimplifyMul")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        c = match.matched_nodes["c"]

        if is_const_value(c, optimizer, 1):
            # Mul(x, 1) -> x
            s_x = optimizer.get_node_shape(x)
            s_root = optimizer.get_node_shape(root)
            if s_x == s_root:
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        elif is_const_value(c, optimizer, 0):
            # Mul(x, 0) -> 0
            s_root = optimizer.get_node_shape(root)
            if s_root is not None:
                dtype = optimizer.get_node_attr(x, "dtype", "float32")
                zero_const = create_const_node(root.name + "_zero", value=0, dtype=dtype, shape=s_root)
                return RewriteResult(new_nodes=[zero_const], node_mapping={root.name: zero_const.name})
        return None

@PassRegistry.register("simplify_div", opt_level=1, priority=7)
class SimplifyDivPass(PatternRewritePass):
    def __init__(self):
        pattern = Op(
            "Div",
            Any(alias="x"),
            Op("Const", alias="c"),
            alias="root"
        )
        super().__init__(pattern, self._rewrite, name="SimplifyDiv")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        c = match.matched_nodes["c"]

        if is_const_value(c, optimizer, 1):
            # Div(x, 1) -> x
            s_x = optimizer.get_node_shape(x)
            s_root = optimizer.get_node_shape(root)
            if s_x == s_root:
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

@PassRegistry.register("simplify_neg", opt_level=1, priority=7)
class SimplifyNegPass(PatternRewritePass):
    def __init__(self):
        pattern = Op("Neg", Op("Neg", Any(alias="x"), alias="inner"), alias="root")
        super().__init__(pattern, self._rewrite, name="SimplifyNeg")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        # Neg(Neg(x)) -> x
        return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})

@PassRegistry.register("simplify_logical_not", opt_level=1, priority=7)
class SimplifyLogicalNotPass(PatternRewritePass):
    def __init__(self):
        pattern = Op("LogicalNot", Op("LogicalNot", Any(alias="x")), alias="root")
        super().__init__(pattern, self._rewrite, name="SimplifyLogicalNot")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        # LogicalNot(LogicalNot(x)) -> x
        return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})

@PassRegistry.register("simplify_redundant_comparison", opt_level=1, priority=7)
class SimplifyRedundantComparisonPass(PatternRewritePass):
    def __init__(self):
        pattern = Op("*", Any(alias="x"), Any(alias="y"), alias="root")
        super().__init__(pattern, self._rewrite, name="SimplifyRedundantComparison")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        op_type = root.op

        if x.name != y.name:
            return None

        # x == y
        s = optimizer.get_node_shape(x)
        if s is None:
            return None # Cannot create const if shape is unknown

        new_node = None
        if op_type == "Equal":
            # Equal(x, x) -> True
            new_node = create_const_node(root.name, value=True, dtype="bool", shape=s)
        if op_type == "NotEqual":
            # NotEqual(x, x) -> False
            new_node = create_const_node(root.name, value=False, dtype="bool", shape=s)
        if op_type in ("Less", "Greater"):
            # Less(x, x) -> False, Greater(x, x) -> False
            new_node = create_const_node(root.name, value=False, dtype="bool", shape=s)
        if op_type in ("LessEqual", "GreaterEqual"):
            # LessEqual(x, x) -> True, GreaterEqual(x, x) -> True
            new_node = create_const_node(root.name, value=True, dtype="bool", shape=s)

        if new_node:
            return RewriteResult(new_nodes=[new_node], node_mapping={root.name: new_node.name})

        return None

@PassRegistry.register("simplify_select", opt_level=1, priority=7)
class SimplifySelectPass(PatternRewritePass):
    def __init__(self):
        pattern = Op("Select", Any(), Any(alias="x"), Any(alias="y"), alias="root")
        super().__init__(pattern, self._rewrite, name="SimplifySelect")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]

        if x.name == y.name:
            # Select(cond, x, x) -> x
            return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

@PassRegistry.register("bypass_identity", opt_level=1, priority=8)
class BypassIdentityPass(PatternRewritePass):
    def __init__(self):
        pattern = Op("Identity", Any(alias="x"), alias="root")
        super().__init__(pattern, self._rewrite, name="BypassIdentity")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        x = match.matched_nodes["x"]

        # Do not remove identities that are protected (e.g., output nodes)
        if hasattr(optimizer, "protected_nodes") and root.name in optimizer.protected_nodes:
            return None

        return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
