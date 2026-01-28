
from __future__ import annotations

from __future__ import annotations

from graph_optimizer.core import (
    Any,
    CommutativeOp,
    Op,
    PassRegistry,
    PatternRewritePass,
    RewriteResult,
)
from graph_optimizer.utils.graph_utils import create_const_node, get_node_shape, is_const_with_value, get_broadcast_shape, is_shape_preserving

# ==============================================================================
# Helper functions
# ==============================================================================

def _comparison_const(optimizer, root_node, value):
    """Helper to create a boolean constant with the same shape as the input."""
    # The result of a comparison should have a shape determined by broadcasting the inputs.
    # For self-comparisons (e.g., Equal(x,x)), the shape is simply the shape of x.
    input_node = optimizer.nodes.get(root_node.input[0].split(":")[0])
    if input_node:
        shape = get_node_shape(optimizer, input_node)
        if shape is not None:
            const_node = create_const_node(root_node.name + "_bool", value=value, dtype="bool", shape=shape)
            return RewriteResult(new_nodes=[const_node], node_mapping={root_node.name: const_node.name})
    return None

# ==============================================================================
# Logical and Comparison Simplification Patterns
# ==============================================================================

# Rule: LogicalNot(LogicalNot(x)) -> x
@PassRegistry.register("logical_simplify_double_not", opt_level=1, priority=7)
class DoubleNotPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("LogicalNot", Op("LogicalNot", Any(alias="x"), alias="inner_not"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="DoubleNot")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        root = match.matched_nodes["root"]
        return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})

# Rule: Equal(x, x) -> True
@PassRegistry.register("comparison_simplify_equal_self", opt_level=1, priority=7)
class EqualSelfPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("Equal", Any(alias="x"), Any(alias="y"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="EqualSelf")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        root = match.matched_nodes["root"]
        if x.name == y.name:
            return _comparison_const(optimizer, root, True)
        return None

# Rule: NotEqual(x, x) -> False
@PassRegistry.register("comparison_simplify_not_equal_self", opt_level=1, priority=7)
class NotEqualSelfPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("NotEqual", Any(alias="x"), Any(alias="y"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="NotEqualSelf")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        root = match.matched_nodes["root"]
        if x.name == y.name:
            return _comparison_const(optimizer, root, False)
        return None

# Rules: Less(x, x) -> False, Greater(x, x) -> False
@PassRegistry.register("comparison_simplify_less_greater_self", opt_level=1, priority=7)
class LessGreaterSelfPass(PatternRewritePass):
    def __init__(self):
        # This pattern is a bit broader, we'll check the op type inside
        self.pattern = Any(alias="root")
        super().__init__(self.pattern, self._rewrite, name="LessGreaterSelf")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        if root.op in ("Less", "Greater") and len(root.input) == 2 and root.input[0] == root.input[1]:
            return _comparison_const(optimizer, root, False)
        return None

# Rules: LessEqual(x, x) -> True, GreaterEqual(x, x) -> True
@PassRegistry.register("comparison_simplify_le_ge_self", opt_level=1, priority=7)
class LessEqualGreaterEqualSelfPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Any(alias="root")
        super().__init__(self.pattern, self._rewrite, name="LessEqualGreaterEqualSelf")

    def _rewrite(self, match, optimizer):
        root = match.matched_nodes["root"]
        if root.op in ("LessEqual", "GreaterEqual") and len(root.input) == 2 and root.input[0] == root.input[1]:
            return _comparison_const(optimizer, root, True)
        return None

# Rule: LogicalAnd(x, True) -> x
@PassRegistry.register("logical_simplify_and_true", opt_level=1, priority=7)
class LogicalAndTruePass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "LogicalAnd",
            Any(alias="x"),
            Op("Const", alias="true_const"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="LogicalAndTrue")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        true_const = match.matched_nodes["true_const"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, true_const, True):
            s_x = get_node_shape(optimizer, x)
            s_true = get_node_shape(optimizer, true_const)
            s_res = get_broadcast_shape(s_x, s_true)
            if is_shape_preserving(s_res, s_x):
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: LogicalAnd(x, x) -> x
@PassRegistry.register("logical_simplify_and_self", opt_level=1, priority=7)
class LogicalAndSelfPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("LogicalAnd", Any(alias="x"), Any(alias="y"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="LogicalAndSelf")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        root = match.matched_nodes["root"]
        if x.name == y.name:
            return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: LogicalAnd(x, False) -> False
@PassRegistry.register("logical_simplify_and_false", opt_level=1, priority=7)
class LogicalAndFalsePass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "LogicalAnd",
            Any(alias="x"),
            Op("Const", alias="false_const"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="LogicalAndFalse")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        false_const = match.matched_nodes["false_const"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, false_const, False):
            s_x = get_node_shape(optimizer, x)
            s_false = get_node_shape(optimizer, false_const)
            s_res = get_broadcast_shape(s_x, s_false)
            if s_res is not None:
                new_const = create_const_node(root.name + "_bool", value=False, dtype="bool", shape=s_res)
                return RewriteResult(new_nodes=[new_const], node_mapping={root.name: new_const.name})
        return None

# Rule: LogicalOr(x, False) -> x
@PassRegistry.register("logical_simplify_or_false", opt_level=1, priority=7)
class LogicalOrFalsePass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "LogicalOr",
            Any(alias="x"),
            Op("Const", alias="false_const"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="LogicalOrFalse")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        false_const = match.matched_nodes["false_const"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, false_const, False):
            s_x = get_node_shape(optimizer, x)
            s_false = get_node_shape(optimizer, false_const)
            s_res = get_broadcast_shape(s_x, s_false)
            if is_shape_preserving(s_res, s_x):
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: LogicalOr(x, x) -> x
@PassRegistry.register("logical_simplify_or_self", opt_level=1, priority=7)
class LogicalOrSelfPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("LogicalOr", Any(alias="x"), Any(alias="y"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="LogicalOrSelf")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        root = match.matched_nodes["root"]
        if x.name == y.name:
            return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: LogicalOr(x, True) -> True
@PassRegistry.register("logical_simplify_or_true", opt_level=1, priority=7)
class LogicalOrTruePass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "LogicalOr",
            Any(alias="x"),
            Op("Const", alias="true_const"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="LogicalOrTrue")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        true_const = match.matched_nodes["true_const"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, true_const, True):
            s_x = get_node_shape(optimizer, x)
            s_true = get_node_shape(optimizer, true_const)
            s_res = get_broadcast_shape(s_x, s_true)
            if s_res is not None:
                new_const = create_const_node(root.name + "_bool", value=True, dtype="bool", shape=s_res)
                return RewriteResult(new_nodes=[new_const], node_mapping={root.name: new_const.name})
        return None
