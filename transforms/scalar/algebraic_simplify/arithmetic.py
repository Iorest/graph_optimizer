
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
from graph_optimizer.utils.graph_utils import create_node, create_const_node
from graph_optimizer.utils.graph_utils import get_node_shape, is_const_with_value, get_broadcast_shape, is_shape_preserving

# ==============================================================================
# Helper functions from the original pass
# ==============================================================================

def _get_node(optimizer, name):
    real_name = name.split(":")[0]
    return optimizer.nodes.get(real_name)

# ==============================================================================
# Arithmetic Simplification Patterns
# ==============================================================================

# Rule: Add(x, 0) -> x
@PassRegistry.register("arithmetic_simplify_add_zero", opt_level=1, priority=7)
class AddZeroPass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "Add",
            Any(alias="x"),
            Op("Const", alias="zero"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="AddZero")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        zero = match.matched_nodes["zero"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, zero, 0):
            s_x = get_node_shape(optimizer, x)
            s_zero = get_node_shape(optimizer, zero)
            s_res = get_broadcast_shape(s_x, s_zero)
            if is_shape_preserving(s_res, s_x):
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: Add(x, Neg(x)) -> 0
@PassRegistry.register("arithmetic_simplify_add_neg", opt_level=1, priority=7)
class AddNegPass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "Add",
            Any(alias="x"),
            Op("Neg", Any(alias="neg_x")),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="AddNeg")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        neg_x = match.matched_nodes["neg_x"]
        root = match.matched_nodes["root"]

        if x.name == neg_x.name:
            shape = get_node_shape(optimizer, x)
            if shape is not None:
                dtype = optimizer.get_node_attr(x, "dtype", "float32")
                zero_const = create_const_node(root.name + "_zero", value=0, dtype=dtype, shape=shape)
                return RewriteResult(new_nodes=[zero_const], node_mapping={root.name: zero_const.name})
        return None

# Rule: Sub(x, 0) -> x
@PassRegistry.register("arithmetic_simplify_sub_zero", opt_level=1, priority=7)
class SubZeroPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op(
            "Sub",
            Any(alias="x"),
            Op("Const", alias="zero"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="SubZero")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        zero = match.matched_nodes["zero"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, zero, 0):
            s_x = get_node_shape(optimizer, x)
            s_zero = get_node_shape(optimizer, zero)
            if s_x == s_zero or s_zero == []:
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: Sub(x, x) -> 0
@PassRegistry.register("arithmetic_simplify_sub_self", opt_level=1, priority=7)
class SubSelfPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op(
            "Sub",
            Any(alias="x"),
            Any(alias="y"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="SubSelf")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        root = match.matched_nodes["root"]

        if x.name == y.name:
            shape = get_node_shape(optimizer, x)
            if shape is not None:
                dtype = optimizer.get_node_attr(x, "dtype", "float32")
                zero_const = create_const_node(root.name + "_zero", value=0, dtype=dtype, shape=shape)
                return RewriteResult(new_nodes=[zero_const], node_mapping={root.name: zero_const.name})
        return None

# Rule: Mul(x, 1) -> x
@PassRegistry.register("arithmetic_simplify_mul_one", opt_level=1, priority=7)
class MulOnePass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "Mul",
            Any(alias="x"),
            Op("Const", alias="one"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="MulOne")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        one = match.matched_nodes["one"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, one, 1):
            s_x = get_node_shape(optimizer, x)
            s_one = get_node_shape(optimizer, one)
            s_res = get_broadcast_shape(s_x, s_one)
            if is_shape_preserving(s_res, s_x):
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: Mul(x, 0) -> 0
@PassRegistry.register("arithmetic_simplify_mul_zero", opt_level=1, priority=7)
class MulZeroPass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "Mul",
            Any(alias="x"),
            Op("Const", alias="zero"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="MulZero")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        zero = match.matched_nodes["zero"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, zero, 0):
            s_x = get_node_shape(optimizer, x)
            s_zero = get_node_shape(optimizer, zero)
            s_res = get_broadcast_shape(s_x, s_zero)
            if s_res is not None:
                dtype = optimizer.get_node_attr(x, "dtype", "float32")
                zero_const = create_const_node(root.name + "_zero", value=0, dtype=dtype, shape=s_res)
                return RewriteResult(new_nodes=[zero_const], node_mapping={root.name: zero_const.name})
        return None

# Rule: Mul(x, x) -> Square(x)
@PassRegistry.register("arithmetic_simplify_mul_self", opt_level=1, priority=7)
class MulSelfToSquarePass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op(
            "Mul",
            Any(alias="x"),
            Any(alias="y"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="MulSelfToSquare")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        root = match.matched_nodes["root"]

        if x.name == y.name:
            square_node = create_node("Square", root.name + "_sq", inputs=[x.name])
            return RewriteResult(new_nodes=[square_node], node_mapping={root.name: square_node.name})
        return None

# Rule: Div(x, 1) -> x
@PassRegistry.register("arithmetic_simplify_div_one", opt_level=1, priority=7)
class DivOnePass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op(
            "Div",
            Any(alias="x"),
            Op("Const", alias="one"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="DivOne")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        one = match.matched_nodes["one"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, one, 1):
            s_x = get_node_shape(optimizer, x)
            s_one = get_node_shape(optimizer, one)
            s_res = get_broadcast_shape(s_x, s_one)
            if is_shape_preserving(s_res, s_x):
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: Div(x, x) -> 1
@PassRegistry.register("arithmetic_simplify_div_self", opt_level=1, priority=7)
class DivSelfPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op(
            "Div",
            Any(alias="x"),
            Any(alias="y"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="DivSelf")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        root = match.matched_nodes["root"]

        if x.name == y.name:
            shape = get_node_shape(optimizer, x)
            if shape is not None:
                dtype = optimizer.get_node_attr(x, "dtype", "float32")
                one_const = create_const_node(root.name + "_one", value=1, dtype=dtype, shape=shape)
                return RewriteResult(new_nodes=[one_const], node_mapping={root.name: one_const.name})
        return None

# Rule: Pow(x, 1) -> x
@PassRegistry.register("arithmetic_simplify_pow_one", opt_level=1, priority=7)
class PowOnePass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op(
            "Pow",
            Any(alias="x"),
            Op("Const", alias="one"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="PowOne")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        one = match.matched_nodes["one"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, one, 1):
            s_x = get_node_shape(optimizer, x)
            s_one = get_node_shape(optimizer, one)
            s_res = get_broadcast_shape(s_x, s_one)
            if is_shape_preserving(s_res, s_x):
                return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: Pow(x, 2) -> Square(x)
@PassRegistry.register("arithmetic_simplify_pow_two", opt_level=1, priority=7)
class PowTwoToSquarePass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op(
            "Pow",
            Any(alias="x"),
            Op("Const", alias="two"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="PowTwoToSquare")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        two = match.matched_nodes["two"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, two, 2):
            s_x = get_node_shape(optimizer, x)
            s_two = get_node_shape(optimizer, two)
            s_res = get_broadcast_shape(s_x, s_two)
            if is_shape_preserving(s_res, s_x):
                square_node = create_node("Square", root.name + "_sq", inputs=[x.name])
                return RewriteResult(new_nodes=[square_node], node_mapping={root.name: square_node.name})
        return None

# Rule: Square(Sqrt(x)) -> x
@PassRegistry.register("arithmetic_simplify_square_sqrt", opt_level=1, priority=7)
class SquareSqrtPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("Square", Op("Sqrt", Any(alias="x"), alias="sqrt"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="SquareSqrt")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        root = match.matched_nodes["root"]
        return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})

# Rule: Sqrt(Square(x)) -> Abs(x)
@PassRegistry.register("arithmetic_simplify_sqrt_square", opt_level=1, priority=7)
class SqrtSquareToAbsPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("Sqrt", Op("Square", Any(alias="x"), alias="square"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="SqrtSquareToAbs")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        root = match.matched_nodes["root"]
        abs_node = create_node("Abs", root.name + "_abs", inputs=[x.name])
        return RewriteResult(new_nodes=[abs_node], node_mapping={root.name: abs_node.name})

# Rule: Mul(x, -1) -> Neg(x)
@PassRegistry.register("arithmetic_simplify_mul_neg_one", opt_level=1, priority=7)
class MulNegOneToNegPass(PatternRewritePass):
    def __init__(self):
        self.pattern = CommutativeOp(
            "Mul",
            Any(alias="x"),
            Op("Const", alias="neg_one"),
            alias="root"
        )
        super().__init__(self.pattern, self._rewrite, name="MulNegOneToNeg")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        neg_one = match.matched_nodes["neg_one"]
        root = match.matched_nodes["root"]

        if is_const_with_value(optimizer, neg_one, -1):
            s_x = get_node_shape(optimizer, x)
            s_neg_one = get_node_shape(optimizer, neg_one)
            s_res = get_broadcast_shape(s_x, s_neg_one)
            if is_shape_preserving(s_res, s_x):
                neg_node = create_node("Neg", root.name + "_neg", inputs=[x.name])
                return RewriteResult(new_nodes=[neg_node], node_mapping={root.name: neg_node.name})
        return None
