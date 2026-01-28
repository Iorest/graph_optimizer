
from __future__ import annotations

from __future__ import annotations

from graph_optimizer.core import (
    Any,
    Op,
    PassRegistry,
    PatternRewritePass,
    RewriteResult,
)
from graph_optimizer.utils.graph_utils import create_node

# ==============================================================================
# Miscellaneous Simplification Patterns
# ==============================================================================

# Rule: Neg(Neg(x)) -> x
@PassRegistry.register("other_simplify_double_neg", opt_level=1, priority=7)
class DoubleNegPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("Neg", Op("Neg", Any(alias="x"), alias="inner_neg"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="DoubleNeg")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        root = match.matched_nodes["root"]
        return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})

# Rule: Abs(Abs(x)) -> Abs(x)
@PassRegistry.register("other_simplify_double_abs", opt_level=1, priority=7)
class DoubleAbsPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("Abs", Op("Abs", Any(alias="x"), alias="inner_abs"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="DoubleAbs")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        root = match.matched_nodes["root"]
        # The result should be a new Abs node pointing to the original input x
        new_abs = create_node("Abs", root.name + "_abs", inputs=[x.name])
        return RewriteResult(new_nodes=[new_abs], node_mapping={root.name: new_abs.name})

# Rule: Select(cond, x, x) -> x
@PassRegistry.register("other_simplify_select_self", opt_level=1, priority=7)
class SelectSelfPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("Select", Any(alias="cond"), Any(alias="x"), Any(alias="y"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="SelectSelf")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        y = match.matched_nodes["y"]
        root = match.matched_nodes["root"]

        if x.name == y.name:
            return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
        return None

# Rule: Identity(x) -> x (Bypass)
# This is a very common and important simplification.
@PassRegistry.register("other_simplify_identity_bypass", opt_level=1, priority=1) # High priority
class IdentityBypassPass(PatternRewritePass):
    def __init__(self):
        self.pattern = Op("Identity", Any(alias="x"), alias="root")
        super().__init__(self.pattern, self._rewrite, name="IdentityBypass")

    def _rewrite(self, match, optimizer):
        x = match.matched_nodes["x"]
        root = match.matched_nodes["root"]

        # Safety checks from the original implementation
        if (
            hasattr(optimizer, "protected_nodes")
            and root.name in optimizer.protected_nodes
        ):
            return None
        if "ReadVariableOp" in root.name:
            return None
        if "_class" in root.attr:
            return None

        # Check if the input is another Identity node (handled by iterative nature of passes)
        # If we map Identity -> its input, and its input is also an Identity, the next iteration
        # of this pass will handle it. No need for special nested logic.

        return RewriteResult(new_nodes=[], node_mapping={root.name: x.name})
