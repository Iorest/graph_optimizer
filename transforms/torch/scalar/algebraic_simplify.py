"""
PyTorch FX Algebraic Simplify Pass
==================================
Performs algebraic simplifications on FX IR nodes:

Arithmetic identity rules:
  Add(x, 0) / Add(0, x) -> x
  Sub(x, 0)             -> x
  Sub(x, x)             -> 0
  Mul(x, 1) / Mul(1, x) -> x
  Mul(x, 0) / Mul(0, x) -> 0
  Div(x, 1)             -> x
  Neg(Neg(x))           -> x
  Pow(x, 1)             -> x
  Pow(x, 0)             -> 1

Reshape identity rules:
  reshape(reshape(x, s1), s2) -> reshape(x, s2)   [chain fusion]
  reshape(x, x.shape)         -> x                 [identity elimination]
"""

import operator
import pytest  # noqa: F401 importorskip pattern
import torch
import torch.fx as fx
from graph_optimizer.core.torch.torch_passes import TorchBasePass
from graph_optimizer.core.passes import PassRegistry


@PassRegistry.register("torch_algebraic_simplify", backend='torch', opt_level=1, priority=7)
class TorchAlgebraicSimplifyPass(TorchBasePass):
    """
    PyTorch FX pass for simplifying basic algebraic operations.
    """

    def __init__(self):
        super().__init__(name="algebraic_simplify")

    def transform(self, graph_module: fx.GraphModule) -> bool:
        """
        Applies algebraic simplifications.

        Returns:
            True if the graph was modified.
        """
        changed = False
        # Collect nodes first so we can safely iterate while mutating
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            result = self._try_simplify(node, graph_module.graph)
            if result:
                changed = True

        if changed:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return changed

    # ------------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------------

    def _try_simplify(self, node: fx.Node, graph: fx.Graph) -> bool:
        t = node.target

        if t in (operator.add, torch.add):
            return self._simplify_add(node, graph)
        if t in (operator.sub, torch.sub):
            return self._simplify_sub(node, graph)
        if t in (operator.mul, torch.mul):
            return self._simplify_mul(node, graph)
        if t in (operator.truediv, torch.div):
            return self._simplify_div(node, graph)
        if t in (operator.neg, torch.neg):
            return self._simplify_neg(node, graph)
        if t in (operator.pow, torch.pow):
            return self._simplify_pow(node, graph)
        if t in (torch.reshape, torch.Tensor.reshape, torch.Tensor.view):
            return self._simplify_reshape(node, graph)
        return False

    # ------------------------------------------------------------------
    # Per-op simplification rules
    # ------------------------------------------------------------------

    def _simplify_add(self, node: fx.Node, graph: fx.Graph) -> bool:
        x, y = node.args[0], node.args[1]
        # Add(x, 0) -> x
        if self._is_scalar_const(y, 0):
            node.replace_all_uses_with(x)
            return True
        # Add(0, x) -> x
        if self._is_scalar_const(x, 0):
            node.replace_all_uses_with(y)
            return True
        return False

    def _simplify_sub(self, node: fx.Node, graph: fx.Graph) -> bool:
        x, y = node.args[0], node.args[1]
        # Sub(x, 0) -> x
        if self._is_scalar_const(y, 0):
            node.replace_all_uses_with(x)
            return True
        # Sub(x, x) -> 0  (use identity, not __eq__, for fx.Node comparison)
        if x is y:
            with graph.inserting_before(node):
                zero = self._make_zero(graph, x)
            node.replace_all_uses_with(zero)
            return True
        return False

    def _simplify_mul(self, node: fx.Node, graph: fx.Graph) -> bool:
        x, y = node.args[0], node.args[1]
        # Mul(x, 1) -> x
        if self._is_scalar_const(y, 1):
            node.replace_all_uses_with(x)
            return True
        # Mul(1, x) -> x
        if self._is_scalar_const(x, 1):
            node.replace_all_uses_with(y)
            return True
        # Mul(x, 0) -> 0  (x is the Tensor-shaped operand)
        if self._is_scalar_const(y, 0):
            with graph.inserting_before(node):
                zero = self._make_zero(graph, x)
            node.replace_all_uses_with(zero)
            return True
        # Mul(0, x) -> 0  (y is the Tensor-shaped operand)
        if self._is_scalar_const(x, 0):
            with graph.inserting_before(node):
                zero = self._make_zero(graph, y)
            node.replace_all_uses_with(zero)
            return True
        return False

    def _simplify_div(self, node: fx.Node, graph: fx.Graph) -> bool:
        x, y = node.args[0], node.args[1]
        # Div(x, 1) -> x
        if self._is_scalar_const(y, 1):
            node.replace_all_uses_with(x)
            return True
        return False

    def _simplify_neg(self, node: fx.Node, graph: fx.Graph) -> bool:
        x = node.args[0]
        # Neg(Neg(x)) -> x
        if (
            isinstance(x, fx.Node)
            and x.op == "call_function"
            and x.target in (operator.neg, torch.neg)
        ):
            inner = x.args[0]
            node.replace_all_uses_with(inner)
            return True
        return False

    def _simplify_pow(self, node: fx.Node, graph: fx.Graph) -> bool:
        x, y = node.args[0], node.args[1]
        # Pow(x, 1) -> x
        if self._is_scalar_const(y, 1):
            node.replace_all_uses_with(x)
            return True
        # Pow(x, 0) -> ones_like(x)
        if self._is_scalar_const(y, 0):
            with graph.inserting_before(node):
                ones = graph.create_node("call_function", torch.ones_like, args=(x,))
            node.replace_all_uses_with(ones)
            return True
        return False

    # ------------------------------------------------------------------
    # Reshape / view identity rules
    # ------------------------------------------------------------------

    _RESHAPE_TARGETS = frozenset(
        [torch.reshape, torch.Tensor.reshape, torch.Tensor.view]
    )

    def _simplify_reshape(self, node: fx.Node, graph: fx.Graph) -> bool:
        if not node.args:
            return False
        parent = node.args[0]
        if not isinstance(parent, fx.Node):
            return False

        # Rule 1: chain fusion — reshape(reshape(x, s1), s2) → reshape(x, s2)
        if (
            parent.op == "call_function"
            and parent.target in self._RESHAPE_TARGETS
            and len(parent.users) == 1  # parent only used by this node — safe to remove
            and parent.args
        ):
            final_shape = self._extract_static_shape(node)
            grandparent = parent.args[0]
            if final_shape is not None and isinstance(grandparent, fx.Node):
                with graph.inserting_before(node):
                    fused = graph.create_node(
                        "call_function",
                        torch.reshape,
                        args=(grandparent, final_shape),
                    )
                node.replace_all_uses_with(fused)
                return True

        # Rule 2: identity elimination — reshape(x, x.shape) → x
        target_shape = self._extract_static_shape(node)
        input_shape = self._static_shape_from_meta(parent)
        if (
            target_shape is not None
            and input_shape is not None
            and -1 not in target_shape
            and tuple(target_shape) == tuple(input_shape)
        ):
            node.replace_all_uses_with(parent)
            return True

        return False

    @staticmethod
    def _extract_static_shape(node: fx.Node):
        """Extract a static tuple-of-ints shape from a reshape node's arguments."""
        if node.target == torch.reshape and len(node.args) >= 2:
            s = node.args[1]
        elif node.target in (torch.Tensor.reshape, torch.Tensor.view):
            rest = node.args[1:]
            s = (
                rest[0]
                if len(rest) == 1 and isinstance(rest[0], (tuple, list))
                else rest
            )
        else:
            return None
        if isinstance(s, (tuple, list)) and all(isinstance(d, int) for d in s):
            return tuple(s)
        return None

    @staticmethod
    def _static_shape_from_meta(node: fx.Node):
        """Try to read the static shape from node.meta['val'] (set by shape propagation)."""
        val = node.meta.get("val")
        if val is not None and hasattr(val, "shape"):
            try:
                return tuple(val.shape)
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_scalar_const(value, target: float) -> bool:
        """Return True if value is a Python scalar literal equal to target."""
        if isinstance(value, (int, float, bool)):
            return float(value) == float(target)
        return False

    @staticmethod
    def _make_zero(graph: fx.Graph, operand) -> fx.Node:
        """
        Create a zero node appropriate for *operand*.

        - If operand is an fx.Node (Tensor), emit ``zeros_like(operand)``
          so the result preserves shape/dtype/device.
        - If operand is a Python scalar (int/float/bool), ``zeros_like``
          would crash (requires a Tensor).  We emit ``torch.tensor(0.0)``
          as a safe fallback; constant-fold will simplify it further.
        """
        if isinstance(operand, fx.Node):
            return graph.create_node("call_function", torch.zeros_like, args=(operand,))
        # Scalar fallback — emit a 0.0 tensor constant
        return graph.create_node("call_function", torch.tensor, args=(0.0,))
