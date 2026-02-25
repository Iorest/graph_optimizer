"""
PyTorch FX Constant Folding Pass
=================================
Evaluates subgraphs where all inputs are compile-time constants (Python scalars
or zero-argument `get_attr` tensors) and replaces them with a single literal node.

Strategy
--------
1. Walk the FX graph in topological order (it already is).
2. For each `call_function` node, check if every argument is a "known constant"
   — either a Python scalar literal, or a predecessor node whose value we have
   already folded.
3. If all args are constants, evaluate the operation with the concrete values
   and replace the node with a new `get_attr` node that stores the folded tensor.
4. Repeat until convergence (the pass is called iteratively by TorchOptimizer).

Supported ops (binary): add, sub, mul, truediv, floordiv, pow, max, min
Supported ops (unary) : neg, abs, exp, log, sqrt, relu, sigmoid
"""

from __future__ import annotations

import operator
from typing import Any, Dict, Optional

import torch
import torch.fx as fx

from graph_optimizer.core.base_pass import BaseOptimizationPass
from graph_optimizer.core.passes import PassRegistry


# ---------------------------------------------------------------------------
# Op dispatch table  (target -> callable on raw Python/tensor values)
# ---------------------------------------------------------------------------

_BINARY_OPS: Dict[Any, Any] = {
    operator.add: operator.add,
    torch.add: operator.add,
    operator.sub: operator.sub,
    torch.sub: operator.sub,
    operator.mul: operator.mul,
    torch.mul: operator.mul,
    operator.truediv: operator.truediv,
    torch.div: operator.truediv,
    operator.floordiv: operator.floordiv,
    operator.pow: operator.pow,
    torch.pow: operator.pow,
}

_UNARY_OPS: Dict[Any, Any] = {
    operator.neg: operator.neg,
    torch.neg: operator.neg,
    operator.abs: operator.abs,
    torch.abs: operator.abs,
    torch.exp: torch.exp,
    torch.log: torch.log,
    torch.sqrt: torch.sqrt,
    torch.relu: torch.relu,
    torch.sigmoid: torch.sigmoid,
}


@PassRegistry.register("torch_constant_fold", opt_level=1, priority=5)
class TorchConstantFoldPass(BaseOptimizationPass):
    """
    Folds constant subgraphs in a PyTorch FX GraphModule.
    """

    @property
    def name(self) -> str:
        return self._name

    def __init__(self):
        self._name = "constant_fold"
        self._attr_counter = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def apply(self, graph_module: fx.GraphModule) -> bool:
        """
        Walk the graph once, fold any fully-constant subgraphs.

        Returns:
            True if at least one node was folded.
        """
        self._attr_counter = 0
        known: Dict[fx.Node, Any] = {}  # node -> concrete Python/tensor value

        # Seed: collect get_attr nodes that refer to scalar/tensor constants
        for node in graph_module.graph.nodes:
            if node.op == "get_attr":
                val = self._get_attr_value(graph_module, node.target)
                if val is not None:
                    known[node] = val

        changed = False
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue
            folded = self._try_fold(node, known)
            if folded is None:
                continue
            # Store as a module attribute and replace the node.
            # IMPORTANT: Tensor values must go through register_buffer so they
            # are included in state_dict() and survive torch.save/torch.load.
            # Plain Python scalars (int/float/bool) can use setattr directly.
            attr_name = f"_const_fold_{self._attr_counter}"
            self._attr_counter += 1
            if isinstance(folded, torch.Tensor):
                graph_module.register_buffer(attr_name, folded.detach())
            else:
                setattr(graph_module, attr_name, folded)
            with graph_module.graph.inserting_before(node):
                new_node = graph_module.graph.get_attr(attr_name)
            node.replace_all_uses_with(new_node)
            known[new_node] = folded
            changed = True

        if changed:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return changed

    # ------------------------------------------------------------------
    # Constant folding logic
    # ------------------------------------------------------------------

    def _try_fold(self, node: fx.Node, known: Dict[fx.Node, Any]) -> Optional[Any]:
        """
        Try to evaluate `node` given the set of already-known constant values.
        Returns the folded value, or None if folding is not possible.
        """
        t = node.target

        if t in _BINARY_OPS:
            if len(node.args) < 2:
                return None
            a = self._resolve(node.args[0], known)
            b = self._resolve(node.args[1], known)
            if a is None or b is None:
                return None
            try:
                return _BINARY_OPS[t](a, b)
            except Exception:
                return None

        if t in _UNARY_OPS:
            if not node.args:
                return None
            a = self._resolve(node.args[0], known)
            if a is None:
                return None
            try:
                return _UNARY_OPS[t](a)
            except Exception:
                return None

        return None

    @staticmethod
    def _resolve(arg: Any, known: Dict[fx.Node, Any]) -> Optional[Any]:
        """Resolve an argument to a concrete Python/tensor value if possible."""
        if isinstance(arg, (int, float, bool)):
            return arg
        if isinstance(arg, fx.Node) and arg in known:
            return known[arg]
        return None

    @staticmethod
    def _get_attr_value(graph_module: fx.GraphModule, target: str) -> Optional[Any]:
        """
        Retrieve a module attribute by dotted path.
        Returns the value if it is a float/int scalar or a 0-d / small tensor.
        """
        try:
            obj = graph_module
            for part in target.split("."):
                obj = getattr(obj, part)
            # Accept plain Python scalars or tensors (any size — folding may
            # produce large tensors, the caller decides what to do with them)
            if isinstance(obj, (int, float, bool, torch.Tensor)):
                return obj
        except AttributeError:
            pass
        return None
