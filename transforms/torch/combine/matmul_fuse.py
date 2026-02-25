"""
PyTorch FX Matmul Fusion Pass
==============================
Fuses a chain of two matrix multiplications where the **right-hand weights**
are compile-time constants (module `get_attr` tensors or literal tensors):

    matmul(matmul(x, A), B)  →  matmul(x, matmul(A, B))

Because ``matmul(A, B)`` is a product of two constants, it is folded to a
single constant tensor, eliminating one multiply operation at runtime.

Conditions for fusion
---------------------
1. The outer matmul's *right* argument (B) must be a constant `get_attr` node.
2. The inner matmul's *right* argument (A) must also be a constant `get_attr`
   node (the left side is the variable input ``x``).
3. The inner matmul result must be used *only* by the outer matmul (no other
   consumers), so it is safe to remove.
4. A and B must be 2-D tensors with compatible shapes for ``A @ B``.

When fused
----------
1. Pre-compute ``AB = A @ B`` using concrete tensor values.
2. Store ``AB`` as a new module attribute.
3. Insert a ``get_attr`` node for ``AB``.
4. Replace ``matmul(matmul(x, A), B)`` with ``matmul(x, AB)``.
5. Dead-code-eliminate the now-unused intermediate nodes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.fx as fx

from graph_optimizer.core.base_pass import BaseOptimizationPass
from graph_optimizer.core.passes import PassRegistry


# Recognized matmul targets.
# NOTE: torch.bmm operates on 3-D tensors; _get_constant_tensor only accepts
# 2-D weights, so including bmm would silently skip all valid bmm chains.
# If batch-matmul fusion is ever needed, add an explicit 3-D path.
_MATMUL_TARGETS = frozenset(
    [
        torch.matmul,
        torch.mm,
        torch.Tensor.matmul,
        torch.Tensor.mm,
    ]
)


def _is_matmul(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in _MATMUL_TARGETS


def _get_constant_tensor(
    node: fx.Node, graph_module: fx.GraphModule
) -> Optional[torch.Tensor]:
    """
    If *node* is a `get_attr` whose attribute is a 2-D Tensor, return it.
    Otherwise return None.
    """
    if node.op != "get_attr":
        return None
    try:
        obj = graph_module
        for part in str(node.target).split("."):
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor) and obj.dim() == 2:
            return obj
    except AttributeError:
        pass
    return None


@PassRegistry.register("torch_matmul_fuse", opt_level=2, priority=25)
class MatmulFusePass(BaseOptimizationPass):
    """
    Fuses ``matmul(matmul(x, A), B)`` into ``matmul(x, A @ B)``
    when A and B are constant weight tensors.
    """

    @property
    def name(self) -> str:
        return self._name

    def __init__(self):
        self._name = "matmul_fuse"
        self._counter = 0

    def apply(self, graph_module: fx.GraphModule) -> bool:
        """
        Scan the graph for fusable matmul chains.

        Returns:
            True if at least one fusion was performed.
        """
        self._counter = 0
        changed = False

        for node in list(graph_module.graph.nodes):
            if not _is_matmul(node):
                continue
            if self._try_fuse(node, graph_module):
                changed = True

        if changed:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return changed

    # ------------------------------------------------------------------
    # Fusion logic
    # ------------------------------------------------------------------

    def _try_fuse(self, outer: fx.Node, gm: fx.GraphModule) -> bool:
        """
        Attempt to fuse:  outer = matmul(inner, B)
                          inner = matmul(x, A)
        """
        if len(outer.args) < 2:
            return False

        inner_node, b_node = outer.args[0], outer.args[1]

        # B must be a constant weight
        if not isinstance(b_node, fx.Node):
            return False
        B = _get_constant_tensor(b_node, gm)
        if B is None:
            return False

        # The left argument must be an inner matmul
        if not isinstance(inner_node, fx.Node) or not _is_matmul(inner_node):
            return False

        # Inner matmul must have exactly ONE user (the outer matmul)
        if len(inner_node.users) != 1:
            return False

        if len(inner_node.args) < 2:
            return False

        x_node, a_node = inner_node.args[0], inner_node.args[1]

        # A must also be a constant weight
        if not isinstance(a_node, fx.Node):
            return False
        A = _get_constant_tensor(a_node, gm)
        if A is None:
            return False

        # Shape compatibility: A is (m, k), B is (k, n) → AB is (m, n)
        if A.dim() != 2 or B.dim() != 2 or A.shape[1] != B.shape[0]:
            return False

        # Pre-compute A @ B and detach so the fused weight does NOT carry
        # a grad_fn linked to A/B (which may be nn.Parameters).  Without
        # .detach() the buffer would be part of the autograd graph and
        # produce wrong gradients during any post-optimization fine-tuning.
        with torch.no_grad():
            AB = (A @ B).detach()

        # Store as a module buffer
        attr_name = f"_fused_weight_{self._counter}"
        self._counter += 1
        gm.register_buffer(attr_name, AB)

        # Insert get_attr + new matmul
        with gm.graph.inserting_before(outer):
            ab_node = gm.graph.get_attr(attr_name)
            fused = gm.graph.create_node(
                "call_function",
                torch.matmul,
                args=(x_node, ab_node),
            )

        outer.replace_all_uses_with(fused)
        return True
