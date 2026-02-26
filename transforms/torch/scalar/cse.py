"""
PyTorch FX Common Subexpression Elimination (CSE) Pass
=======================================================
Identifies and deduplicates `call_function` nodes that compute the same value,
replacing all consumers of the duplicate with the canonical node.

`call_method` nodes are currently excluded: their target is a plain string
(e.g. ``"contiguous"``), which does not capture the receiver's device/dtype.
Two calls like ``cpu_tensor.contiguous()`` and ``cuda_tensor.contiguous()`` would
share the same signature and be incorrectly merged.

Algorithm
---------
1. Assign a *signature* to each node: (op, target, canonical_args...).
   - For a `call_function` node: (target, frozenset(kwargs), tuple(args))
     where each arg node is referenced by its own signature (recursively),
     so structurally identical sub-expressions get the same signature.
   - Literals (int, float, bool, str) are included as-is.
2. Walk the graph in topological order (FX graph is already topological).
3. Maintain a `seen: Dict[signature, fx.Node]` map.
4. When a node's signature is already in `seen`, replace all uses of the
   current node with the canonical node and schedule the current node for removal.
5. Call `eliminate_dead_code()` + `recompile()` once at the end.

Non-eliminable ops
------------------
Any op that has side-effects or is non-deterministic is excluded:
  - `call_module` (may modify module state)
  - `placeholder`, `output` (graph boundaries)
  - `get_attr` (module attributes may differ)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.fx as fx

from graph_optimizer.core.torch.torch_passes import TorchBasePass
from graph_optimizer.core.passes import PassRegistry


# Ops whose results are unconditionally excluded from deduplication.
_NON_PURE_TARGETS = frozenset(
    [
        # Random
        torch.rand,
        torch.randn,
        torch.randperm,
        torch.bernoulli,
        # In-place or side-effecting
        torch.Tensor.fill_,
        torch.Tensor.copy_,
        torch.Tensor.zero_,
    ]
)


# Type alias for signatures
_Sig = Tuple[Any, ...]


def _make_sig(node: fx.Node, sig_cache: Dict[fx.Node, _Sig]) -> Optional[_Sig]:
    """
    Compute a structural signature for *node*.

    Returns None if the node is not eligible for CSE.
    """
    # Only call_function is eligible; call_method is excluded because its
    # target is a plain string that doesn't capture receiver device/dtype —
    # merging two calls to e.g. `.contiguous()` on tensors of different
    # devices would silently produce wrong results.
    if node.op != "call_function":
        return None

    if node.target in _NON_PURE_TARGETS:
        return None

    # Resolve each positional arg to a signature or scalar literal
    resolved_args = []
    for a in node.args:
        if isinstance(a, fx.Node):
            a_sig = sig_cache.get(a)
            if a_sig is None:
                # Depends on a node we couldn't fingerprint (e.g. placeholder)
                return None
            resolved_args.append(a_sig)
        elif isinstance(
            a,
            (
                int,
                float,
                bool,
                str,
                type(None),
                torch.device,
                torch.dtype,
                torch.layout,
                torch.memory_format,
            ),
        ):
            resolved_args.append(a)
        else:
            # Unknown argument type — be conservative
            return None

    # Resolve kwargs
    resolved_kwargs: Dict[str, Any] = {}
    for k, v in node.kwargs.items():
        if isinstance(v, fx.Node):
            v_sig = sig_cache.get(v)
            if v_sig is None:
                return None
            resolved_kwargs[k] = v_sig
        elif isinstance(
            v,
            (
                int,
                float,
                bool,
                str,
                type(None),
                torch.device,
                torch.dtype,
                torch.layout,
                torch.memory_format,
            ),
        ):
            resolved_kwargs[k] = v
        else:
            return None

    return (
        node.op,
        node.target,
        tuple(resolved_args),
        tuple(sorted(resolved_kwargs.items())),
    )


@PassRegistry.register("torch_cse", backend="torch", opt_level=1, priority=20)
class TorchCSEPass(TorchBasePass):
    """
    Common Subexpression Elimination for PyTorch FX graphs.

    Eliminates duplicate `call_function` / `call_method` nodes that compute
    the same value and replaces all their consumers with the canonical node.
    """

    def __init__(self):
        super().__init__(name="cse")

    def transform(self, graph_module: fx.GraphModule) -> bool:
        """
        Run one pass of CSE on the graph.

        Returns:
            True if at least one redundant node was eliminated.
        """
        graph = graph_module.graph

        # sig_cache: fx.Node -> its structural signature (or None/missing if not eligible)
        sig_cache: Dict[fx.Node, _Sig] = {}
        # seen:     signature -> canonical node
        seen: Dict[_Sig, fx.Node] = {}

        to_eliminate: Dict[fx.Node, fx.Node] = {}  # duplicate -> canonical

        for node in graph.nodes:
            sig = _make_sig(node, sig_cache)

            if sig is not None:
                if sig in seen:
                    # This node is a duplicate — replace with canonical
                    to_eliminate[node] = seen[sig]
                else:
                    seen[sig] = node
                    sig_cache[node] = sig
            else:
                # Not eligible for CSE but still needs a "fingerprint" for downstream
                # nodes — use its own identity (object id) so it doesn't match others
                sig_cache[node] = (id(node),)

        if not to_eliminate:
            return False

        # Redirect uses
        for dup, canonical in to_eliminate.items():
            dup.replace_all_uses_with(canonical)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return True
