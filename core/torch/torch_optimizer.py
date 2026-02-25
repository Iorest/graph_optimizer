"""
Core optimizer for PyTorch FX GraphModules.

Mirrors TFGraphOptimizer in:
- Pass lifecycle:  begin_pass → iterate → end_pass (with convergence loop)
- Rollback:        graph state backed up before each pass; restored on exception
- Debug dumps:     serialise FX IR to <debug_dir>/<step>_<pass>.txt after each pass
- Structured log:  uses the shared "GraphOptimizer" logger (same as TF side)
"""

from __future__ import annotations

import copy
import logging
import os
import traceback
from typing import Any, List, Optional

from ..base_optimizer import BaseOptimizer
from ..passes import OptimizationContext, PassRegistry

logger = logging.getLogger("GraphOptimizer.torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_torch_passes_registered() -> None:
    """Import all Torch transform sub-packages to trigger @PassRegistry.register."""
    import graph_optimizer.transforms.torch.scalar  # noqa: F401
    import graph_optimizer.transforms.torch.combine  # noqa: F401


def _default_pass_names(opt_level: int) -> List[str]:
    """Return sorted Torch pass names active at the given opt_level."""
    _ensure_torch_passes_registered()
    all_names = PassRegistry.get_passes_by_level(opt_level)
    return [n for n in all_names if n.startswith("torch_")]


def _snapshot(graph_module: Any) -> dict:
    """
    Create a snapshot of the FX graph and module buffer names for rollback.

    We deep-copy the graph structure and record the set of buffer names that
    existed *before* the pass ran.  On rollback, any buffers added by the
    failed pass are deleted so the module is left in a clean state.

    We intentionally do NOT deep-copy nn.Parameters — passes are not permitted
    to mutate existing parameter values, only to add new buffers.
    """
    return {
        "graph": copy.deepcopy(graph_module.graph),
        "buffer_names": set(graph_module._buffers.keys()),
    }


def _restore(graph_module: Any, snapshot: dict) -> None:
    """Restore the FX graph and remove any buffers added by the failed pass."""
    graph_module.graph = snapshot["graph"]

    # Remove buffers that the failed pass registered
    current_buffers = set(graph_module._buffers.keys())
    added_buffers = current_buffers - snapshot["buffer_names"]
    for buf_name in added_buffers:
        graph_module._buffers.pop(buf_name, None)
        # Also remove the plain attribute if the pass used setattr
        if hasattr(graph_module, buf_name):
            delattr(graph_module, buf_name)

    graph_module.recompile()


def _save_debug_graph(graph_module: Any, debug_dir: str, filename: str) -> None:
    """Dump the FX readable IR to <debug_dir>/<filename>.txt."""
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"{filename}.txt")
    try:
        with open(path, "w") as f:
            graph_module.print_readable(print_output=False, file=f)
        logger.debug(f"Saved debug graph to {path}")
    except Exception as exc:
        logger.debug(f"Could not save debug graph: {exc}")


# ---------------------------------------------------------------------------
# TorchOptimizer
# ---------------------------------------------------------------------------


class TorchOptimizer(BaseOptimizer):
    """
    Core optimizer for PyTorch FX GraphModules.

    Usage::

        optimizer = TorchOptimizer(gm)                # default opt_level=1
        optimizer = TorchOptimizer(gm, opt_level=2)   # richer pass set
        optimizer = TorchOptimizer(gm, passes=["torch_cse"])  # explicit
    """

    def __init__(
        self,
        graph_module: Any,
        passes: Optional[List[str]] = None,
        opt_level: int = 1,
    ):
        """
        Args:
            graph_module: A ``torch.fx.GraphModule`` instance.
            passes: Explicit list of registered pass names.
                    If None, uses all passes at ``opt_level``.
            opt_level: Optimization aggressiveness (1 = safe, 2 = aggressive).
        """
        _ensure_torch_passes_registered()

        if passes is not None:
            resolved = PassRegistry.sort_passes(passes)
        else:
            resolved = _default_pass_names(opt_level)

        if not resolved:
            logger.warning(
                "TorchOptimizer: no passes selected "
                f"(opt_level={opt_level}, explicit={passes})"
            )

        self.passes = [PassRegistry.get_pass(name) for name in resolved]
        self.graph_module = graph_module
        self.opt_level = opt_level

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self.graph_module.graph.nodes)

    def optimize(
        self,
        max_iterations: int = 5,
        context: Optional[OptimizationContext] = None,
        debug_dir: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Run the configured optimization passes on the FX graph until convergence.

        Lifecycle (matches TFGraphOptimizer):
        ::

            for sweep in range(max_iterations):
                for pass in passes:
                    snapshot = backup()
                    context.begin_pass(pass.name)
                    context.begin_iteration()
                    try:
                        changed = pass.apply(gm)
                    except:
                        restore(snapshot)
                        context.end_pass(failed=True)
                        continue
                    context.end_iteration(...)
                    context.end_pass(...)
                    if debug_dir: dump_graph()
                if not changed_in_sweep: break

        Args:
            max_iterations: Safety cap on convergence sweeps.
            context: Optional ``OptimizationContext`` for telemetry/logging.
            debug_dir: If set, dump the FX IR after every pass to this directory.

        Returns:
            The optimised ``torch.fx.GraphModule``.
        """
        logger.info(
            f"TorchOptimizer: starting optimization "
            f"({len(self.passes)} passes, opt_level={self.opt_level}, "
            f"max_iter={max_iterations})"
        )

        step = 0  # global step counter for debug filenames

        for iteration in range(max_iterations):
            changed_in_sweep = False

            for fx_pass in self.passes:
                pass_name = fx_pass.name
                step += 1

                # Snapshot graph state before this pass (for rollback)
                snapshot = _snapshot(self.graph_module)

                # --- context lifecycle: begin pass ---
                if context:
                    context.begin_pass(pass_name)
                    nodes_before = self.node_count

                logger.debug(
                    f"[{pass_name}] sweep {iteration + 1}, nodes={self.node_count}"
                )

                # --- apply pass (one iteration) ---
                if context:
                    context.begin_iteration()

                try:
                    pass_changed = fx_pass.apply(self.graph_module)
                except Exception as exc:
                    logger.error(f"[{pass_name}] raised an exception: {exc}")
                    logger.debug(f"[{pass_name}] traceback:\n{traceback.format_exc()}")
                    logger.warning(f"[{pass_name}] rolling back graph state...")
                    _restore(self.graph_module, snapshot)

                    if context:
                        nodes_after = self.node_count
                        context.end_iteration(0, nodes_before, nodes_after)
                        context.end_pass(nodes_before, nodes_after, failed=True)
                    continue

                # --- report ---
                if context:
                    nodes_after = self.node_count
                    changes = 1 if pass_changed else 0
                    context.end_iteration(changes, nodes_before, nodes_after)
                    context.end_pass(nodes_before, nodes_after, failed=False)

                # --- debug dump ---
                if debug_dir and pass_changed:
                    _save_debug_graph(
                        self.graph_module,
                        debug_dir,
                        f"{step:02d}_{pass_name}",
                    )

                if pass_changed:
                    changed_in_sweep = True

            if not changed_in_sweep:
                logger.info(
                    f"TorchOptimizer: converged after {iteration + 1} sweep(s)."
                )
                break
        else:
            logger.warning(
                f"TorchOptimizer: reached max_iterations={max_iterations} "
                "without convergence."
            )

        # Final safety recompile
        self.graph_module.recompile()
        return self.graph_module
