import time
from typing import Dict, Set, Optional
from ..utils.logger import logger as logging


class OptimizationContext:
    """
    Unified context for graph optimization operations.

    Manages:
    - Protected nodes that should not be pruned or modified
    - Optimization statistics collection
    - Iteration tracking for convergence detection
    - Unified logging with pass prefix

    This context is passed through the optimization pipeline to ensure
    consistent behavior across all passes and operations.
    """

    def __init__(
        self,
        protected_nodes: Optional[Set[str]] = None,
        auto_cleanup: bool = True,
        max_iterations: int = 100,
        debug_dir: Optional[str] = None,
    ):
        """
        Initialize optimization context.

        Args:
            protected_nodes: Set of node names that should not be pruned/modified
            auto_cleanup: Whether to automatically prune dead nodes
            max_iterations: Maximum iterations for convergence (safety limit)
            debug_dir: Optional directory to save intermediate graphs
        """
        self._protected_nodes: Set[str] = set(protected_nodes or [])
        self.auto_cleanup = auto_cleanup
        self.max_iterations = max_iterations
        self.debug_dir = debug_dir

        # Current pass info
        self._current_pass: Optional[str] = None
        self._current_iteration: int = 0

        # Statistics (embedded, not a separate class)
        self._pass_stats: Dict[str, dict] = {}
        self._current_pass_start: Optional[float] = None

    # =========================================================================
    # Protected Nodes Management
    # =========================================================================

    @property
    def protected_nodes(self) -> Set[str]:
        """Get the set of protected nodes."""
        return self._protected_nodes

    def add_protected(self, *node_names: str):
        """Add nodes to the protected set."""
        for name in node_names:
            if name:
                self._protected_nodes.add(name)

    def remove_protected(self, *node_names: str):
        """Remove nodes from the protected set."""
        for name in node_names:
            self._protected_nodes.discard(name)

    def is_protected(self, node_name: str) -> bool:
        """Check if a node is protected."""
        return node_name in self._protected_nodes

    def clear_protected(self):
        """Clear all protected nodes."""
        self._protected_nodes.clear()

    # =========================================================================
    # Pass & Iteration Management
    # =========================================================================

    def begin_pass(self, pass_name: str):
        """Mark the beginning of a pass."""
        self._current_pass = pass_name
        self._current_iteration = 0
        self._current_pass_start = time.time()
        if pass_name not in self._pass_stats:
            self._pass_stats[pass_name] = {
                "iterations": [],
                "total_changes": 0,
                "duration": 0.0,
                "nodes_before": 0,
                "nodes_after": 0,
            }
        logging.info(f"[{pass_name}] Starting...")

    def begin_iteration(self) -> int:
        """Mark the beginning of an iteration, returns iteration number (1-based)."""
        self._current_iteration += 1
        return self._current_iteration

    def end_iteration(self, changes: int, nodes_before: int, nodes_after: int):
        """Mark the end of an iteration with statistics."""
        pass_name = self._current_pass
        if pass_name and pass_name in self._pass_stats:
            self._pass_stats[pass_name]["iterations"].append(
                {
                    "iteration": self._current_iteration,
                    "changes": changes,
                    "nodes_before": nodes_before,
                    "nodes_after": nodes_after,
                }
            )
            self._pass_stats[pass_name]["total_changes"] += changes
            if self._current_iteration == 1:
                self._pass_stats[pass_name]["nodes_before"] = nodes_before
            self._pass_stats[pass_name]["nodes_after"] = nodes_after

        if changes > 0:
            logging.info(
                f"[{pass_name}] Iteration {self._current_iteration}: "
                f"{changes} changes, {nodes_before} -> {nodes_after} nodes"
            )
        else:
            logging.debug(
                f"[{pass_name}] Iteration {self._current_iteration}: converged"
            )

    def end_pass(self, nodes_before: int, nodes_after: int, failed: bool = False):
        """Mark the end of a pass with final statistics."""
        pass_name = self._current_pass
        duration = time.time() - (self._current_pass_start or time.time())

        if pass_name and pass_name in self._pass_stats:
            self._pass_stats[pass_name]["duration"] = duration
            if failed:
                self._pass_stats[pass_name]["failed"] = True

        total_changes = self._pass_stats.get(pass_name, {}).get("total_changes", 0)
        iterations = self._current_iteration

        if failed:
            logging.error(f"[{pass_name}] Failed after {duration:.3f}s")
        else:
            logging.info(
                f"[{pass_name}] Completed in {duration:.3f}s "
                f"({iterations} iteration{'s' if iterations != 1 else ''}). "
                f"Nodes: {nodes_before} -> {nodes_after} ({total_changes} changes)"
            )
        self._current_pass = None
        self._current_iteration = 0
        self._current_pass_start = None

    def warn_max_iterations(self):
        """Log warning when max iterations reached."""
        logging.warning(
            f"[{self._current_pass}] Reached max iterations ({self.max_iterations})"
        )

    @property
    def current_pass(self) -> Optional[str]:
        """Get current pass name."""
        return self._current_pass

    @property
    def current_iteration(self) -> int:
        """Get current iteration number."""
        return self._current_iteration

    # =========================================================================
    # Logging Helpers
    # =========================================================================

    def log_info(self, message: str):
        """Log info with current pass prefix."""
        prefix = f"[{self._current_pass}] " if self._current_pass else ""
        logging.info(f"{prefix}{message}")

    def log_debug(self, message: str):
        """Log debug with current pass prefix."""
        prefix = f"[{self._current_pass}] " if self._current_pass else ""
        logging.debug(f"{prefix}{message}")

    def log_warning(self, message: str):
        """Log warning with current pass prefix."""
        prefix = f"[{self._current_pass}] " if self._current_pass else ""
        logging.warning(f"{prefix}{message}")

    # =========================================================================
    # Statistics Access
    # =========================================================================

    def get_pass_total_changes(self, pass_name: str) -> int:
        """Get total changes for a pass."""
        return self._pass_stats.get(pass_name, {}).get("total_changes", 0)

    def get_summary(self) -> str:
        """Get summary of all optimization passes."""
        lines = ["Optimization Summary:"]
        for name, stats in self._pass_stats.items():
            if stats.get("failed"):
                lines.append(f"  {name}: FAILED ({stats.get('duration', 0.0):.3f}s)")
            else:
                lines.append(
                    f"  {name}: {stats['nodes_before']} -> {stats['nodes_after']} nodes "
                    f"({stats['total_changes']} changes, {len(stats['iterations'])} iterations, "
                    f"{stats['duration']:.3f}s)"
                )
        return "\n".join(lines)


class PassRegistry:
    """Registry for managing optimization passes."""

    _registered_passes = {}
    _pass_metadata = {}

    @classmethod
    def register(cls, name, opt_level=1, priority=100):
        """Decorator to register a pass class with an optimization level and priority."""

        def decorator(pass_cls):
            cls._registered_passes[name] = pass_cls
            cls._pass_metadata[name] = {"opt_level": opt_level, "priority": priority}
            return pass_cls

        return decorator

    @classmethod
    def get_pass(cls, name, *args, **kwargs):
        """Creates an instance of the pass by its registered name."""
        if name not in cls._registered_passes:
            raise ValueError(f"Unknown pass: {name}")
        return cls._registered_passes[name](*args, **kwargs)

    @classmethod
    def get_priority(cls, name):
        """Returns the priority for a pass name."""
        meta = cls._pass_metadata.get(name)
        return meta.get("priority", 100) if meta else 100

    @classmethod
    def sort_passes(cls, pass_names):
        """Sorts a list of pass names based on their registered priority."""
        return sorted(pass_names, key=lambda name: (cls.get_priority(name), name))

    @classmethod
    def get_passes_by_level(cls, level):
        """Returns a list of pass names enabled at the given optimization level, sorted by priority."""
        candidates = [
            name
            for name, meta in cls._pass_metadata.items()
            if meta["opt_level"] <= level
        ]
        return cls.sort_passes(candidates)
