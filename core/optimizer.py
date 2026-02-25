import tensorflow.compat.v1 as tf
import hashlib
import collections
from ..utils.logger import logger as logging, log_optimization
from .graph import GraphState
from .matcher import PatternMatcher
from .passes import OptimizationContext


class GraphOptimizer(GraphState):
    """
    Graph state container, query context, and pattern-based optimizer.

    Responsibilities:
    - Graph state management (via GraphState inheritance)
    - Pattern registration and matching (via PatternMatcher)
    - Driving optimization passes
    """

    def __init__(self, graph_def: tf.GraphDef):
        super().__init__(graph_def)
        self._matcher = PatternMatcher()

    # =========================================================================
    # Pattern Matching (delegates to PatternMatcher)
    # =========================================================================

    def add_transformation(self, pattern, rewriter):
        """Adds a transformation rule (pattern -> rewriter)."""
        logging.info(
            f"Adding transformation: rule={rewriter.__name__} pattern={pattern}"
        )
        self._matcher.register(pattern, rewriter)

    def clear_transformations(self):
        """Clear all registered transformations."""
        self._matcher.clear()

    @property
    def pattern_index(self):
        """Access pattern index (for backward compatibility)."""
        return self._matcher.pattern_index

    @property
    def wildcard_patterns(self):
        """Access wildcard patterns (for backward compatibility)."""
        return self._matcher.wildcard_patterns

    @log_optimization
    def optimize(
        self,
        pass_name=None,
        max_iterations=100,
        auto_cleanup=True,
        protected_nodes=None,
        context: OptimizationContext = None,
    ):
        """
        Run pattern-based optimization until convergence.

        Args:
            pass_name: Pass name for logging
            max_iterations: Maximum iterations (can be overridden by context)
            auto_cleanup: Whether to prune dead nodes (can be overridden by context)
            protected_nodes: Protected node names (can be overridden by context)
            context: Optional OptimizationContext for unified management
        """
        # Use context if provided, otherwise create from parameters
        if context:
            protected_set = context.protected_nodes
            auto_cleanup = context.auto_cleanup
            max_iterations = context.max_iterations
        else:
            protected_set = set(protected_nodes or [])

        current_graph_def = self.graph_def

        last_graph_hashes = collections.deque(maxlen=5)

        for _ in range(max_iterations):
            self.load_state(current_graph_def)

            # Simple loop detection: check if graph state has been seen recently
            graph_hash = self._compute_graph_hash(current_graph_def)
            if graph_hash in last_graph_hashes:
                logging.warning(
                    f"[{pass_name or 'unnamed'}] Infinite loop detected (stable graph state repeating). "
                    "Stopping optimization pass."
                )
                break
            last_graph_hashes.append(graph_hash)

            new_graph_def, changes = self.match_patterns_once(
                pass_name=pass_name,
                auto_cleanup=auto_cleanup,
                protected_nodes=protected_set,
            )
            if changes == 0:
                break
            current_graph_def = new_graph_def

        if auto_cleanup:
            nodes_before = len(current_graph_def.node)
            current_graph_def = self.final_prune(
                current_graph_def, pass_name=pass_name, protected_nodes=protected_set
            )
            nodes_after = len(current_graph_def.node)
            if nodes_before != nodes_after:
                prefix = f"[{pass_name}] " if pass_name else ""
                logging.info(
                    f"{prefix}Final cleanup: {nodes_before} -> {nodes_after} nodes"
                )

        return current_graph_def

    def match_patterns_once(
        self, pass_name=None, auto_cleanup=True, protected_nodes=None, context=None
    ):
        """Run a single iteration of pattern-based matching."""
        if context:
            protected_nodes = context.protected_nodes
            auto_cleanup = context.auto_cleanup
        return self._matcher.match_once(self, pass_name, auto_cleanup, protected_nodes)

    def _compute_graph_hash(self, graph_def: tf.GraphDef) -> str:
        """Computes a simple hash of the graph structure to detect cycles."""
        hasher = hashlib.sha256()
        # Sort nodes by name so ordering changes don't trigger false hashes
        # Also include node count to catch expansions/contractions directly
        sorted_nodes = sorted(graph_def.node, key=lambda n: n.name)
        hasher.update(str(len(sorted_nodes)).encode())
        for node in sorted_nodes:
            hasher.update(node.name.encode())
            hasher.update(node.op.encode())
            for inp in sorted(node.input):
                hasher.update(inp.encode())
        return hasher.hexdigest()
