import tensorflow.compat.v1 as tf
from ...utils.logger import tf_logger as logging, log_optimization
from .graph import GraphState
from .matcher import PatternMatcher
from ..passes import OptimizationContext
from ..base_optimizer import BaseOptimizer


class TFGraphOptimizer(GraphState, BaseOptimizer):
    """
    Graph state container, query context, and pattern-based optimizer.

    Responsibilities:
    - Graph state management (via GraphState inheritance)
    - Pattern registration and matching (via PatternMatcher)
    - Driving optimization passes
    """

    def __init__(
        self,
        graph_def: tf.GraphDef,
        passes: list[str] = None,
        opt_level: int = 1,
    ):
        GraphState.__init__(self, graph_def)
        BaseOptimizer.__init__(self, graph_def, passes, opt_level)
        self._matcher = PatternMatcher()

        if passes is not None:
            resolved = passes
        else:
            from ..passes import PassRegistry

            all_passes = PassRegistry.get_passes_by_level(opt_level)
            resolved = [p for p in all_passes if not p.startswith("torch_")]

        from ..passes import PassRegistry

        self.passes = [
            PassRegistry.get_pass(name) for name in PassRegistry.sort_passes(resolved)
        ]

    @property
    def node_count(self) -> int:
        """Return the current number of nodes in the TensorFlow graph."""
        return len(self.nodes)

    # =========================================================================
    # Pattern Matching
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
        context: OptimizationContext = None,
        debug_dir: str = None,
        run_cleanup_between_passes: bool = True,
        cleanup_passes: list[str] = None,
        **kwargs,
    ):
        """
        Run the configured optimization passes on the graph.

        Args:
            context: Optional context for telemetry.
            debug_dir: Optional directory for intermediate debug dumps.
            run_cleanup_between_passes: Whether to run cleanup passes (default True).
            cleanup_passes: Passes to run for cleanup.
        """
        if context is None:
            context = OptimizationContext(debug_dir=debug_dir)

        if cleanup_passes is None:
            cleanup_passes = ["cse", "constant_fold", "algebraic_simplify"]

        # Filter out cleanup passes from the main pass list to avoid duplicating work
        if run_cleanup_between_passes:
            cleanup_set = set(cleanup_passes)
            main_passes = [p for p in self.passes if p.name not in cleanup_set]
        else:
            main_passes = self.passes

        logging.info(
            f"TFGraphOptimizer: starting optimization ({len(main_passes)} main passes, "
            f"opt_level={self.opt_level})"
        )

        for i, fx_pass in enumerate(main_passes):
            pass_name = fx_pass.name
            # Cheap pre-pass snapshot: just freeze node names (O(N) string hashing).
            # We only do the expensive CopyFrom if the pass actually mutated the graph
            # before raising, so the common success path does no unnecessary copy.
            pre_pass_graph = tf.GraphDef()
            pre_pass_graph.CopyFrom(self.graph_def)

            try:
                self.clear_transformations()
                fx_pass.transform(
                    self,
                    step=i + 1,
                    debug_dir=debug_dir,
                    context=context,
                )

                is_last_pass = i == len(main_passes) - 1
                if run_cleanup_between_passes and not is_last_pass:
                    self._run_cleanup_passes(
                        context,
                        debug_dir,
                        cleanup_passes,
                        step_num=i + 1,
                        is_final=False,
                    )
            except Exception as e:
                import traceback

                logging.error(f"Error applying pass '{pass_name}': {e}")
                logging.debug(f"Full traceback:\n{traceback.format_exc()}")
                logging.warning(
                    f"Rolling back graph state before pass '{pass_name}'..."
                )
                self.load_state(pre_pass_graph)
                continue

        if run_cleanup_between_passes:
            self._run_cleanup_passes(
                context,
                debug_dir,
                cleanup_passes,
                step_num=len(main_passes),
                is_final=True,
            )

        return self.graph_def

    def _run_cleanup_passes(
        self, context, debug_dir, cleanup_passes, step_num, is_final=False
    ):
        """Execute cleanup passes."""
        context_str = "final" if is_final else f"after step {step_num}"
        from ..passes import PassRegistry

        for pass_name in cleanup_passes:
            try:
                pass_instance = PassRegistry.get_pass(pass_name)
            except ValueError:
                continue

            logging.debug(f"Running cleanup pass '{pass_name}' {context_str}...")

            if is_final:
                debug_suffix = f"final_cleanup_{pass_name}"
                stats_name = f"{pass_name} (cleanup@final)"
            else:
                debug_suffix = f"{step_num:02d}_cleanup_{pass_name}"
                stats_name = f"{pass_name} (cleanup@{step_num})"

            backup_graph = tf.GraphDef()
            backup_graph.CopyFrom(self.graph_def)

            try:
                pass_instance.transform(
                    self,
                    step=None,
                    debug_dir=debug_dir,
                    context=context,
                    pass_name_override=stats_name,
                )

                if debug_dir:
                    import os
                    from ...utils.tf.graph_utils import save_graph

                    debug_filename = f"{debug_suffix}.pb"
                    cleanup_debug_path = os.path.join(debug_dir, debug_filename)
                    save_graph(self.graph_def, cleanup_debug_path)

            except Exception as e:
                logging.error(f"Error applying cleanup pass '{pass_name}': {e}")
                logging.warning(
                    f"Rolling back graph state before cleanup pass '{pass_name}'..."
                )
                self.load_state(backup_graph)
                continue

    def optimize_patterns(
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
        if context:
            protected_set = context.protected_nodes
            auto_cleanup = context.auto_cleanup
            max_iterations = context.max_iterations
        else:
            protected_set = set(protected_nodes or [])

        current_graph_def = self.graph_def

        # O(N) oscillation detector: track frozenset of node names seen so far.
        # This catches A→B→A rewrite cycles without O(N log N) sorting + SHA256.
        seen_fingerprints: set = set()

        for _ in range(max_iterations):
            self.load_state(current_graph_def)

            # Lightweight fingerprint: just the set of node names.
            fp = frozenset(n.name for n in current_graph_def.node)
            if fp in seen_fingerprints:
                logging.warning(
                    f"[{pass_name or 'unnamed'}] Oscillation detected (graph topology repeating). "
                    "Stopping optimization pass."
                )
                break
            seen_fingerprints.add(fp)

            new_graph_def, changes = self.match_patterns_once(
                pass_name=pass_name,
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
        self, pass_name=None, protected_nodes=None, context=None, **kwargs
    ):
        """Run a single iteration of pattern-based matching."""
        auto_cleanup = kwargs.get("auto_cleanup", True)
        if context:
            protected_nodes = context.protected_nodes
            auto_cleanup = context.auto_cleanup

        # Capture ref counts on the OLD graph before any rewrite.
        # Nodes present here with zero refs in the new graph are confirmed dead
        # (they were the replaced nodes). New output nodes added by the rewriter
        # are not in refs_before, so scoped-mode prune will never touch them —
        # even if they have zero consumers in this snapshot.
        refs_before = self.compute_reference_counts()
        new_nodes, changes = self._matcher.match_once(
            self, pass_name, False, protected_nodes
        )

        new_graph_def = tf.GraphDef()
        new_graph_def.node.extend(new_nodes)

        if changes > 0 and auto_cleanup:
            new_graph_def = self.prune_dead_nodes(
                new_graph_def, pass_name, refs_before, set(protected_nodes or [])
            )

        return new_graph_def, changes
