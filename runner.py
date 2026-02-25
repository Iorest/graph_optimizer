import os
import time
import datetime
import logging
import traceback
import tensorflow.compat.v1 as tf
from typing import List, Optional, Dict, Any, Iterable
from .core import GraphOptimizer, PassRegistry, OptimizationContext
from .utils import load_graph, save_graph, logger as custom_logger
from .utils.reporting import OptimizationReport


class OptimizationPipeline:
    """
    A facade class to configure and run the graph optimization process.
    Encapsulates the logic previously found in main.py.
    """

    def __init__(
        self,
        input_graph: Optional[str] = None,
        output_graph: Optional[str] = None,
        graph_def=None,
        level: int = 1,
        debug: bool = False,
        passes: Optional[List[str]] = None,
        add_passes: Optional[List[str]] = None,
        remove_passes: Optional[List[str]] = None,
        log_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        protected_nodes: Optional[Iterable[str]] = None,
        output_nodes: Optional[Iterable[str]] = None,
        run_cleanup_between_passes: bool = True,
        cleanup_passes: Optional[List[str]] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            input_graph (str, optional): Path to input graph PB file.
            output_graph (str, optional): Path to save optimized graph.
            graph_def (GraphDef, optional): Input graph_def object (takes priority over input_graph).
            level (int): Optimization level (1 or 2). Default 1.
            debug (bool): Enable debug mode (dump intermediate files). Default False.
            passes (list[str]): Explicit list of passes to run (overrides level).
            add_passes (list[str]): List of passes to append to the default set.
            remove_passes (list[str]): List of passes to remove from the set.
            log_file (str): Path to log file.
            config (dict): Optional dictionary containing configuration overrides.
                           Keys match constructor args.
            protected_nodes (Iterable[str], optional): Nodes to protect from pruning.
            output_nodes (Iterable[str], optional): Output nodes (automatically protected).
            run_cleanup_between_passes (bool): If True, run cleanup passes (CSE, constant folding, etc.)
                                                between each main optimization pass. Default False.
            cleanup_passes (list[str], optional): List of cleanup pass names to run between main passes.
                                                   Default: ['common_subexpression_elimination']

        Note:
            If both graph_def and input_graph are provided, graph_def takes priority.
            At least one of graph_def or input_graph must be provided.
        """
        self.input_graph = input_graph
        self.graph_def = graph_def
        self.output_graph = output_graph
        self.level = level
        self.debug = debug
        self.passes = passes
        self.add_passes = add_passes or []
        self.remove_passes = remove_passes or []
        self.log_file = log_file
        self.output_nodes = output_nodes or []
        self.protected_nodes = list(protected_nodes or [])
        self.run_cleanup_between_passes = run_cleanup_between_passes
        self.cleanup_passes = cleanup_passes or [
            "cse",
            "constant_fold",
            "algebraic_simplify",
        ]

        # Automatically protect output nodes from pruning
        for node_name in self.output_nodes:
            if node_name not in self.protected_nodes:
                self.protected_nodes.append(node_name)

        # Apply config overrides if provided
        if config:
            self._apply_config(config)

        self.debug_dir = None
        self.resolved_passes = []

    # =========================================================================
    # Fluent API (Chaining)
    # =========================================================================

    def with_input(self, input_graph: str):
        """Sets input graph path."""
        self.input_graph = input_graph
        return self

    def with_output(self, output_graph: str):
        """Sets output graph path."""
        self.output_graph = output_graph
        return self

    def with_level(self, level: int):
        """Sets optimization level."""
        self.level = level
        return self

    def with_debug(self, debug: bool = True):
        """Enables/disables debug mode."""
        self.debug = debug
        return self

    def add_pass(self, pass_name: str):
        """Appends a pass to the sequence."""
        if pass_name not in self.add_passes:
            self.add_passes.append(pass_name)
        return self

    def remove_pass(self, pass_name: str):
        """Removes a pass from the sequence."""
        if pass_name not in self.remove_passes:
            self.remove_passes.append(pass_name)
        return self

    def with_protected_nodes(self, nodes: Iterable[str]):
        """Adds nodes to protected set."""
        for n in nodes:
            if n not in self.protected_nodes:
                self.protected_nodes.append(n)
        return self

    def with_output_nodes(self, nodes: Iterable[str]):
        """Sets output nodes (and protects them)."""
        for n in nodes:
            if n not in self.output_nodes:
                self.output_nodes.append(n)
            if n not in self.protected_nodes:
                self.protected_nodes.append(n)
        return self

    def with_cleanup(self, enabled: bool = True, passes: Optional[List[str]] = None):
        """Configures intermediate cleanup passes."""
        self.run_cleanup_between_passes = enabled
        if passes:
            self.cleanup_passes = passes
        return self

    def _apply_config(self, config):
        """Merges configuration dict into instance attributes."""
        if "input_graph" in config and not self.input_graph:
            self.input_graph = config["input_graph"]
        if "output_graph" in config and not self.output_graph:
            self.output_graph = config["output_graph"]
        if "level" in config:
            self.level = config["level"]
        if "debug" in config:
            self.debug = config["debug"] or self.debug
        if "log_file" in config:
            self.log_file = config["log_file"]
        if "passes" in config:
            self.passes = config["passes"]
        if "add_passes" in config:
            self.add_passes.extend(config["add_passes"])
        if "remove_passes" in config:
            self.remove_passes.extend(config["remove_passes"])
        if "protected_nodes" in config:
            self.protected_nodes.extend(config["protected_nodes"])
        if "output_nodes" in config:
            new_outputs = config["output_nodes"]
            self.output_nodes.extend(new_outputs)
            for node_name in new_outputs:
                if node_name not in self.protected_nodes:
                    self.protected_nodes.append(node_name)
        if "run_cleanup_between_passes" in config:
            self.run_cleanup_between_passes = config["run_cleanup_between_passes"]
        if "cleanup_passes" in config:
            self.cleanup_passes = config["cleanup_passes"]

    def _setup_logging_and_debug(self):
        """Configures logging and creates debug directory."""
        if self.debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = f"outputs/run_{timestamp}"
            os.makedirs(self.debug_dir, exist_ok=True)
            # Redirect log to debug dir if not explicit
            if not self.log_file:
                self.log_file = os.path.join(self.debug_dir, "optimization.log")

        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - [%(levelname)s] - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
                )
            )
            # Attach to root logger or specific logger?
            # main.py attached to root. Let's attach to root to capture everything.
            logging.getLogger().addHandler(file_handler)
            custom_logger.info(f"Logging to file: {self.log_file}")

    def _resolve_passes(self):
        """Determines the final list of passes to execute."""
        if self.passes:
            final_passes = list(self.passes)
            custom_logger.debug(
                f"Using explicit pass list (no ordering enforced by registry): {final_passes}"
            )
        else:
            final_passes = PassRegistry.get_passes_by_level(self.level)
            custom_logger.info(
                f"Selected passes for Level {self.level}: {final_passes}"
            )

            for p in self.add_passes:
                if p not in final_passes:
                    final_passes.append(p)
                    custom_logger.debug(f"Added pass: {p}")

            for p in self.remove_passes:
                if p in final_passes:
                    final_passes.remove(p)
                    custom_logger.debug(f"Removed pass: {p}")
                else:
                    custom_logger.warning(
                        f"Pass '{p}' in remove_passes was not in the list"
                    )

            # Only re-sort if we're using default Level passes + additions,
            # this ensures added passes are placed in their proper priority order
            final_passes = PassRegistry.sort_passes(final_passes)

        # Filter out passes that are already in cleanup_passes (avoid duplicate execution)
        if self.run_cleanup_between_passes and self.cleanup_passes:
            filtered_passes = []
            for p in final_passes:
                if p in self.cleanup_passes:
                    custom_logger.debug(
                        f"Pass '{p}' excluded from main passes (already in cleanup_passes)"
                    )
                else:
                    filtered_passes.append(p)
            final_passes = filtered_passes

        self.resolved_passes = final_passes

    def _run_single_pass(
        self,
        optimizer,
        context,
        pass_name,
        step_num=None,
        pass_name_override=None,
        save_debug_suffix=None,
    ):
        """Safely executes a single pass with rollback on failure."""
        if pass_name not in PassRegistry._registered_passes:
            custom_logger.warning(
                f"Pass '{pass_name}' not found in registry. Skipping."
            )
            return False

        # Create a backup copy of the graph def for potential rollback
        backup_graph = tf.GraphDef()
        backup_graph.CopyFrom(optimizer.graph_def)

        try:
            pass_instance = PassRegistry.get_pass(pass_name)
            optimizer.clear_transformations()

            # Pass step=None if we are handling debug saving manually via save_debug_suffix
            pass_step = step_num if not save_debug_suffix else None

            pass_instance.transform(
                optimizer,
                step=pass_step,
                debug_dir=self.debug_dir if not save_debug_suffix else None,
                context=context,
                pass_name_override=pass_name_override,
            )

            # Manual debug graph save for cleanup passes
            if save_debug_suffix and self.debug_dir:
                debug_filename = f"{save_debug_suffix}.pb"
                cleanup_debug_path = os.path.join(self.debug_dir, debug_filename)
                save_graph(optimizer.graph_def, cleanup_debug_path)
                custom_logger.debug(f"Saved debug graph to {cleanup_debug_path}")

            return True

        except Exception as e:
            error_pass_name = pass_name_override or pass_name
            custom_logger.error(f"Error applying pass '{error_pass_name}': {e}")
            custom_logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            custom_logger.warning(
                f"Rolling back graph state before pass '{error_pass_name}'..."
            )
            optimizer.load_state(backup_graph)
            return False

    def _run_cleanup_passes(self, optimizer, context, step_num, is_final=False):
        """Run configured cleanup passes after a main optimization pass."""
        if not self.cleanup_passes:
            return

        context_str = (
            "final"
            if is_final
            else ("initial" if step_num == 0 else f"after step {step_num}")
        )

        for pass_name in self.cleanup_passes:
            try:
                pass_instance = PassRegistry.get_pass(pass_name)
            except ValueError:
                custom_logger.warning(
                    f"Pass '{pass_name}' not found in registry. Skipping."
                )
                continue

            custom_logger.debug(f"Running cleanup pass '{pass_name}' {context_str}...")

            # Determine debug file suffix and stats name
            if is_final:
                debug_suffix = f"{step_num:02d}_{pass_name}_final_cleanup"
                stats_name = f"{pass_instance.name} (cleanup@final)"
            elif step_num == 0:
                debug_suffix = f"00_{pass_name}_initial_cleanup"
                stats_name = f"{pass_instance.name} (cleanup@init)"
            else:
                debug_suffix = f"{step_num:02d}_{pass_name}_cleanup"
                stats_name = f"{pass_instance.name} (cleanup@{step_num})"

            self._run_single_pass(
                optimizer,
                context,
                pass_name,
                pass_name_override=stats_name,
                save_debug_suffix=debug_suffix,
            )

    def _execute_main_passes(self, optimizer, context):
        """Execute all main optimization passes sequentially."""
        for i, pass_name in enumerate(self.resolved_passes):
            success = self._run_single_pass(
                optimizer, context, pass_name, step_num=i + 1
            )

            # Skip cleanup if pass failed (as graph was rolled back anyway),
            # or if this is the last pass (handled by final cleanup)
            is_last_pass = i == len(self.resolved_passes) - 1
            if success and self.run_cleanup_between_passes and not is_last_pass:
                self._run_cleanup_passes(
                    optimizer, context, step_num=i + 1, is_final=False
                )

    def _load_graph(self):
        """Loads or returns the initial graph definition."""
        if self.graph_def is not None:
            custom_logger.debug("Using provided graph_def object")
            return self.graph_def
        elif self.input_graph:
            custom_logger.info(f"Loading graph from {self.input_graph}")
            try:
                return load_graph(self.input_graph)
            except Exception as e:
                custom_logger.error(f"Failed to load graph: {e}")
                raise
        else:
            raise ValueError("Either graph_def or input_graph must be provided.")

    def run(self):
        """Executes the optimization pipeline."""
        self._setup_logging_and_debug()
        self._resolve_passes()
        graph_def = self._load_graph()

        custom_logger.info("Initializing optimizer...")
        optimizer = GraphOptimizer(graph_def)
        initial_node_count = len(optimizer.nodes)

        if self.debug_dir:
            save_graph(
                optimizer.graph_def, os.path.join(self.debug_dir, "00_initial.pb")
            )

        custom_logger.info(
            f"Applying {len(self.resolved_passes)} passes: {self.resolved_passes}"
        )

        if self.run_cleanup_between_passes:
            custom_logger.debug(
                f"Cleanup passes between main passes: {self.cleanup_passes}"
            )

        if self.protected_nodes:
            custom_logger.info(
                f"Protected nodes ({len(self.protected_nodes)}): {self.protected_nodes}"
            )

        # Create global context for all passes
        context = OptimizationContext(
            protected_nodes=self.protected_nodes,
            auto_cleanup=True,
            debug_dir=self.debug_dir,
        )

        start_time = time.time()
        try:
            # Run initial cleanup passes before all main passes (if enabled)
            if self.run_cleanup_between_passes and self.cleanup_passes:
                custom_logger.debug(
                    f"Running initial cleanup passes: {self.cleanup_passes}"
                )
                self._run_cleanup_passes(optimizer, context, step_num=0, is_final=False)

            # Execute all main optimization passes
            self._execute_main_passes(optimizer, context)

            # Run final cleanup passes after all main passes are done
            if self.run_cleanup_between_passes and self.cleanup_passes:
                custom_logger.debug(
                    f"Running final cleanup passes: {self.cleanup_passes}"
                )
                self._run_cleanup_passes(
                    optimizer,
                    context,
                    step_num=len(self.resolved_passes) + 1,
                    is_final=True,
                )
        finally:
            total_time = time.time() - start_time
            final_node_count = len(optimizer.nodes)

            if self.output_graph:
                custom_logger.info(f"Saving optimized graph to {self.output_graph}")
                save_graph(optimizer.graph_def, self.output_graph)

            if self.debug_dir:
                save_graph(
                    optimizer.graph_def, os.path.join(self.debug_dir, "final.pb")
                )

            # Log final summary
            self._log_final_summary(
                context, initial_node_count, final_node_count, total_time
            )

        return OptimizationReport(
            initial_nodes=initial_node_count,
            final_nodes=final_node_count,
            total_time=total_time,
            pass_stats=context._pass_stats,
            graph_def=optimizer.graph_def,
        )

    def _log_final_summary(
        self, context, initial_node_count, final_node_count, total_time
    ):
        """Log final optimization summary with per-pass statistics."""
        nodes_removed = initial_node_count - final_node_count

        custom_logger.info("")
        custom_logger.info("=" * 70)
        custom_logger.info("OPTIMIZATION SUMMARY")
        custom_logger.info("=" * 70)

        # Per-pass statistics
        if context._pass_stats:
            custom_logger.info("")
            custom_logger.info("Per-Pass Statistics:")
            custom_logger.info("-" * 70)
            custom_logger.info(
                f"{'Pass':<30} {'Iters':>6} {'Changes':>8} {'Nodes':>15} {'Time':>8}"
            )
            custom_logger.info("-" * 70)

            for pass_name, stats in context._pass_stats.items():
                duration = stats.get("duration", 0.0)
                if stats.get("failed"):
                    custom_logger.info(
                        f"  {pass_name:<28} {'FAILED':>6} {'-':>8} {'-':>15} {duration:>7.3f}s"
                    )
                    continue

                iterations = len(stats.get("iterations", []))
                total_changes = stats.get("total_changes", 0)
                nodes_before = stats.get("nodes_before", 0)
                nodes_after = stats.get("nodes_after", 0)

                nodes_str = (
                    f"{nodes_before} -> {nodes_after}" if nodes_before > 0 else "N/A"
                )

                custom_logger.info(
                    f"  {pass_name:<28} {iterations:>6} {total_changes:>8} {nodes_str:>15} {duration:>7.3f}s"
                )

            custom_logger.info("-" * 70)

        # Overall statistics
        custom_logger.info("")
        custom_logger.info("Overall:")
        custom_logger.info(f"  Total passes executed: {len(context._pass_stats)}")
        custom_logger.info(f"  Total time: {total_time:.3f}s")
        custom_logger.info(
            f"  Nodes: {initial_node_count} -> {final_node_count} (removed: {nodes_removed})"
        )

        if initial_node_count > 0:
            reduction_pct = (nodes_removed / initial_node_count) * 100
            custom_logger.info(f"  Reduction: {reduction_pct:.1f}%")

        custom_logger.info("=" * 70)
