import os
import datetime
import logging
from typing import List, Optional, Dict, Any, Iterable
from .core import GraphOptimizer, PassRegistry
from .utils import load_graph, save_graph, logger as custom_logger


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
        run_cleanup_between_passes: bool = False,
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
        self.cleanup_passes = cleanup_passes or ['common_subexpression_elimination'] 
        
        # Automatically protect output nodes from pruning
        for node_name in self.output_nodes:
            if node_name not in self.protected_nodes:
                self.protected_nodes.append(node_name)

        # Apply config overrides if provided
        if config:
            self._apply_config(config)

        self.debug_dir = None
        self.resolved_passes = []

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
            self.debug_dir = f"run_{timestamp}"
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
            custom_logger.debug(f"Using explicit pass list: {final_passes}")
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

        # Priority sorting is handled by get_passes_by_level, but added passes might not be sorted.
        # Ideally, we should fetch priority for all checks and sort again?
        # User requirement: "sort passes by priority".
        # If I explicit add 'p', it might be out of order.
        # I should re-sort final_passes based on metadata logic if possible.
        # PassRegistry has metadata.
        # Let's re-sort to be safe.
        final_passes.sort(key=self._get_pass_priority)
        
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

    def _get_pass_priority(self, name):
        meta = PassRegistry._pass_metadata.get(name)
        if meta and "priority" in meta:
            return (meta["priority"], name)
        return (100, name)  # Default priority
    
    def _run_cleanup_passes(self, optimizer, opt_context, step_num, is_final=False):
        """
        Run cleanup passes (CSE, constant folding, etc.) after a main optimization pass.
        
        Args:
            optimizer: GraphOptimizer instance
            opt_context: OptimizationContext for tracking statistics (optional for backward compat)
            step_num: Current step number (0 for initial, N for after step N, N+1 for final)
            is_final: True if this is the final cleanup after all main passes
        """
        if not self.cleanup_passes:
            return
        
        # Determine context string for logging
        if is_final:
            context_str = "final"
        elif step_num == 0:
            context_str = "initial (before main passes)"
        else:
            context_str = f"after step {step_num}"
        
        for cleanup_pass_name in self.cleanup_passes:
            if cleanup_pass_name not in PassRegistry._registered_passes:
                custom_logger.warning(
                    f"Cleanup pass '{cleanup_pass_name}' not found in registry. Skipping."
                )
                continue
            
            try:
                cleanup_instance = PassRegistry.get_pass(cleanup_pass_name)
                custom_logger.debug(
                    f"Running cleanup pass '{cleanup_pass_name}' {context_str}..."
                )
                
                # Clear transformations before cleanup pass
                optimizer.clear_transformations()
                
                # Determine debug file path
                if self.debug_dir:
                    if is_final:
                        debug_filename = f"{step_num:02d}_{cleanup_pass_name}_final_cleanup.pb"
                    elif step_num == 0:
                        debug_filename = f"00_{cleanup_pass_name}_initial_cleanup.pb"
                    else:
                        debug_filename = f"{step_num:02d}_{cleanup_pass_name}_cleanup.pb"
                    cleanup_debug_path = os.path.join(self.debug_dir, debug_filename)
                else:
                    cleanup_debug_path = None
                
                # Use unique name for statistics
                if is_final:
                    stats_name = f"{cleanup_instance.name} (cleanup@final)"
                elif step_num == 0:
                    stats_name = f"{cleanup_instance.name} (cleanup@init)"
                else:
                    stats_name = f"{cleanup_instance.name} (cleanup@{step_num})"
                
                # Run cleanup pass with name override for separate statistics
                # Pass step=None to prevent BasePass from saving its own debug file
                cleanup_instance.transform(
                    optimizer,
                    step=None,  # Disable auto-save in BasePass, we save manually below
                    debug_dir=None,
                    context=opt_context,
                    pass_name_override=stats_name,
                )
                
                # Save cleanup result
                if cleanup_debug_path:
                    save_graph(optimizer.graph_def, cleanup_debug_path)
                    custom_logger.debug(f"Saved cleanup graph to {cleanup_debug_path}")
                    
            except Exception as e:
                import traceback
                custom_logger.warning(
                    f"Error in cleanup pass '{cleanup_pass_name}': {e}"
                )
                custom_logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                # Don't rollback for cleanup passes, just continue
    
    def _execute_main_passes(self, optimizer, context):
        """
        Execute all main optimization passes.
        
        Args:
            optimizer: GraphOptimizer instance
            context: OptimizationContext for tracking statistics
        """
        import tensorflow.compat.v1 as tf
        
        for i, pass_name in enumerate(self.resolved_passes):
            if pass_name not in PassRegistry._registered_passes:
                custom_logger.warning(
                    f"Pass '{pass_name}' not found in registry. Skipping."
                )
                continue
            
            try:
                # Create a backup copy of the graph def for potential rollback
                backup_graph = tf.GraphDef()
                backup_graph.CopyFrom(optimizer.graph_def)
                
                # Get pass instance and clear old transformations
                pass_instance = PassRegistry.get_pass(pass_name)
                optimizer.clear_transformations()
                
                # Execute the pass with shared context
                pass_instance.transform(
                    optimizer,
                    step=i + 1,
                    debug_dir=self.debug_dir,
                    context=context,
                )
                
                # Run cleanup passes after each main pass (if enabled)
                if self.run_cleanup_between_passes:
                    self._run_cleanup_passes(optimizer, context, step_num=i + 1, is_final=False)
                
            except Exception as e:
                import traceback
                custom_logger.error(f"Error applying pass '{pass_name}': {e}")
                custom_logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                custom_logger.warning(
                    f"Rolling back graph state before pass '{pass_name}'..."
                )
                optimizer.load_state(backup_graph)
                # Continue execution of other passes
                continue

    def run(self):
        """Executes the optimization pipeline."""
        import time
        from .core import OptimizationContext
        
        self._setup_logging_and_debug()
        self._resolve_passes()

        # Priority: graph_def > input_graph
        if self.graph_def is not None:
            custom_logger.debug("Using provided graph_def object")
            graph_def = self.graph_def
        elif self.input_graph:
            custom_logger.info(f"Loading graph from {self.input_graph}")
            try:
                graph_def = load_graph(self.input_graph)
            except Exception as e:
                custom_logger.error(f"Failed to load graph: {e}")
                raise
        else:
            raise ValueError("Either graph_def or input_graph must be provided.")

        custom_logger.info("Initializing optimizer...")
        optimizer = GraphOptimizer(graph_def)
        initial_node_count = len(optimizer.nodes)

        if self.debug_dir:
            save_graph(
                optimizer.graph_def, os.path.join(self.debug_dir, "00_initial.pb")
            )

        custom_logger.info(f"Applying {len(self.resolved_passes)} passes: {self.resolved_passes}")
        
        if self.run_cleanup_between_passes:
            custom_logger.debug(f"Cleanup passes between main passes: {self.cleanup_passes}")
        
        # Log protected nodes (output nodes + explicitly protected nodes)
        if self.protected_nodes:
            custom_logger.info(f"Protected nodes ({len(self.protected_nodes)}): {self.protected_nodes}")
        
        # Create global context for all passes
        context = OptimizationContext(
            protected_nodes=self.protected_nodes,
            auto_cleanup=True,
            debug_dir=self.debug_dir,
        )
        
        # Start timing
        start_time = time.time()
        
        # Run initial cleanup passes before all main passes (if enabled)
        if self.run_cleanup_between_passes and self.cleanup_passes:
            custom_logger.debug(f"Running initial cleanup passes: {self.cleanup_passes}")
            self._run_cleanup_passes(optimizer, context, step_num=0, is_final=False)
        
        # Execute all main optimization passes
        self._execute_main_passes(optimizer, context)
        
        # Run final cleanup passes after all main passes are done
        # Only run if run_cleanup_between_passes is True
        if self.run_cleanup_between_passes and self.cleanup_passes:
            custom_logger.debug(f"Running final cleanup passes: {self.cleanup_passes}")
            self._run_cleanup_passes(optimizer, context, step_num=len(self.resolved_passes) + 1, is_final=True)
        
        # Calculate total time
        total_time = time.time() - start_time
        final_node_count = len(optimizer.nodes)
        nodes_removed = initial_node_count - final_node_count
        
        if self.output_graph:
            custom_logger.info(f"Saving optimized graph to {self.output_graph}")
            save_graph(optimizer.graph_def, self.output_graph)

        if self.debug_dir:
            save_graph(optimizer.graph_def, os.path.join(self.debug_dir, "final.pb"))

        # Log final summary
        self._log_final_summary(context, initial_node_count, final_node_count, total_time)
        
        return optimizer.graph_def
    
    def _log_final_summary(self, context, initial_node_count, final_node_count, total_time):
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
            custom_logger.info(f"{'Pass':<30} {'Iters':>6} {'Changes':>8} {'Nodes':>15} {'Time':>8}")
            custom_logger.info("-" * 70)
            
            for pass_name, stats in context._pass_stats.items():
                iterations = len(stats['iterations'])
                total_changes = stats['total_changes']
                nodes_before = stats['nodes_before']
                nodes_after = stats['nodes_after']
                duration = stats['duration']
                
                node_diff = nodes_before - nodes_after
                nodes_str = f"{nodes_before} -> {nodes_after}" if nodes_before > 0 else "N/A"
                
                custom_logger.info(
                    f"  {pass_name:<28} {iterations:>6} {total_changes:>8} {nodes_str:>15} {duration:>7.3f}s"
                )
            
            custom_logger.info("-" * 70)
        
        # Overall statistics
        custom_logger.info("")
        custom_logger.info("Overall:")
        custom_logger.info(f"  Total passes executed: {len(context._pass_stats)}")
        custom_logger.info(f"  Total time: {total_time:.3f}s")
        custom_logger.info(f"  Nodes: {initial_node_count} -> {final_node_count} (removed: {nodes_removed})")
        
        if initial_node_count > 0:
            reduction_pct = (nodes_removed / initial_node_count) * 100
            custom_logger.info(f"  Reduction: {reduction_pct:.1f}%")
        
        custom_logger.info("=" * 70)
