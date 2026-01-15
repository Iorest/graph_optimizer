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
            custom_logger.info(f"Using explicit pass list: {final_passes}")
        else:
            final_passes = PassRegistry.get_passes_by_level(self.level)
            custom_logger.info(
                f"Selected passes for Level {self.level}: {final_passes}"
            )

            for p in self.add_passes:
                if p not in final_passes:
                    final_passes.append(p)
                    custom_logger.info(f"Added pass: {p}")

            for p in self.remove_passes:
                if p in final_passes:
                    final_passes.remove(p)
                    custom_logger.info(f"Removed pass: {p}")
                else:
                    custom_logger.warning(
                        f"Pass '{p}' in remove_passes was not in the list."
                    )

        # Priority sorting is handled by get_passes_by_level, but added passes might not be sorted.
        # Ideally, we should fetch priority for all checks and sort again?
        # User requirement: "sort passes by priority".
        # If I explicit add 'p', it might be out of order.
        # I should re-sort final_passes based on metadata logic if possible.
        # PassRegistry has metadata.
        # Let's re-sort to be safe.
        final_passes.sort(key=self._get_pass_priority)

        self.resolved_passes = final_passes

    def _get_pass_priority(self, name):
        meta = PassRegistry._pass_metadata.get(name)
        if meta and "priority" in meta:
            return (meta["priority"], name)
        return (100, name)  # Default priority
    
    def _run_cleanup_passes(self, optimizer, step_num, is_final=False):
        """
        Run cleanup passes (CSE, constant folding, etc.) after a main optimization pass.
        
        Args:
            optimizer: GraphOptimizer instance
            step_num: Current step number (for debug output)
            is_final: True if this is the final cleanup after all main passes
        """
        if not self.cleanup_passes:
            return
        
        context = "final" if is_final else f"after step {step_num}"
        
        for cleanup_pass_name in self.cleanup_passes:
            if cleanup_pass_name not in PassRegistry._registered_passes:
                custom_logger.warning(
                    f"Cleanup pass '{cleanup_pass_name}' not found in registry. Skipping."
                )
                continue
            
            try:
                cleanup_instance = PassRegistry.get_pass(cleanup_pass_name)
                custom_logger.info(
                    f"Running cleanup pass '{cleanup_pass_name}' {context}..."
                )
                
                # Clear transformations before cleanup pass
                optimizer.clear_transformations()
                
                # Run cleanup pass (save as substep if debug enabled)
                if self.debug_dir:
                    if is_final:
                        debug_filename = f"{step_num:02d}_{cleanup_pass_name}_final_cleanup.pb"
                    else:
                        debug_filename = f"{step_num:02d}_{cleanup_pass_name}_cleanup.pb"
                    cleanup_debug_path = os.path.join(self.debug_dir, debug_filename)
                else:
                    cleanup_debug_path = None
                
                cleanup_instance.transform(
                    optimizer,
                    step=f"{step_num}_cleanup" if not is_final else "final_cleanup",
                    debug_dir=None,  # Don't let cleanup pass save its own debug files
                    protected_nodes=self.protected_nodes,
                )
                
                # Save cleanup result if debug is on
                if cleanup_debug_path:
                    save_graph(optimizer.graph_def, cleanup_debug_path)
                    
            except Exception as e:
                import traceback
                custom_logger.warning(
                    f"Error in cleanup pass '{cleanup_pass_name}': {e}"
                )
                custom_logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                # Don't rollback for cleanup passes, just continue
    
    def _execute_main_passes(self, optimizer):
        """
        Execute all main optimization passes.
        
        Args:
            optimizer: GraphOptimizer instance
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
                
                # Execute the pass
                pass_instance.transform(
                    optimizer,
                    step=i + 1,
                    debug_dir=self.debug_dir,
                    protected_nodes=self.protected_nodes,
                )
                
                # Run cleanup passes after each main pass (if enabled)
                if self.run_cleanup_between_passes:
                    self._run_cleanup_passes(optimizer, step_num=i + 1, is_final=False)
                
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
        self._setup_logging_and_debug()
        self._resolve_passes()

        # Priority: graph_def > input_graph
        if self.graph_def is not None:
            custom_logger.info("Using provided graph_def object")
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

        if self.debug_dir:
            save_graph(
                optimizer.graph_def, os.path.join(self.debug_dir, "00_initial.pb")
            )

        custom_logger.info(f"Applying passes: {self.resolved_passes}")
        
        if self.run_cleanup_between_passes:
            custom_logger.info(f"Cleanup passes will run between main passes: {self.cleanup_passes}")
        
        # Execute all main optimization passes
        self._execute_main_passes(optimizer)
        
        # Run final cleanup passes after all main passes are done
        if self.cleanup_passes:
            custom_logger.info(f"Running final cleanup passes: {self.cleanup_passes}")
            self._run_cleanup_passes(optimizer, step_num=len(self.resolved_passes) + 1, is_final=True)
        
        if self.output_graph:
            custom_logger.info(f"Saving optimized graph to {self.output_graph}")
            save_graph(optimizer.graph_def, self.output_graph)

        if self.debug_dir:
            save_graph(optimizer.graph_def, os.path.join(self.debug_dir, "final.pb"))

        custom_logger.info("Optimization completed successfully.")
        return optimizer.graph_def
