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
        input_graph: str,
        output_graph: Optional[str] = None,
        level: int = 1,
        debug: bool = False,
        passes: Optional[List[str]] = None,
        add_passes: Optional[List[str]] = None,
        remove_passes: Optional[List[str]] = None,
        log_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        protected_nodes: Optional[Iterable[str]] = None,
        output_nodes: Optional[Iterable[str]] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            input_graph (str): Path to input graph PB file.
            output_graph (str): Path to save optimized graph.
            level (int): Optimization level (1 or 2). Default 1.
            debug (bool): Enable debug mode (dump intermediate files). Default False.
            passes (list[str]): Explicit list of passes to run (overrides level).
            add_passes (list[str]): List of passes to append to the default set.
            remove_passes (list[str]): List of passes to remove from the set.
            log_file (str): Path to log file.
            config (dict): Optional dictionary containing configuration overrides.
                           Keys match constructor args.
        """
        self.input_graph = input_graph
        self.output_graph = output_graph
        self.level = level
        self.debug = debug
        self.passes = passes
        self.add_passes = add_passes or []
        self.remove_passes = remove_passes or []
        self.log_file = log_file
        self.output_nodes = output_nodes or []
        self.protected_nodes = list(protected_nodes or [])
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

    def run(self):
        """Executes the optimization pipeline."""
        self._setup_logging_and_debug()
        self._resolve_passes()

        if not self.input_graph:
            raise ValueError("Input graph path is not specified.")

        custom_logger.info(f"Loading graph from {self.input_graph}")
        try:
            graph_def = load_graph(self.input_graph)
        except Exception as e:
            custom_logger.error(f"Failed to load graph: {e}")
            raise

        custom_logger.info("Initializing optimizer...")
        optimizer = GraphOptimizer(graph_def)

        if self.debug_dir:
            save_graph(
                optimizer.graph_def, os.path.join(self.debug_dir, "00_initial.pb")
            )

        custom_logger.info(f"Applying passes: {self.resolved_passes}")
        for i, pass_name in enumerate(self.resolved_passes):
            if pass_name in PassRegistry._registered_passes:
                try:
                    # BACKUP: Create a deep copy of the graph def
                    # Since GraphDef is a protobuf, assume tf.GraphDef
                    import tensorflow.compat.v1 as tf

                    backup_graph = tf.GraphDef()
                    backup_graph.CopyFrom(optimizer.graph_def)

                    pass_instance = PassRegistry.get_pass(pass_name)
                    # FIX: Clear old transformations to avoid accumulation and loops
                    optimizer.clear_transformations()

                    # Pass handles saving inside transform
                    pass_instance.transform(
                        optimizer,
                        step=i + 1,
                        debug_dir=self.debug_dir,
                        protected_nodes=self.protected_nodes,
                    )
                except Exception as e:
                    custom_logger.error(f"Error applying pass '{pass_name}': {e}")
                    custom_logger.warning(
                        f"Rolling back graph state before pass '{pass_name}'..."
                    )
                    optimizer.load_state(backup_graph)
                    # Continue execution of other passes
                    continue
            else:
                custom_logger.warning(
                    f"Pass '{pass_name}' not found in registry. Skipping."
                )

        if self.output_graph:
            custom_logger.info(f"Saving optimized graph to {self.output_graph}")
            save_graph(optimizer.graph_def, self.output_graph)

        if self.debug_dir:
            save_graph(optimizer.graph_def, os.path.join(self.debug_dir, "99_final.pb"))

        custom_logger.info("Optimization completed successfully.")
        return optimizer.graph_def
