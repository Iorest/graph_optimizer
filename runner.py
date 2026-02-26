import os
import time
import datetime
import logging
from typing import List, Optional, Dict, Any, Iterable
from .core import OptimizationContext
from .utils.tf.graph_utils import load_graph, save_graph
from .utils import core_logger as custom_logger
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
        graph_module=None,
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
            graph_module (any, optional): PyTorch FX GraphModule object (triggers PyTorch backend).

        Note:
            If both graph_def and input_graph are provided, graph_def takes priority.
            For PyTorch, pass `graph_module`.
        """
        self.input_graph = input_graph
        self.graph_def = graph_def
        self.graph_module = graph_module
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

        for node_name in self.output_nodes:
            if node_name not in self.protected_nodes:
                self.protected_nodes.append(node_name)

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
            if not self.log_file:
                self.log_file = os.path.join(self.debug_dir, "optimization.log")

        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - [%(levelname)s] - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
                )
            )
            logging.getLogger().addHandler(file_handler)
            custom_logger.info(f"Logging to file: {self.log_file}")

    def _resolve_explicit_passes(self, backend: str) -> Optional[List[str]]:
        from .core.passes import PassRegistry

        if self.passes:
            explicit = list(self.passes)
        else:
            explicit = None  # Let optimizer resolve via opt_level

        if explicit is not None:
            for p in self.add_passes:
                if p not in explicit:
                    explicit.append(p)
            for p in self.remove_passes:
                explicit = [n for n in explicit if n != p]
            return PassRegistry.sort_passes(explicit)

        elif self.add_passes or self.remove_passes:
            explicit = PassRegistry.get_passes_by_backend(backend, self.level)

            for p in self.add_passes:
                if p not in explicit:
                    explicit.append(p)
            for p in self.remove_passes:
                explicit = [n for n in explicit if n != p]
            return PassRegistry.sort_passes(explicit)

        return None

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

    def _detect_backend(self) -> str:
        """Determines proper backend from provided instances."""
        if getattr(self, "graph_module", None) is not None:
            return "torch"
        if (
            getattr(self, "graph_def", None) is not None
            or getattr(self, "input_graph", None) is not None
        ):
            return "tensorflow"
        raise ValueError(
            "No valid input provided. Must supply graph_def, input_graph, or graph_module."
        )

    def run(self):
        """Executes the uniform optimization pipeline."""
        self._setup_logging_and_debug()
        backend = self._detect_backend()

        if backend == "torch":
            custom_logger.info("Initializing PyTorch TorchOptimizer...")
            from .core.torch import TorchOptimizer

            explicit_passes = self._resolve_explicit_passes(backend="torch")
            optimizer = TorchOptimizer(
                self.graph_module,
                passes=explicit_passes,
                opt_level=self.level,
            )
        else:
            custom_logger.info("Initializing TensorFlow TFGraphOptimizer...")
            from .core.tensorflow import TFGraphOptimizer

            graph_def = self._load_graph()
            explicit_passes = self._resolve_explicit_passes(backend="tensorflow")
            optimizer = TFGraphOptimizer(
                graph_def,
                passes=explicit_passes,
                opt_level=self.level,
            )

        initial_node_count = optimizer.node_count

        if self.protected_nodes:
            custom_logger.info(
                f"Protected nodes ({len(self.protected_nodes)}): {self.protected_nodes}"
            )

        context = OptimizationContext(
            protected_nodes=self.protected_nodes,
            auto_cleanup=True,
            debug_dir=self.debug_dir,
        )

        start_time = time.time()
        custom_logger.info(
            f"Running pipeline (backend={backend}, opt_level={self.level})..."
        )

        optimizer.optimize(
            context=context,
            debug_dir=self.debug_dir,
            run_cleanup_between_passes=self.run_cleanup_between_passes,
            cleanup_passes=self.cleanup_passes,
        )

        total_time = time.time() - start_time
        final_node_count = optimizer.node_count

        if backend == "tensorflow":
            if self.output_graph:
                custom_logger.info(f"Saving optimized graph to {self.output_graph}")
                save_graph(optimizer.graph_def, self.output_graph)
            if self.debug_dir:
                save_graph(
                    optimizer.graph_def, os.path.join(self.debug_dir, "final.pb")
                )

        self._log_final_summary(
            context, initial_node_count, final_node_count, total_time
        )

        return OptimizationReport(
            initial_nodes=initial_node_count,
            final_nodes=final_node_count,
            total_time=total_time,
            pass_stats=context._pass_stats,
            graph_def=getattr(optimizer, "graph_def", None),
            graph_module=getattr(optimizer, "graph_module", None),
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
