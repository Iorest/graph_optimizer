import tensorflow.compat.v1 as tf
from typing import TYPE_CHECKING
from ...utils.logger import trace_transformation
from ..base_pass import BaseOptimizationPass
from ..passes import OptimizationContext

if TYPE_CHECKING:
    from .tf_optimizer import TFGraphOptimizer


class BasePass(BaseOptimizationPass):
    """Base class for all TensorFlow graph optimization passes."""

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    def __init__(
        self, name=None, optimizer_alias=None, iterative=False, max_iterations=100
    ):
        """
        Initialize a pass.

        Args:
            name: Human-readable pass name (defaults to class name)
            optimizer_alias: Short alias for node naming (e.g., 'pack_hoist', 'concat_fuse')
                           If not provided, defaults to a simplified version of name
            iterative: If True, run transform_once() repeatedly until convergence (no changes)
            max_iterations: Maximum iterations for iterative mode (safety limit)
        """
        self.name = name or self.__class__.__name__
        self.optimizer_alias = optimizer_alias or self._generate_default_alias()
        self.iterative = iterative
        self.max_iterations = max_iterations
        self._node_counters = {}  # Per-operation-type counters for unique node naming
        self._node_cache = {}  # Node signature -> node name cache for deduplication

    def _generate_default_alias(self):
        """Generate a default optimizer alias from the pass name."""
        # Convert CamelCase to snake_case and remove 'Pass' suffix
        import re

        name = self.name
        # Remove 'Pass' suffix if present
        if name.endswith("Pass"):
            name = name[:-4]
        # Convert CamelCase to snake_case
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name

    def apply(self, optimizer: "TFGraphOptimizer") -> bool:  # type: ignore[override]
        """
        Satisfy the `BaseOptimizationPass` interface.

        Delegates to `transform()`, which is the full TF execution entry point.
        Returns True if the graph changed.
        """
        before_nodes = set(n.name for n in optimizer.graph_def.node)
        self.transform(optimizer)
        after_nodes = set(n.name for n in optimizer.graph_def.node)
        return before_nodes != after_nodes

    def make_node_name(self, root_node_name, op_type, suffix=""):
        """
        Create standardized node name for optimizer-generated nodes.
        """
        original_root = root_node_name
        for part in root_node_name.split("/"):
            if part.endswith("_pass") or part == self.optimizer_alias:
                idx = root_node_name.find(f"/{part}/")
                if idx != -1:
                    original_root = root_node_name[:idx]
                    break

        if op_type:
            base_name = f"{original_root}/{self.optimizer_alias}/{op_type}"
            if suffix:
                return f"{base_name}_{suffix}"
            return base_name
        else:
            return f"{original_root}/{self.optimizer_alias}/{suffix}"

    def make_unique_node_name(self, root_node_name, op_type):
        """
        Create a unique node name with automatic counter management.
        """
        op_type_lower = op_type.lower()
        if op_type_lower not in self._node_counters:
            self._node_counters[op_type_lower] = 0
        counter = self._node_counters[op_type_lower]
        self._node_counters[op_type_lower] += 1
        return self.make_node_name(root_node_name, "", f"{op_type_lower}_{counter}")

    def reset_counters(self):
        """Reset all node counters and caches."""
        self._node_counters.clear()
        self._node_cache.clear()

    @staticmethod
    def clean_input_name(input_name):
        from ...utils.graph_utils import extract_base_name

        return extract_base_name(input_name)

    def get_or_create_cached_node(
        self, op_type, inputs, attrs, root_node_name, context_desc="", create_func=None
    ):
        """Return an existing cached node or create a new one to avoid duplicate nodes within a pass."""
        from ...utils import create_node
        from ...utils.logger import logger as logging

        inputs_tuple = tuple(inputs)
        attrs_tuple = tuple(
            (k, attrs[k].SerializeToString())
            for k in sorted(attrs.keys())
            if not k.startswith("_")
        )
        node_signature = (op_type, inputs_tuple, attrs_tuple)

        if node_signature in self._node_cache:
            cached_name = self._node_cache[node_signature]
            logging.debug(
                f"[{self.name}] Cache hit: reusing {op_type} node {cached_name}"
            )
            return cached_name, False, None

        new_name = self.make_unique_node_name(root_node_name, op_type)

        if create_func:
            new_node = create_func(new_name, inputs, attrs)
        else:
            new_node = create_node(op_type, new_name, inputs=inputs, attr=attrs)

        self._node_cache[node_signature] = new_name
        logging.debug(f"[{self.name}] Created new {op_type} node: {new_name}")

        return new_name, True, new_node

    def transform(
        self,
        optimizer: "TFGraphOptimizer",
        step=None,
        debug_dir=None,
        auto_cleanup=True,
        protected_nodes=None,
        context: OptimizationContext = None,
        pass_name_override: str = None,
    ):
        """Execute the optimization pass."""
        self.reset_counters()

        if context is None:
            context = OptimizationContext(
                protected_nodes=protected_nodes,
                auto_cleanup=auto_cleanup,
                max_iterations=self.max_iterations,
                debug_dir=debug_dir,
            )

        effective_name = pass_name_override or self.name
        protected_set = context.protected_nodes
        original_node_count = len(optimizer.nodes)

        context.begin_pass(effective_name)

        failed = False
        try:
            if not self.iterative:
                nodes_before = len(optimizer.nodes)
                context.begin_iteration()

                result = self.transform_once(
                    optimizer, context.auto_cleanup, protected_set
                )
                changes = self._apply_transform_result(optimizer, result, nodes_before)
                nodes_after = len(optimizer.nodes)

                context.end_iteration(changes, nodes_before, nodes_after)
            else:
                while context.current_iteration < context.max_iterations:
                    context.begin_iteration()
                    nodes_before = len(optimizer.nodes)

                    result = self.transform_once(
                        optimizer, context.auto_cleanup, protected_set
                    )
                    changes = self._apply_transform_result(
                        optimizer, result, nodes_before
                    )
                    nodes_after = len(optimizer.nodes)

                    context.end_iteration(changes, nodes_before, nodes_after)

                    if changes == 0:
                        break

                if context.current_iteration >= context.max_iterations:
                    context.warn_max_iterations()

        except Exception:
            failed = True
            raise
        finally:
            context.end_pass(original_node_count, len(optimizer.nodes), failed=failed)
            self._save_debug_graph(
                optimizer.graph_def, step, context.debug_dir or debug_dir
            )

        return optimizer.graph_def

    def _apply_transform_result(self, optimizer, result, nodes_before=None):
        if isinstance(result, int):
            return result
        elif isinstance(result, tf.GraphDef):
            optimizer.load_state(result)
            return 1  # Assume at least one change if new graph returned
        else:
            return 0

    def _save_debug_graph(self, graph_def, step, debug_dir):
        """Save debug graph if debug_dir and step are provided."""
        if debug_dir and step is not None:
            import os
            from ...utils import save_graph

            if isinstance(step, int):
                filename = f"{step:02d}_{self.name}.pb"
            else:
                filename = f"{step}_{self.name}.pb"
            file_path = os.path.join(debug_dir, filename)
            save_graph(graph_def, file_path)

    def transform_once(
        self,
        optimizer: "TFGraphOptimizer",
        auto_cleanup: bool = True,
        protected_nodes: set = None,
    ):
        """Execute a single iteration of the optimization pass."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform_once()"
        )


class PatternRewritePass(BasePass):
    """
    A pass that applies a pattern-matching-based rewrite.
    """

    def __init__(self, pattern, rewriter, name=None, optimizer_alias=None):
        super().__init__(name, optimizer_alias, iterative=True, max_iterations=100)
        self.pattern = pattern
        self.rewriter = trace_transformation(rewriter)

    def transform_once(
        self,
        optimizer: "TFGraphOptimizer",
        auto_cleanup: bool = True,
        protected_nodes: set = None,
    ):
        """Execute a single iteration of pattern-based optimization."""
        optimizer.clear_transformations()
        optimizer.add_transformation(self.pattern, self.rewriter)

        new_graph_def, changes = optimizer.match_patterns_once(
            pass_name=self.name,
            auto_cleanup=auto_cleanup,
            protected_nodes=protected_nodes,
        )

        if changes > 0:
            optimizer.load_state(new_graph_def)

        return changes
