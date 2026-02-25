import time
import tensorflow.compat.v1 as tf
from typing import Dict, Set, Optional, TYPE_CHECKING
from ..utils.graph_utils import extract_base_name
from ..utils.logger import logger as logging, trace_transformation

if TYPE_CHECKING:
    from .optimizer import GraphOptimizer


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


class BasePass:
    """Base class for all graph optimization passes."""

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

    def make_node_name(self, root_node_name, op_type, suffix=""):
        """
        Create standardized node name for optimizer-generated nodes.

        Format: {original_root}/{optimizer_alias}/{op_type}[_{suffix}]
                or {original_root}/{optimizer_alias}/{suffix} (if op_type is empty)

        This method extracts the original root name by removing any intermediate
        optimizer layers (e.g., '/pack_hoist/', '/concat_fusion/') to prevent
        nested naming during recursive optimizations.

        Args:
            root_node_name: The root node name (may contain optimizer layers)
            op_type: Operation type (e.g., 'Pack', 'MatMul', 'Concat'), can be empty
            suffix: Optional suffix for disambiguation (e.g., 'pack_0', 'matmul_1')

        Returns:
            Formatted node name without nested optimizer layers

        Examples:
            root='model/layer1/Pack', op='', suffix='pack_0'
            -> 'model/layer1/Pack/pack_hoist/pack_0'

            root='model/layer1/Pack/pack_hoist/pack_0', op='', suffix='matmul_1'
            -> 'model/layer1/Pack/pack_hoist/matmul_1' (not nested!)
        """
        # Remove any existing optimizer layer to get the original root
        # Simplified logic: find first instance of an optimizer layer and take everything before it
        # Optimizer layer typically follows /{alias}/ pattern
        original_root = root_node_name
        for part in root_node_name.split("/"):
            if part.endswith("_pass") or part == self.optimizer_alias:
                idx = root_node_name.find(f"/{part}/")
                if idx != -1:
                    original_root = root_node_name[:idx]
                    break

        # Build the name based on whether op_type is provided
        if op_type:
            base_name = f"{original_root}/{self.optimizer_alias}/{op_type}"
            if suffix:
                return f"{base_name}_{suffix}"
            return base_name
        else:
            # When op_type is empty, suffix should contain the full name part
            return f"{original_root}/{self.optimizer_alias}/{suffix}"

    def make_unique_node_name(self, root_node_name, op_type):
        """
        Create a unique node name with automatic counter management.

        This is a convenience method that combines make_node_name with automatic
        per-operation-type counting to ensure unique names across the optimization.

        Format: {original_root}/{optimizer_alias}/{op_type_lower}_{counter}

        Args:
            root_node_name: The root node name (may contain optimizer layers)
            op_type: Operation type (e.g., 'Pack', 'MatMul', 'Concat')

        Returns:
            Unique formatted node name with auto-incremented counter

        Examples:
            First call with op_type='MatMul' -> 'path/pack_hoist/matmul_0'
            Second call with op_type='MatMul' -> 'path/pack_hoist/matmul_1'
            First call with op_type='BiasAdd' -> 'path/pack_hoist/biasadd_0'
        """
        op_type_lower = op_type.lower()

        # Initialize counter for this op type if not exists
        if op_type_lower not in self._node_counters:
            self._node_counters[op_type_lower] = 0

        # Get current counter and increment
        counter = self._node_counters[op_type_lower]
        self._node_counters[op_type_lower] += 1

        # Generate name with counter as suffix
        return self.make_node_name(root_node_name, "", f"{op_type_lower}_{counter}")

    def reset_counters(self):
        """
        Reset all node counters and caches.

        This should typically be called at the start of each transform() to ensure
        consistent naming across optimization passes.
        """
        self._node_counters.clear()
        self._node_cache.clear()

    @staticmethod
    def clean_input_name(input_name):
        """
        Extract base node name from input (strip port and control marker).

        This is an alias for extract_base_name for compatibility with subclasses.
        """
        return extract_base_name(input_name)

    def get_or_create_cached_node(
        self, op_type, inputs, attrs, root_node_name, context_desc="", create_func=None
    ):
        """
        获取或创建缓存节点（用于 pass 内部避免重复创建相同节点）。

        缓存策略：基于 (op_type, inputs, attrs_serialized) 签名
        - 如果签名相同，返回已缓存的节点名
        - 如果签名不同，创建新节点并缓存

        Args:
            op_type: 操作类型
            inputs: 输入列表（节点名称列表，保留端口号）
            attrs: 属性字典（AttrValue 对象）
            root_node_name: 根节点名称（用于生成唯一名称）
            context_desc: 上下文描述（用于日志）
            create_func: 可选的节点创建函数 func(name, inputs, attrs) -> NodeDef

        Returns:
            tuple: (node_name, is_new_node, node_def_or_none)
        """
        from ..utils import create_node

        # 创建签名：(op_type, inputs_tuple, attrs_serialized)
        # inputs 保持原样（包含端口号）
        inputs_tuple = tuple(inputs)
        # attrs 序列化为 bytes 确保可哈希
        attrs_tuple = tuple(
            (k, attrs[k].SerializeToString())
            for k in sorted(attrs.keys())
            if not k.startswith("_")  # 跳过内部属性
        )
        node_signature = (op_type, inputs_tuple, attrs_tuple)

        # Check cache
        if node_signature in self._node_cache:
            cached_name = self._node_cache[node_signature]
            logging.debug(
                f"[{self.name}] Cache hit: reusing {op_type} node {cached_name}"
            )
            return cached_name, False, None

        # Create new node
        new_name = self.make_unique_node_name(root_node_name, op_type)

        if create_func:
            new_node = create_func(new_name, inputs, attrs)
        else:
            new_node = create_node(op_type, new_name, inputs=inputs, attr=attrs)

        # Cache node
        self._node_cache[node_signature] = new_name
        logging.debug(f"[{self.name}] Created new {op_type} node: {new_name}")

        return new_name, True, new_node

    def transform(
        self,
        optimizer: "GraphOptimizer",
        step=None,
        debug_dir=None,
        auto_cleanup=True,
        protected_nodes=None,
        context: OptimizationContext = None,
        pass_name_override: str = None,
    ):
        """
        Execute the optimization pass.

        If self.iterative is True, runs transform_once() repeatedly until convergence.
        Otherwise, runs transform_once() exactly once.

        Args:
            optimizer: The GraphOptimizer instance
            step: Optional step number for debugging
            debug_dir: Optional directory to save debug output
            auto_cleanup: If True, automatically prune dead nodes after optimization
            protected_nodes: List of node names that should not be pruned
            context: Optional OptimizationContext for unified management
            pass_name_override: Optional name override for statistics

        Returns:
            GraphDef: The optimized graph
        """
        self.reset_counters()

        # Use context if provided, otherwise create a temporary one
        if context is None:
            context = OptimizationContext(
                protected_nodes=protected_nodes,
                auto_cleanup=auto_cleanup,
                max_iterations=self.max_iterations,
                debug_dir=debug_dir,
            )

        # Use override name for statistics if provided
        effective_name = pass_name_override or self.name

        protected_set = context.protected_nodes
        original_node_count = len(optimizer.nodes)

        # Always begin pass for statistics tracking
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
            # End pass and record statistics
            context.end_pass(original_node_count, len(optimizer.nodes), failed=failed)

            # Save debug output
            self._save_debug_graph(
                optimizer.graph_def, step, context.debug_dir or debug_dir
            )

        return optimizer.graph_def

    def _apply_transform_result(self, optimizer, result, nodes_before=None):
        """
        Apply transform_once result and return change count.

        Args:
            optimizer: GraphOptimizer instance
            result: Return value from transform_once (int, GraphDef, or None)
            nodes_before: Node count before transform (for computing diff)

        Returns:
            int: Number of changes made
        """
        if isinstance(result, int):
            # transform_once returned change count (state already updated in-place)
            return result
        elif isinstance(result, tf.GraphDef):
            # transform_once returned new graph - load it
            optimizer.load_state(result)
            return 1  # Assume at least one change if new graph returned
        else:
            return 0

    def _save_debug_graph(self, graph_def, step, debug_dir):
        """Save debug graph if debug_dir and step are provided."""
        if debug_dir and step is not None:
            import os
            from ..utils import save_graph

            # Handle both int and string step values
            if isinstance(step, int):
                filename = f"{step:02d}_{self.name}.pb"
            else:
                filename = f"{step}_{self.name}.pb"
            file_path = os.path.join(debug_dir, filename)
            save_graph(graph_def, file_path)

    def transform_once(
        self,
        optimizer: "GraphOptimizer",
        auto_cleanup: bool = True,
        protected_nodes: set = None,
    ):
        """
        Execute a single iteration of the optimization pass.

        Subclasses should override this method to implement the actual optimization logic.

        Args:
            optimizer: The GraphOptimizer instance (already has current graph state)
            auto_cleanup: If True, automatically prune dead nodes
            protected_nodes: Set of node names that should not be pruned

        Returns:
            One of:
            - int: Number of changes made (for iterative convergence check)
            - GraphDef: New graph definition
            - None: No changes made
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement transform_once()"
        )


class PatternRewritePass(BasePass):
    """
    A pass that applies a pattern-matching-based rewrite.

    Uses BasePass's iterative framework with GraphOptimizer.match_patterns_once()
    for the actual pattern matching. Iterates until convergence (no more matches).
    """

    def __init__(self, pattern, rewriter, name=None, optimizer_alias=None):
        # Use iterative mode - run until convergence
        super().__init__(name, optimizer_alias, iterative=True, max_iterations=100)
        self.pattern = pattern
        self.rewriter = trace_transformation(rewriter)

    def transform_once(
        self,
        optimizer: "GraphOptimizer",
        auto_cleanup: bool = True,
        protected_nodes: set = None,
    ):
        """
        Execute a single iteration of pattern-based optimization.

        Returns:
            int: Number of changes made
        """
        # Register the pattern (clear first to avoid duplicates)
        optimizer.clear_transformations()
        optimizer.add_transformation(self.pattern, self.rewriter)

        # Run one pattern matching iteration
        new_graph_def, changes = optimizer.match_patterns_once(
            pass_name=self.name,
            auto_cleanup=auto_cleanup,
            protected_nodes=protected_nodes,
        )

        if changes > 0:
            optimizer.load_state(new_graph_def)

        return changes


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
