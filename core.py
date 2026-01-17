import tensorflow.compat.v1 as tf
import collections
import time
from typing import Dict, List, Set, Optional, Any as AnyType, Tuple, Union
from .utils.logger import (
    logger as logging,
    trace_transformation,
    log_optimization,
    log_match,
)
from .utils import save_graph
from .utils.graph_utils import (
    extract_base_name,
    canonicalize_axis,
    compute_reference_counts,
    update_node_inputs,
    remove_nodes,
    prune_dead_nodes,
    final_prune,
    check_external_consumers,
    log_external_consumer_warning,
    build_consumer_index,
)


# =============================================================================
# Optimization Context - Unified management for protected nodes, logging, 
#                        iteration tracking, and statistics
# =============================================================================

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
            self._pass_stats[pass_name]["iterations"].append({
                "iteration": self._current_iteration,
                "changes": changes,
                "nodes_before": nodes_before,
                "nodes_after": nodes_after,
            })
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
            logging.debug(f"[{pass_name}] Iteration {self._current_iteration}: converged")
    
    def end_pass(self, nodes_before: int, nodes_after: int):
        """Mark the end of a pass with final statistics."""
        pass_name = self._current_pass
        duration = time.time() - (self._current_pass_start or time.time())
        
        if pass_name and pass_name in self._pass_stats:
            self._pass_stats[pass_name]["duration"] = duration
        
        total_changes = self._pass_stats.get(pass_name, {}).get("total_changes", 0)
        iterations = self._current_iteration
        
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
            lines.append(
                f"  {name}: {stats['nodes_before']} -> {stats['nodes_after']} nodes "
                f"({stats['total_changes']} changes, {len(stats['iterations'])} iterations, "
                f"{stats['duration']:.3f}s)"
            )
        return "\n".join(lines)


class RewriteResult:
    """
    Result object returned by rewriter functions.
    
    Attributes:
        new_nodes: New nodes to add to the graph
        replaced_nodes: Node names to mark as replaced (in addition to anchor node)
        node_mapping: Optional dict mapping old node names to new node names for consumer updates
    """
    
    def __init__(
        self, 
        new_nodes: List[tf.NodeDef], 
        replaced_nodes: Optional[List[str]] = None,
        node_mapping: Optional[Dict[str, str]] = None
    ):
        self.new_nodes = new_nodes
        self.replaced_nodes = replaced_nodes or []
        self.node_mapping = node_mapping or {}
    
    @staticmethod
    def from_nodes(nodes_or_result):
        """Convert list/RewriteResult/None to RewriteResult format."""
        if nodes_or_result is None:
            return None
        if isinstance(nodes_or_result, RewriteResult):
            return nodes_or_result
        if isinstance(nodes_or_result, list):
            return RewriteResult(nodes_or_result)
        raise TypeError(f"Invalid rewriter return type: {type(nodes_or_result)}")


class GraphOptimizer:
    """
    Graph state container, query context, and pattern-based optimizer.
    
    Responsibilities:
    - Graph state management (graph_def, nodes dict, consumer index)
    - Node query methods (get_node_attr, get_node_shape, etc.)
    - Pattern registration and matching (via PatternMatcher)
    - Dead node pruning (delegates to utils)
    """

    def __init__(self, graph_def: tf.GraphDef):
        self._matcher = PatternMatcher()
        self.load_state(graph_def)

    # =========================================================================
    # State Management
    # =========================================================================
    
    def load_state(self, graph_def: tf.GraphDef):
        """Load graph state and rebuild consumer index."""
        self.graph_def = graph_def
        self.nodes = {node.name: node for node in graph_def.node}
        self.consumers = build_consumer_index(graph_def)
    
    def refresh_state(self):
        """Refresh internal state from current graph_def (after in-place modifications)."""
        self.nodes = {node.name: node for node in self.graph_def.node}
        self.consumers = build_consumer_index(self.graph_def)
    
    # =========================================================================
    # Node Query Methods
    # =========================================================================
    
    def get_node_attr(self, node_or_name, attr_name, default=None):
        """Returns the unwrapped attribute value of a node."""
        node = (
            self.nodes.get(node_or_name)
            if isinstance(node_or_name, str)
            else node_or_name
        )
        if node is None or attr_name not in node.attr:
            return default
        return get_attr_value(node.attr[attr_name])

    def get_node_shape(self, node_or_name):
        """Returns the output shape of a node, if available."""
        node = (
            self.nodes.get(node_or_name)
            if isinstance(node_or_name, str)
            else node_or_name
        )
        if node is None:
            return None

        if "_output_shapes" in node.attr:
            shapes = node.attr["_output_shapes"].list.shape
            if shapes:
                return [dim.size for dim in shapes[0].dim]

        if "shape" in node.attr:
            return [dim.size for dim in node.attr["shape"].shape.dim]

        return None

    def get_node_rank(self, node_or_name):
        """Returns the rank (number of dimensions) of a node's output."""
        shape = self.get_node_shape(node_or_name)
        return len(shape) if shape is not None else None
    
    def canonicalize_axis(self, axis, rank):
        """Standardizes negative axes for easier comparison."""
        return canonicalize_axis(axis, rank)

    # =========================================================================
    # Graph Modification / Pruning (delegates to utils)
    # =========================================================================
    
    def compute_reference_counts(self, graph_def: tf.GraphDef = None) -> Dict[str, int]:
        """Compute reference count for each node."""
        return compute_reference_counts(graph_def or self.graph_def)
    
    # Alias for backward compatibility
    _compute_reference_counts = compute_reference_counts
    
    def remove_nodes(self, graph_def, nodes_to_remove, pass_name=None, reason=None):
        """Create new GraphDef without specified nodes."""
        return remove_nodes(graph_def, nodes_to_remove, pass_name, reason, logging)
    
    # Alias for backward compatibility
    _remove_nodes = remove_nodes
    
    def prune_dead_nodes(self, graph_def, pass_name=None, refs_before=None, protected_nodes=None):
        """Remove nodes that became dead after optimization."""
        return prune_dead_nodes(graph_def, pass_name, refs_before, protected_nodes, logging)
    
    # Alias for backward compatibility
    _prune_dead_nodes = prune_dead_nodes
    
    def final_prune(self, graph_def, pass_name=None, protected_nodes=None):
        """Final cleanup pass to remove all remaining dead nodes."""
        return final_prune(graph_def, pass_name, protected_nodes, logger=logging)
    
    # Alias for backward compatibility
    _final_prune = final_prune
    
    def prune(self, output_names):
        """Prune the graph to only include nodes required to compute output_names."""
        from tensorflow.python.framework import graph_util
        self.graph_def = graph_util.extract_sub_graph(self.graph_def, output_names)
        self.load_state(self.graph_def)
        return self.graph_def
    
    # =========================================================================
    # Node Input Update (delegates to utils)
    # =========================================================================
    
    @staticmethod
    def _extract_base_name(input_name: str) -> str:
        """Extract base node name from input (strip port and control marker)."""
        return extract_base_name(input_name)
    
    def update_node_inputs(self, node: tf.NodeDef, node_mapping: Dict[str, str]):
        """Update node's inputs based on node_mapping."""
        update_node_inputs(node, node_mapping)
    
    # Alias for backward compatibility
    _update_node_inputs = update_node_inputs
    
    # =========================================================================
    # External Consumer Check (delegates to utils)
    # =========================================================================
    
    def check_external_consumers(self, replaced_nodes, all_replaced, internal_names):
        """Check if replaced nodes have external consumers."""
        return check_external_consumers(self.consumers, replaced_nodes, all_replaced, internal_names)
    
    # Alias for backward compatibility
    _check_external_consumers = check_external_consumers
    
    @staticmethod
    def log_external_consumer_warning(nodes_with_ext_consumers):
        """Log warning about nodes with external consumers."""
        log_external_consumer_warning(nodes_with_ext_consumers, logging)
    
    # Alias for backward compatibility
    _log_external_consumer_warning = log_external_consumer_warning

    # =========================================================================
    # Pattern Matching (delegates to PatternMatcher)
    # =========================================================================
    
    def add_transformation(self, pattern, rewriter):
        """Adds a transformation rule (pattern -> rewriter)."""
        logging.info(f"Adding transformation: rule={rewriter.__name__} pattern={pattern}")
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
        context: OptimizationContext = None
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
        
        for iteration in range(max_iterations):
            self.load_state(current_graph_def)
            new_graph_def, changes = self._matcher.match_once(
                self, pass_name=pass_name, auto_cleanup=auto_cleanup, protected_nodes=protected_set
            )
            if changes == 0:
                break
            current_graph_def = new_graph_def
        
        if auto_cleanup:
            nodes_before = len(current_graph_def.node)
            current_graph_def = self.final_prune(current_graph_def, pass_name, protected_set)
            nodes_after = len(current_graph_def.node)
            if nodes_before != nodes_after:
                prefix = f"[{pass_name}] " if pass_name else ""
                logging.info(f"{prefix}Final cleanup: {nodes_before} -> {nodes_after} nodes")
        
        return current_graph_def
    
    def match_patterns_once(self, pass_name=None, auto_cleanup=True, protected_nodes=None, context=None):
        """Run a single iteration of pattern-based matching."""
        if context:
            protected_nodes = context.protected_nodes
            auto_cleanup = context.auto_cleanup
        return self._matcher.match_once(self, pass_name, auto_cleanup, protected_nodes)


class PatternMatcher:
    """
    Pattern matching engine for graph optimization.
    
    Responsibilities:
    - Register patterns and rewriters
    - Execute single-pass pattern matching on a graph
    - Handle control dependency preservation
    - Handle node replacement and mapping
    
    Does NOT handle:
    - Iteration/convergence (handled by BasePass.transform)
    - Graph state management (handled by GraphOptimizer)
    """
    
    def __init__(self):
        self.pattern_index: Dict[str, List[Tuple["Pattern", AnyType]]] = collections.defaultdict(list)
        self.wildcard_patterns: List[Tuple["Pattern", AnyType]] = []
    
    def register(self, pattern, rewriter):
        """Register a pattern-rewriter pair."""
        op_type = pattern.get_indexed_op_type()
        if op_type is None:
            self.wildcard_patterns.append((pattern, rewriter))
        else:
            self.pattern_index[op_type].append((pattern, rewriter))
    
    def clear(self):
        """Clear all registered patterns."""
        self.pattern_index = collections.defaultdict(list)
        self.wildcard_patterns = []
    
    def match_once(
        self,
        optimizer: GraphOptimizer,
        pass_name: str = None,
        auto_cleanup: bool = True,
        protected_nodes: set = None,
    ):
        """
        Run a single iteration of pattern matching.
        
        Args:
            optimizer: GraphOptimizer with current graph state
            pass_name: Pass name for logging
            auto_cleanup: If True, prune dead nodes after matching
            protected_nodes: Nodes that should not be pruned
        
        Returns:
            tuple: (new_graph_def, changes_count)
        """
        protected_nodes = set(protected_nodes or [])
        optimizer.protected_nodes = protected_nodes
        optimizer.current_pass_name = pass_name  # Set for logging in Pattern.match
        
        refs_before = optimizer.compute_reference_counts()
        nodes_before = len(optimizer.graph_def.node)
        prefix = f"[{pass_name}] " if pass_name else ""
        
        new_nodes = []
        replaced_node_names = set()
        added_node_names = []  # Track newly added nodes for logging
        global_node_mapping = {}
        modified = False

        for node in optimizer.graph_def.node:
            if node.name in replaced_node_names:
                continue

            candidates = self.pattern_index.get(node.op, []) + self.wildcard_patterns

            found_match = False
            for pattern, rewriter in candidates:
                match = pattern.match(node, optimizer)
                if match:
                    rewriter_output = rewriter(match, optimizer)
                    if rewriter_output is not None:
                        result = RewriteResult.from_nodes(rewriter_output)
                        
                        # Preserve external control dependencies
                        internal_names = match.all_matched_nodes
                        relevant_controls = [
                            ci for ci in match.control_inputs
                            if ci.lstrip("^") not in internal_names
                        ]

                        if relevant_controls and result.new_nodes:
                            target_node = result.new_nodes[0]
                            for new_node in result.new_nodes:
                                if new_node.name == node.name:
                                    target_node = new_node
                                    break
                            if target_node:
                                existing = set(target_node.input)
                                for ci in relevant_controls:
                                    if ci not in existing:
                                        target_node.input.append(ci)
                                        existing.add(ci)

                        # Log replaced root node
                        logging.info(f"{prefix}Replaced: {node.name} (op: {node.op})")
                        replaced_node_names.add(node.name)
                        
                        # Track and log new nodes
                        for new_node in result.new_nodes:
                            new_nodes.append(new_node)
                            # Only log truly new nodes (not same name as replaced)
                            if new_node.name != node.name:
                                added_node_names.append((new_node.name, new_node.op))
                        
                        if result.node_mapping:
                            global_node_mapping.update(result.node_mapping)
                        
                        if result.replaced_nodes:
                            all_replaced = {node.name} | set(result.replaced_nodes)
                            nodes_with_ext_consumers = optimizer.check_external_consumers(
                                result.replaced_nodes, all_replaced, internal_names
                            )
                            
                            if nodes_with_ext_consumers:
                                optimizer.log_external_consumer_warning(nodes_with_ext_consumers)
                                safe_to_replace = [
                                    name for name in result.replaced_nodes
                                    if name not in [n for n, _ in nodes_with_ext_consumers]
                                ]
                                replaced_node_names.update(safe_to_replace)
                            else:
                                replaced_node_names.update(result.replaced_nodes)

                        found_match = True
                        modified = True
                        break

            if not found_match:
                new_nodes.append(node)

        if not modified:
            return optimizer.graph_def, 0

        # Log newly added nodes
        for node_name, node_op in added_node_names:
            logging.info(f"{prefix}Added: {node_name} (op: {node_op})")

        # Apply node mappings
        if global_node_mapping:
            logging.debug(f"Applying node mapping: {len(global_node_mapping)} remappings")
            for node in new_nodes:
                optimizer.update_node_inputs(node, global_node_mapping)
        
        # Build new graph
        new_graph_def = tf.GraphDef()
        new_graph_def.node.extend(new_nodes)
        
        if auto_cleanup:
            new_graph_def = optimizer.prune_dead_nodes(
                new_graph_def, pass_name, refs_before, protected_nodes
            )
        
        # Log iteration summary
        nodes_after = len(new_graph_def.node)
        node_diff = nodes_before - nodes_after
        logging.info(
            f"{prefix}Summary: {nodes_before} -> {nodes_after} nodes "
            f"(replaced: {len(replaced_node_names)}, added: {len(added_node_names)}, diff: -{node_diff})"
        )
        
        return new_graph_def, len(replaced_node_names)


class MatchContext:
    def __init__(self):
        self.matched_nodes = {}  # alias -> NodeDef or list of NodeDef
        self.all_matched_nodes = set()  # set of node names
        self.control_inputs = set()  # set of "^node_name"


class Pattern:
    def __init__(self, alias=None):
        self.alias = alias
        self.consumer_count = None  # Expected number of consumers (None = any)

    @log_match
    def match(
        self,
        node: tf.NodeDef,
        optimizer: "GraphOptimizer",
        context: Optional["MatchContext"] = None,
    ) -> Optional["MatchContext"]:
        if context is None:
            context = MatchContext()
        if self._match_internal(node, optimizer, context):
            context.all_matched_nodes.add(node.name)
            if self.alias:
                context.matched_nodes[self.alias] = node
            return context
        return None

    def _match_internal(self, node, optimizer, context):
        res = self._do_match(node, optimizer, context)
        if res:
            context.all_matched_nodes.add(node.name)
            # Accumulate ALL control dependencies from all matched nodes
            for input_name in node.input:
                if input_name.startswith("^"):
                    context.control_inputs.add(input_name)

            if self.alias:
                context.matched_nodes[self.alias] = node
        return res

    def _do_match(self, node, optimizer, context):
        raise NotImplementedError()

    def get_indexed_op_type(self):
        """Return op_type for indexing, or None for wildcard patterns.

        Returns:
            str: Operation type to index under, or None for patterns that match any op.
        """
        return None  # Default: treat as wildcard


class OpPattern(Pattern):
    def __init__(self, op_type, inputs=None, attrs=None, shape=None, alias=None):
        super().__init__(alias)
        self.op_type = op_type
        self.inputs = inputs or []  # List of Pattern
        self.attrs = attrs or {}  # Map of attr_name -> attr_value (or predicate)
        self.shape = shape  # Expected output shape (list of ints or None for wildcard)
        self.consumer_count = None  # Expected number of consumers

    def get_indexed_op_type(self):
        """Return op_type for indexing. Wildcards (*) return None."""
        return None if self.op_type == "*" else self.op_type

    def _do_match(self, node, optimizer, context):
        if self.op_type != "*" and node.op != self.op_type:
            return False

        # Match attributes
        if self.attrs:
            for attr_name, expected in self.attrs.items():
                if attr_name not in node.attr:
                    return False
                actual_attr = node.attr[attr_name]
                actual = self._get_attr_value(actual_attr)

                if callable(expected):
                    if not expected(actual):
                        return False
                elif actual != expected:
                    return False

        # Match shape
        if self.shape is not None:
            if not self._match_shape(node):
                return False
        # Match inputs
        if len(self.inputs) > 0:
            # Split inputs into data and control
            data_inputs = [i for i in node.input if not i.startswith("^")]

            # Check if any input pattern is variadic
            variadic_idx = self._find_variadic_pattern()

            if variadic_idx is None:
                # Exact matching (existing behavior)
                if len(data_inputs) != len(self.inputs):
                    return False

                for i, input_pattern in enumerate(self.inputs):
                    if not self._match_single_input(
                        data_inputs[i], input_pattern, optimizer, context
                    ):
                        return False
            else:
                # Variadic matching
                if not self._match_variadic_inputs(
                    data_inputs, optimizer, context, variadic_idx
                ):
                    return False

        # Match consumer count
        if self.consumer_count is not None:
            if len(optimizer.consumers[node.name]) != self.consumer_count:
                return False

        return True

    def _find_variadic_pattern(self):
        """Find index of variadic pattern in inputs, or None if no variadic."""
        for i, pattern in enumerate(self.inputs):
            if isinstance(pattern, VariadicPattern):
                return i
        return None

    def _match_single_input(self, input_name, input_pattern, optimizer, context):
        """Match a single input against a pattern."""
        base_name = input_name.split(":")[0].lstrip("^")
        if base_name not in optimizer.nodes:
            return False
        input_node = optimizer.nodes[base_name]
        return input_pattern._match_internal(input_node, optimizer, context)

    def _match_variadic_inputs(self, data_inputs, optimizer, context, variadic_idx):
        """Match data inputs when a variadic pattern is present."""
        variadic_pattern = self.inputs[variadic_idx]
        min_count = variadic_pattern.min_count
        max_count = (
            variadic_pattern.max_count
            if variadic_pattern.max_count is not None
            else float("inf")
        )

        # Calculate expected input counts
        fixed_before = variadic_idx
        fixed_after = len(self.inputs) - variadic_idx - 1
        min_total = fixed_before + min_count + fixed_after
        max_total = fixed_before + max_count + fixed_after

        if not (min_total <= len(data_inputs) <= max_total):
            return False

        if variadic_pattern.alias:
            context.matched_nodes[variadic_pattern.alias] = []

        # Match fixed inputs before variadic
        for i in range(fixed_before):
            if not self._match_single_input(
                data_inputs[i], self.inputs[i], optimizer, context
            ):
                return False

        # Match variadic inputs
        variadic_count = len(data_inputs) - fixed_before - fixed_after
        for i in range(variadic_count):
            input_name = data_inputs[fixed_before + i]
            base_name = input_name.split(":")[0].lstrip("^")
            input_node = optimizer.nodes[base_name]

            if not variadic_pattern.pattern._match_internal(
                input_node, optimizer, context
            ):
                return False

            if variadic_pattern.alias:
                context.matched_nodes[variadic_pattern.alias].append(input_node)

        # Match fixed inputs after variadic
        for i in range(fixed_after):
            if not self._match_single_input(
                data_inputs[fixed_before + variadic_count + i],
                self.inputs[variadic_idx + 1 + i],
                optimizer,
                context,
            ):
                return False

        return True

    def _match_shape(self, node):
        """Checks if the node's output shape matches self.shape."""
        actual_shape = self._get_node_shape(node)
        if actual_shape is None:
            return False

        if len(actual_shape) != len(self.shape):
            return False

        for actual_dim, expected_dim in zip(actual_shape, self.shape):
            if expected_dim is not None and actual_dim != expected_dim:
                return False
        return True

    def _get_node_shape(self, node):
        """Extracts shape information from a NodeDef."""
        # Try 'shape' attribute (common for Placeholders/Const)
        if "shape" in node.attr:
            return [dim.size for dim in node.attr["shape"].shape.dim]

        # Try '_output_shapes' attribute (common for general ops)
        if "_output_shapes" in node.attr:
            # Usually it's a list of shapes, we take the first one
            try:
                shape_list = node.attr["_output_shapes"].list.shape
                if shape_list:
                    return [dim.size for dim in shape_list[0].dim]
            except Exception:
                pass
        return None

    def _get_attr_value(self, attr_proto):
        """Unwraps a TensorFlow AttrValue proto into a Python literal."""
        return get_attr_value(attr_proto)


def get_attr_value(attr_proto):
    """Unwraps a TensorFlow AttrValue proto into a Python literal."""
    field = attr_proto.WhichOneof("value")
    if field == "s":
        return attr_proto.s.decode("utf-8")
    if field == "i":
        return attr_proto.i
    if field == "f":
        return attr_proto.f
    if field == "b":
        return attr_proto.b
    if field == "type":
        return attr_proto.type
    if field == "shape":
        return [dim.size for dim in attr_proto.shape.dim]
    if field == "tensor":
        from tensorflow.python.framework import tensor_util
        import numpy as np

        t = tensor_util.MakeNdarray(attr_proto.tensor)
        if np.isscalar(t) or t.ndim == 0:
            return t.item()
        return t
    # Fallback to the proto itself for complex types
    return attr_proto


class WildcardPattern(Pattern):
    def _do_match(self, node, optimizer, context):
        if self.consumer_count is not None:
            if len(optimizer.consumers[node.name]) != self.consumer_count:
                return False
        return True

    def get_indexed_op_type(self):
        """Wildcard patterns match any operation."""
        return None


class VariadicPattern(Pattern):
    """Matches zero or more consecutive inputs matching the same pattern.

    This is used within OpPattern.inputs to indicate that the operator
    can accept a variable number of inputs matching the specified pattern.
    """

    def __init__(self, pattern, min_count=0, max_count=None, alias=None):
        super().__init__(alias)
        self.pattern = pattern  # Pattern that each variadic input must match
        self.min_count = min_count  # Minimum number of inputs
        self.max_count = max_count  # Maximum number of inputs (None = unlimited)

    def _do_match(self, node, optimizer, context):
        # VariadicPattern is only used within OpPattern.inputs
        # It should not be directly matched against nodes
        raise NotImplementedError(
            "VariadicPattern should only be used within OpPattern.inputs"
        )

    def get_indexed_op_type(self):
        """Variadic is not an operation, it's a pattern modifier."""
        return None


# Helper functions to build patterns
def Op(op_type, *inputs, alias=None, attrs=None, shape=None, consumer_count=None):
    pattern = OpPattern(op_type, list(inputs), attrs, shape, alias)
    pattern.consumer_count = consumer_count
    return pattern


def Attr(name, value):
    """Helper for attribute matching."""
    return {name: value}


def Shape(dims):
    """Helper for shape matching."""
    return list(dims)


def Any(alias=None, consumer_count=None):
    pattern = WildcardPattern(alias)
    pattern.consumer_count = consumer_count
    return pattern


def Variadic(pattern, min_count=0, max_count=None, alias=None):
    """Create a variadic pattern for matching multiple inputs.

    Args:
        pattern: Pattern that each variadic input must match
        min_count: Minimum number of inputs (default: 0)
        max_count: Maximum number of inputs (default: unlimited)
        alias: Optional alias for the variadic group

    Returns:
        VariadicPattern instance

    Example:
        # Match Concat with at least 2 constant inputs
        Op("ConcatV2", Variadic(Op("Const"), min_count=2), Op("Const", alias="axis"))
    """
    return VariadicPattern(pattern, min_count, max_count, alias)


class CommutativeOpPattern(OpPattern):
    """Matches an Op where the order of the first two inputs doesn't matter."""

    def __init__(self, op_type, inputs, attrs=None, shape=None, alias=None):
        super().__init__(op_type, inputs, attrs, shape, alias)

    def get_indexed_op_type(self):
        """Inherit from OpPattern - index by specific op_type."""
        return super().get_indexed_op_type()

    def _do_match(self, node, optimizer, context):
        if self.op_type != "*" and node.op != self.op_type:
            return False

        # Try default order
        if super()._do_match(node, optimizer, context):
            return True

        # Special handling for inputs: try swapped order for 2 data inputs
        data_inputs = [i for i in node.input if not i.startswith("^")]
        if len(data_inputs) != 2 or len(self.inputs) != 2:
            return False

        # Instead of full copy, just swap temporarily and swap back
        original_inputs = list(node.input)

        # Find indices of data inputs
        data_indices = [
            i for i, name in enumerate(node.input) if not name.startswith("^")
        ]
        if len(data_indices) != 2:
            return False

        # Swap them
        i1, i2 = data_indices[0], data_indices[1]
        node.input[i1], node.input[i2] = original_inputs[i2], original_inputs[i1]

        try:
            # Try matching with swapped inputs
            saved_nodes = context.matched_nodes.copy()
            saved_all = context.all_matched_nodes.copy()

            res = super()._do_match(node, optimizer, context)
            if not res:
                # Restore context if match failed
                context.matched_nodes = saved_nodes
                context.all_matched_nodes = saved_all
            return res
        finally:
            # ALWAYS restore original inputs
            node.input[i1], node.input[i2] = original_inputs[i1], original_inputs[i2]


def CommutativeOp(
    op_type, p1, p2, alias=None, attrs=None, shape=None, consumer_count=None
):
    pattern = CommutativeOpPattern(op_type, [p1, p2], attrs, shape, alias)
    pattern.consumer_count = consumer_count
    return pattern


def ConstValue(value, alias=None):
    """Matches a Const node with a specific value."""

    def check_value(unwrapped_value):
        return unwrapped_value == value

    return Op("Const", attrs={"value": check_value}, alias=alias)


class BasePass:
    """Base class for all graph optimization passes."""

    def __init__(self, name=None, optimizer_alias=None, iterative=False, max_iterations=100):
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
        if name.endswith('Pass'):
            name = name[:-4]
        # Convert CamelCase to snake_case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
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
        # Pattern: anything like '/optimizer_alias/' should be removed
        parts = root_node_name.split('/')
        
        # Find and remove optimizer layer patterns (e.g., 'pack_hoist', 'concat_fusion')
        cleaned_parts = []
        skip_next = False
        for i, part in enumerate(parts):
            if skip_next:
                skip_next = False
                continue
            # Check if this looks like an optimizer alias (snake_case with common patterns)
            if ('_' in part and (part.endswith('_hoist') or part.endswith('_fusion') or 
                                  part.endswith('_rm') or part.endswith('_fuse'))):
                # This is an optimizer layer, skip it and the next part (op name)
                skip_next = True
                continue
            cleaned_parts.append(part)
        
        original_root = '/'.join(cleaned_parts)
        
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
    
    def get_or_create_cached_node(self, op_type, inputs, attrs, root_node_name, 
                                   context_desc="", create_func=None):
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
        from .utils import create_node
        
        # 创建签名：(op_type, inputs_tuple, attrs_serialized)
        # inputs 保持原样（包含端口号）
        inputs_tuple = tuple(inputs)
        # attrs 序列化为 bytes 确保可哈希
        attrs_tuple = tuple(
            (k, attrs[k].SerializeToString()) 
            for k in sorted(attrs.keys()) 
            if not k.startswith('_')  # 跳过内部属性
        )
        node_signature = (op_type, inputs_tuple, attrs_tuple)
        
        # Check cache
        if node_signature in self._node_cache:
            cached_name = self._node_cache[node_signature]
            logging.debug(f"[{self.name}] Cache hit: reusing {op_type} node {cached_name}")
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
        optimizer: GraphOptimizer,
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
        
        State Management:
        - transform_once() can modify optimizer state in-place (call optimizer.refresh_state())
        - Or return a new GraphDef (this method will call optimizer.load_state())
        - Or return int (change count) if state was already updated in-place

        Args:
            optimizer: The GraphOptimizer instance
            step: Optional step number for debugging
            debug_dir: Optional directory to save debug output
            auto_cleanup: If True, automatically prune dead nodes after optimization
            protected_nodes: List of node names that should not be pruned
            context: Optional OptimizationContext for unified management
            pass_name_override: Optional name override for statistics (e.g., "CSE (cleanup)")
            
        Returns:
            GraphDef: The optimized graph
        """
        from .utils import save_graph
        
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
        
        if not self.iterative:
            # Single execution mode - still track as 1 iteration
            nodes_before = len(optimizer.nodes)
            context.begin_iteration()
            
            result = self.transform_once(optimizer, context.auto_cleanup, protected_set)
            changes = self._apply_transform_result(optimizer, result, nodes_before)
            nodes_after = len(optimizer.nodes)
            
            context.end_iteration(changes, nodes_before, nodes_after)
        else:
            # Iterative mode - run until convergence
            while context.current_iteration < context.max_iterations:
                iteration = context.begin_iteration()
                nodes_before = len(optimizer.nodes)
                
                result = self.transform_once(optimizer, context.auto_cleanup, protected_set)
                changes = self._apply_transform_result(optimizer, result, nodes_before)
                nodes_after = len(optimizer.nodes)
                
                context.end_iteration(changes, nodes_before, nodes_after)
                
                if changes == 0:
                    break
            
            if context.current_iteration >= context.max_iterations:
                context.warn_max_iterations()
        
        # End pass and record statistics
        context.end_pass(original_node_count, len(optimizer.nodes))
        
        # Save debug output
        self._save_debug_graph(optimizer.graph_def, step, context.debug_dir or debug_dir)
        
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
            if nodes_before is not None:
                return abs(nodes_before - len(optimizer.nodes))
            return 1  # Assume at least one change if new graph returned
        else:
            # None or other - check node count diff
            if nodes_before is not None:
                return abs(nodes_before - len(optimizer.nodes))
            return 0
    
    def _save_debug_graph(self, graph_def, step, debug_dir):
        """Save debug graph if debug_dir and step are provided."""
        if debug_dir and step is not None:
            import os
            from .utils import save_graph
            # Handle both int and string step values
            if isinstance(step, int):
                filename = f"{step:02d}_{self.name}.pb"
            else:
                filename = f"{step}_{self.name}.pb"
            file_path = os.path.join(debug_dir, filename)
            save_graph(graph_def, file_path)
    
    def transform_once(
        self,
        optimizer: GraphOptimizer,
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
        optimizer: GraphOptimizer,
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
    def list_available_passes(cls):
        """Returns a list of all registered pass names."""
        return list(cls._registered_passes.keys())

    @classmethod
    def get_passes_by_level(cls, level):
        """Returns a list of pass names enabled at the given optimization level, sorted by priority."""
        # Collect passes that match the level
        candidates = []
        for name, meta in cls._pass_metadata.items():
            if meta["opt_level"] <= level:
                candidates.append((name, meta["priority"]))

        # Sort by priority (asc), then name (asc)
        candidates.sort(key=lambda x: (x[1], x[0]))

        return [name for name, _ in candidates]
