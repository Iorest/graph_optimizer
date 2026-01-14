import tensorflow.compat.v1 as tf
import collections
from typing import Dict, List, Set, Optional, Any as AnyType, Tuple
from .utils.logger import (
    logger as logging,
    trace_transformation,
    log_optimization,
    log_match,
)
from .utils import save_graph


class GraphOptimizer:
    """
    Main engine for applying optimization passes to a TensorFlow GraphDef.
    Manages graph state, consumer indexing, and iterative pattern matching.
    """

    def __init__(self, graph_def: tf.GraphDef):
        self.load_state(graph_def)

        # Pattern indexing for O(1) lookup by op_type
        self.pattern_index: Dict[str, List[Tuple["Pattern", AnyType]]] = (
            collections.defaultdict(list)
        )  # op_type -> [(pattern, rewriter)]
        self.wildcard_patterns: List[
            Tuple["Pattern", AnyType]
        ] = []  # Patterns that match any op_type

    def load_state(self, graph_def: tf.GraphDef):
        """Restores the optimizer state from a GraphDef."""
        self.graph_def = graph_def
        self.nodes: Dict[str, tf.NodeDef] = {node.name: node for node in graph_def.node}
        self.consumers: Dict[str, List[str]] = collections.defaultdict(list)
        for node in graph_def.node:
            for input_name in node.input:
                base_name = input_name.split(":")[0].lstrip("^")
                self.consumers[base_name].append(node.name)
        self.pattern_index = collections.defaultdict(
            list
        )  # op_type -> [(pattern, rewriter)]
        self.wildcard_patterns = []  # Patterns that match any op_type

    def add_transformation(self, pattern, rewriter):
        """Adds a transformation rule (pattern -> rewriter)."""
        # Note: rewriter may already be wrapped with trace_transformation by PatternRewritePass
        # Don't wrap it again to avoid duplicate logging
        logging.info(
            f"Adding transformation: rule={rewriter.__name__} pattern={pattern}"
        )

        # Index pattern by op_type for faster lookup
        op_type = pattern.get_indexed_op_type()
        if op_type is None:
            # Wildcard pattern - must check against all nodes
            self.wildcard_patterns.append((pattern, rewriter))
        else:
            # Specific op_type - only check against matching nodes
            self.pattern_index[op_type].append((pattern, rewriter))

    def clear_transformations(self):
        """Clears all registered transformations."""
        self.pattern_index = collections.defaultdict(list)
        self.wildcard_patterns = []

    @log_optimization
    def optimize(
        self,
        pass_name=None,
        max_iterations=100,
        auto_cleanup=True,
        protected_nodes=None,
    ):
        # protected_nodes: nodes with zero consumers that should NOT be pruned (e.g. outputs)
        protected_nodes = set(protected_nodes or [])
        # Simplistic approach: iterate nodes and try to match
        # In a real optimizer, we might need multiple passes or a worklist
        modified = True
        iteration = 0
        current_graph_def = self.graph_def

        while modified:
            if iteration >= max_iterations:
                logging.warning(
                    f"Optimization pass '{pass_name}' reached max iterations ({max_iterations}). Stopping."
                )
                break
            iteration += 1
            modified = False
            new_nodes = []
            replaced_node_names = set()

            # Rebuild state for current graph and compute reference counts BEFORE optimization
            self.nodes = {node.name: node for node in current_graph_def.node}
            self.consumers = collections.defaultdict(list)
            for node in current_graph_def.node:
                for input_name in node.input:
                    base_name = input_name.split(":")[0].lstrip("^")
                    self.consumers[base_name].append(node.name)

            # Store reference counts before optimization
            refs_before = self._compute_reference_counts(current_graph_def)

            for node in current_graph_def.node:
                if node.name in replaced_node_names:
                    continue

                match = None

                # OPTIMIZED: Only check patterns relevant to this op_type
                # Get patterns for this specific op_type + wildcard patterns
                candidates = (
                    self.pattern_index.get(node.op, []) + self.wildcard_patterns
                )

                found_match = False
                for pattern, rewriter in candidates:
                    match = pattern.match(node, self)
                    if match:
                        new_stuff = rewriter(match, self)
                        if new_stuff is not None:
                            # PRESERVATION: Carry over ALL control dependencies from the match
                            # Filter out control deps that point to nodes WITHIN the match (internal deps)
                            internal_names = match.all_matched_nodes
                            relevant_controls = [
                                ci
                                for ci in match.control_inputs
                                if ci.lstrip("^") not in internal_names
                            ]

                            if relevant_controls and new_stuff:
                                target_node = None
                                # 1. Try to find node with same name as root
                                for new_node in new_stuff:
                                    if new_node.name == node.name:
                                        target_node = new_node
                                        break
                                # 2. Fallback: use first node in new_stuff
                                if target_node is None:
                                    target_node = new_stuff[0]

                                if target_node:
                                    existing = set(target_node.input)
                                    for ci in relevant_controls:
                                        if ci not in existing:
                                            target_node.input.append(ci)
                                            existing.add(ci)

                            new_nodes.extend(new_stuff)
                            # CRITICAL BUG FIX: Only mark the anchor node as replaced.
                            # Other nodes in the match (internal nodes) will be naturally pruned
                            # by auto_cleanup IF they no longer have any consumers (like the root).
                            # If they still have other consumers (external to this match),
                            # they MUST be preserved to avoid dangling inputs.
                            replaced_node_names.add(node.name)

                            found_match = True
                            modified = True
                            break

                if not found_match:
                    new_nodes.append(node)

            if modified:
                next_graph_def = tf.GraphDef()
                next_graph_def.node.extend(new_nodes)
                # Auto-prune dead nodes after each optimization iteration (if enabled)
                # Pass the reference counts before optimization to detect newly dead nodes
                if auto_cleanup:
                    current_graph_def = self._prune_dead_nodes(
                        next_graph_def, pass_name, refs_before, protected_nodes
                    )
                else:
                    current_graph_def = next_graph_def

        # Final cleanup: prune any remaining dead nodes (if enabled)
        # This catches nodes that became dead across multiple iterations
        if auto_cleanup:
            current_graph_def = self._final_prune(
                current_graph_def, pass_name, protected_nodes
            )

        return current_graph_def

    def _compute_reference_counts(self, graph_def: tf.GraphDef) -> Dict[str, int]:
        """Compute reference count for each node (how many other nodes use it)."""
        reference_counts: Dict[str, int] = collections.defaultdict(int)

        for node in graph_def.node:
            for input_name in node.input:
                # Strip port number and control dependency marker
                base_name = input_name.split(":")[0].lstrip("^")
                reference_counts[base_name] += 1

        return reference_counts

    def _final_prune(
        self,
        graph_def: tf.GraphDef,
        pass_name: Optional[str] = None,
        protected_nodes: Optional[Set[str]] = None,
    ) -> tf.GraphDef:
        """Final cleanup pass to remove all remaining dead nodes."""
        refs = self._compute_reference_counts(graph_def)
        protected_nodes = protected_nodes or set()

        dead_nodes: Set[str] = set()
        for node in graph_def.node:
            # Never prune Placeholders or Protected nodes
            if node.op == "Placeholder" or node.name in protected_nodes:
                continue

            # Prune any node with zero references (except Placeholders)
            if refs[node.name] == 0:
                dead_nodes.add(node.name)

        if not dead_nodes:
            return graph_def

        from .utils.logger import logger as local_logger

        local_logger.info(
            f"[{pass_name or 'optimize'}] Final pruning: removing {len(dead_nodes)} dead nodes: {', '.join(dead_nodes)}"
        )

        pruned_graph_def = tf.GraphDef()
        for node in graph_def.node:
            if node.name not in dead_nodes:
                pruned_graph_def.node.add().CopyFrom(node)

        return pruned_graph_def

    def _prune_dead_nodes(
        self, graph_def, pass_name=None, refs_before=None, protected_nodes=None
    ):
        """Remove nodes that are not referenced by any other node.

        This is called after each optimization iteration to clean up
        intermediate nodes that are no longer used (e.g., concat nodes
        that have been fused into other concat operations).

        Strategy:
        - Prune nodes that had references before but now have none (newly dead)
        - Always prune Const nodes with zero references
        - Preserve Placeholder nodes
        - Preserve Protected nodes
        - Preserve nodes that were already unreferenced (likely outputs)

        Args:
            graph_def: The current GraphDef
            pass_name: Name of the pass for logging purposes
            refs_before: Reference counts before the optimization (optional)
            protected_nodes: Set of node names to NOT prune

        Returns:
            A new GraphDef with dead nodes removed
        """
        # Compute current reference counts
        refs_after = self._compute_reference_counts(graph_def)
        protected_nodes = protected_nodes or set()

        # Identify nodes that can be pruned
        dead_nodes = set()

        for node in graph_def.node:
            ref_after = refs_after[node.name]

            # Never prune Placeholders or Protected nodes
            if node.op == "Placeholder" or node.name in protected_nodes:
                continue

            # Always prune Const nodes with zero references
            if node.op == "Const" and ref_after == 0:
                dead_nodes.add(node.name)
                continue

            # Prune nodes that BECAME dead (had refs before, now don't)
            if refs_before is not None and node.name in refs_before:
                ref_before = refs_before[node.name]
                if ref_before > 0 and ref_after == 0:
                    # This node was used before but not anymore - it's been optimized away
                    dead_nodes.add(node.name)

        # If no dead nodes, return original graph
        if not dead_nodes:
            return graph_def

        # Log dead nodes being removed
        from .utils.logger import logger as local_logger

        local_logger.info(
            f"[{pass_name or 'optimize'}] Pruning {len(dead_nodes)} dead nodes: "
            f"{', '.join(sorted(list(dead_nodes)[:5]))}"
            + (f" and {len(dead_nodes) - 5} more..." if len(dead_nodes) > 5 else "")
        )

        # Create new graph without dead nodes
        pruned_graph_def = tf.GraphDef()
        for node in graph_def.node:
            if node.name not in dead_nodes:
                pruned_graph_def.node.add().CopyFrom(node)

        return pruned_graph_def

    def canonicalize_axis(self, axis, rank):
        """Standardizes negative axes for easier comparison."""
        if axis is None:
            return None
        if axis >= 0:
            return axis
        if rank is None:
            return None  # Cannot canonicalize negative axis without rank
        return axis + rank

    def get_node_attr(self, node_or_name, attr_name, default=None):
        """Returns the unwrapped attribute value of a node."""
        node = (
            self.nodes.get(node_or_name)
            if isinstance(node_or_name, str)
            else node_or_name
        )
        if not node or attr_name not in node.attr:
            return default
        return get_attr_value(node.attr[attr_name])

    def get_node_shape(self, node_or_name):
        """Returns the output shape of a node as a list of ints."""
        node = (
            self.nodes.get(node_or_name)
            if isinstance(node_or_name, str)
            else node_or_name
        )
        if not node:
            return None

        # Try 'shape' attribute
        if "shape" in node.attr:
            return [dim.size for dim in node.attr["shape"].shape.dim]

        # Try '_output_shapes' attribute
        if "_output_shapes" in node.attr:
            try:
                shape_list = node.attr["_output_shapes"].list.shape
                if shape_list:
                    return [dim.size for dim in shape_list[0].dim]
            except Exception:
                pass
        return None

    def get_node_rank(self, node_or_name):
        """Returns the rank of a node's output tensor."""
        shape = self.get_node_shape(node_or_name)
        return len(shape) if shape is not None else None

    def prune(self, output_names):
        """Removes nodes not needed for the given outputs."""
        from tensorflow.python.framework import graph_util

        return graph_util.extract_sub_graph(self.graph_def, output_names)


class MatchContext:
    def __init__(self):
        self.matched_nodes = {}  # alias -> NodeDef or list of NodeDef
        self.all_matched_nodes = set()  # set of node names
        self.control_inputs = set()  # set of "^node_name"


class Pattern:
    def __init__(self, alias=None):
        self.alias = alias

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

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def transform(
        self,
        optimizer: GraphOptimizer,
        step=None,
        debug_dir=None,
        auto_cleanup=True,
        protected_nodes=None,
    ):
        """
        Applies the transformation to the given graph.
        Should return a new GraphDef.

        Args:
            optimizer: The GraphOptimizer instance
            step: Optional step number for debugging
            debug_dir: Optional directory to save debug output
            auto_cleanup: If True, automatically prune dead nodes after optimization (default: True)
            protected_nodes: List of node names that should not be pruned (e.g. outputs)
        """
        raise NotImplementedError()


class PatternRewritePass(BasePass):
    """A pass that applies a pattern-matching-based rewrite."""

    def __init__(self, pattern, rewriter, name=None):
        super().__init__(name)
        self.pattern = pattern
        self.rewriter = trace_transformation(rewriter)

    def transform(
        self,
        optimizer: GraphOptimizer,
        step=None,
        debug_dir=None,
        auto_cleanup=True,
        protected_nodes=None,
    ):
        """Apply this pass using GraphOptimizer.optimize()."""
        # Add this transformation to the optimizer
        optimizer.add_transformation(self.pattern, self.rewriter)

        # Run optimize() to apply the transformation iteratively
        optimized_graph = optimizer.optimize(
            pass_name=self.name,
            auto_cleanup=auto_cleanup,
            protected_nodes=protected_nodes,
        )

        # PERSISTENCE FIX: Update optimizer state for next pass
        # Must call load_state() to sync graph_def, nodes, and consumers
        optimizer.load_state(optimized_graph)

        if debug_dir and step is not None:
            import os

            filename = f"{step:02d}_{self.name}.pb"
            file_path = os.path.join(debug_dir, filename)
            save_graph(optimized_graph, file_path)

        return optimized_graph


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
