import tensorflow.compat.v1 as tf
import collections
from typing import Dict, List, Set, Optional, Any as AnyType, Tuple, Union
from .utils.logger import (
    logger as logging,
    trace_transformation,
    log_optimization,
    log_match,
)
from .utils import save_graph


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
        """Restore optimizer state from GraphDef and rebuild consumer index."""
        self.graph_def = graph_def
        self.nodes = {node.name: node for node in graph_def.node}
        self._rebuild_consumer_index()

    def add_transformation(self, pattern, rewriter):
        """Adds a transformation rule (pattern -> rewriter)."""
        logging.info(
            f"Adding transformation: rule={rewriter.__name__} pattern={pattern}"
        )

        op_type = pattern.get_indexed_op_type()
        if op_type is None:
            self.wildcard_patterns.append((pattern, rewriter))
        else:
            self.pattern_index[op_type].append((pattern, rewriter))

    def clear_transformations(self):
        """Clear all registered transformations."""
        self.pattern_index = collections.defaultdict(list)
        self.wildcard_patterns = []
    
    def _rebuild_consumer_index(self):
        """Rebuild consumer index from current graph nodes."""
        self.consumers = collections.defaultdict(list)
        for node in self.graph_def.node:
            for input_name in node.input:
                base_name = input_name.split(":")[0].lstrip("^")
                self.consumers[base_name].append(node.name)
    
    @staticmethod
    def _extract_base_name(input_name: str) -> str:
        """
        Extract base node name from input (strip port and control marker).
        
        Examples:
            'node:0' -> 'node'
            '^control_dep' -> 'control_dep'
            'node:1' -> 'node'
            'node' -> 'node'
        """
        return input_name.split(":")[0].lstrip("^")
    
    def _update_node_inputs(self, node: tf.NodeDef, node_mapping: Dict[str, str]):
        """
        Update node's inputs based on node_mapping (old_name -> new_name).
        Preserves port numbers and control dependency markers.
        """
        updated_inputs = []
        for input_name in node.input:
            # Parse input: ^control or name:port or name
            is_control = input_name.startswith("^")
            base_name = self._extract_base_name(input_name)
            
            # Extract port if present
            port = ""
            if not is_control and ":" in input_name:
                port = ":" + input_name.split(":", 1)[1]
            
            # Remap if in mapping
            if base_name in node_mapping:
                new_base = node_mapping[base_name]
                if is_control:
                    updated_inputs.append(f"^{new_base}")
                else:
                    updated_inputs.append(f"{new_base}{port}")
            else:
                updated_inputs.append(input_name)
        
        # Update in place
        del node.input[:]
        node.input.extend(updated_inputs)

    @log_optimization
    def optimize(
        self,
        pass_name=None,
        max_iterations=100,
        auto_cleanup=True,
        protected_nodes=None,
    ):
        protected_nodes = set(protected_nodes or [])
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
            global_node_mapping = {}  # Collect all node mappings in this iteration

            # Rebuild state and compute reference counts before optimization
            self.nodes = {node.name: node for node in current_graph_def.node}
            self._rebuild_consumer_index()
            refs_before = self._compute_reference_counts(current_graph_def)

            for node in current_graph_def.node:
                if node.name in replaced_node_names:
                    continue

                match = None

                # Check patterns relevant to this op_type + wildcard patterns
                candidates = (
                    self.pattern_index.get(node.op, []) + self.wildcard_patterns
                )

                found_match = False
                for pattern, rewriter in candidates:
                    match = pattern.match(node, self)
                    if match:
                        rewriter_output = rewriter(match, self)
                        if rewriter_output is not None:
                            result = RewriteResult.from_nodes(rewriter_output)
                            
                            # Preserve external control dependencies
                            internal_names = match.all_matched_nodes
                            relevant_controls = [
                                ci
                                for ci in match.control_inputs
                                if ci.lstrip("^") not in internal_names
                            ]

                            if relevant_controls and result.new_nodes:
                                target_node = None
                                for new_node in result.new_nodes:
                                    if new_node.name == node.name:
                                        target_node = new_node
                                        break
                                if target_node is None:
                                    target_node = result.new_nodes[0]

                                if target_node:
                                    existing = set(target_node.input)
                                    for ci in relevant_controls:
                                        if ci not in existing:
                                            target_node.input.append(ci)
                                            existing.add(ci)

                            new_nodes.extend(result.new_nodes)
                            replaced_node_names.add(node.name)
                            
                            # Collect node mappings from this rewrite
                            if result.node_mapping:
                                global_node_mapping.update(result.node_mapping)
                            
                            # Mark additional nodes as replaced if specified
                            if result.replaced_nodes:
                                # Safety check: verify no external consumers
                                all_replaced = {node.name} | set(result.replaced_nodes)
                                nodes_with_ext_consumers = self._check_external_consumers(
                                    result.replaced_nodes, all_replaced, internal_names
                                )
                                
                                if nodes_with_ext_consumers:
                                    self._log_external_consumer_warning(nodes_with_ext_consumers)
                                    safe_to_replace = [
                                        name for name in result.replaced_nodes
                                        if name not in [n for n, _ in nodes_with_ext_consumers]
                                    ]
                                    replaced_node_names.update(safe_to_replace)
                                    logging.info(
                                        f"Marked {len(safe_to_replace)}/{len(result.replaced_nodes)} nodes as replaced "
                                        f"(skipped {len(nodes_with_ext_consumers)} with external consumers)"
                                    )
                                else:
                                    replaced_node_names.update(result.replaced_nodes)

                            found_match = True
                            modified = True
                            break

                if not found_match:
                    new_nodes.append(node)

            if modified:
                # Apply node mappings to update all consumer references
                if global_node_mapping:
                    logging.info(f"Applying node mapping: {len(global_node_mapping)} remappings")
                    for node in new_nodes:
                        self._update_node_inputs(node, global_node_mapping)
                
                next_graph_def = tf.GraphDef()
                next_graph_def.node.extend(new_nodes)
                if auto_cleanup:
                    current_graph_def = self._prune_dead_nodes(
                        next_graph_def, pass_name, refs_before, protected_nodes
                    )
                else:
                    current_graph_def = next_graph_def

        # Final cleanup
        if auto_cleanup:
            current_graph_def = self._final_prune(
                current_graph_def, pass_name, protected_nodes
            )

        return current_graph_def

    def _compute_reference_counts(self, graph_def: tf.GraphDef) -> Dict[str, int]:
        """Compute reference count for each node."""
        reference_counts: Dict[str, int] = collections.defaultdict(int)
        for node in graph_def.node:
            for input_name in node.input:
                reference_counts[self._extract_base_name(input_name)] += 1
        return reference_counts
    
    def _check_external_consumers(self, replaced_nodes, all_replaced, internal_names):
        """Check if replaced nodes have external consumers (not in replaced set)."""
        nodes_with_ext_consumers = []
        for replaced_name in replaced_nodes:
            consumers = self.consumers.get(replaced_name, [])
            external_consumers = [
                c for c in consumers 
                if c not in all_replaced and c not in internal_names
            ]
            if external_consumers:
                nodes_with_ext_consumers.append((replaced_name, external_consumers))
        return nodes_with_ext_consumers
    
    def _log_external_consumer_warning(self, nodes_with_ext_consumers):
        """Log warning about nodes with external consumers."""
        logging.warning("Nodes marked as replaced still have external consumers:")
        for replaced_name, ext_consumers in nodes_with_ext_consumers[:3]:
            consumer_list = ', '.join(ext_consumers[:5])
            if len(ext_consumers) > 5:
                consumer_list += f" and {len(ext_consumers) - 5} more..."
            logging.warning(f"  - {replaced_name}: consumed by {consumer_list}")
        if len(nodes_with_ext_consumers) > 3:
            logging.warning(f"  ... and {len(nodes_with_ext_consumers) - 3} more nodes")

    def _final_prune(self, graph_def, pass_name=None, protected_nodes=None):
        """
        Final cleanup pass to remove all remaining dead nodes.
        Iteratively removes nodes with zero references until no more dead nodes exist.
        """
        protected_nodes = protected_nodes or set()
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            refs = self._compute_reference_counts(graph_def)
            
            dead_nodes = {
                node.name for node in graph_def.node
                if node.op != "Placeholder" 
                and node.name not in protected_nodes
                and refs[node.name] == 0
            }
            
            if not dead_nodes:
                # No more dead nodes to remove
                break
            
            logging.info(
                f"[{pass_name or 'optimize'}] Final pruning iteration {iteration + 1}: "
                f"removing {len(dead_nodes)} dead nodes: "
                f"{', '.join(list(dead_nodes)[:5])}"
                + (f" and {len(dead_nodes) - 5} more..." if len(dead_nodes) > 5 else "")
            )
            
            graph_def = self._remove_nodes(graph_def, dead_nodes)
            iteration += 1
        
        if iteration >= max_iterations:
            logging.warning(
                f"[{pass_name or 'optimize'}] Final pruning reached max iterations ({max_iterations})"
            )
        elif iteration > 0:
            logging.info(
                f"[{pass_name or 'optimize'}] Final pruning completed in {iteration} iteration(s)"
            )
        
        return graph_def

    def _prune_dead_nodes(self, graph_def, pass_name=None, refs_before=None, protected_nodes=None):
        """Remove nodes that became dead after optimization."""
        refs_after = self._compute_reference_counts(graph_def)
        protected_nodes = protected_nodes or set()

        dead_nodes = set()
        for node in graph_def.node:
            if node.op == "Placeholder" or node.name in protected_nodes:
                continue

            # Always prune unreferenced Const nodes
            if node.op == "Const" and refs_after[node.name] == 0:
                dead_nodes.add(node.name)
                continue

            # Prune nodes that became dead (had refs before, now don't)
            if refs_before and node.name in refs_before:
                if refs_before[node.name] > 0 and refs_after[node.name] == 0:
                    dead_nodes.add(node.name)

        if dead_nodes:
            logging.info(
                f"[{pass_name or 'optimize'}] Pruning {len(dead_nodes)} dead nodes: "
                f"{', '.join(sorted(list(dead_nodes)[:5]))}"
                + (f" and {len(dead_nodes) - 5} more..." if len(dead_nodes) > 5 else "")
            )
            return self._remove_nodes(graph_def, dead_nodes)
        
        return graph_def
    
    def _remove_nodes(self, graph_def, nodes_to_remove):
        """Create new GraphDef without specified nodes."""
        pruned_graph_def = tf.GraphDef()
        for node in graph_def.node:
            if node.name not in nodes_to_remove:
                pruned_graph_def.node.add().CopyFrom(node)
            else:
                # 为每个被删除的节点添加日志
                logging.debug(f"[GraphOptimizer] Removing node: {node.name} (op: {node.op})")
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

    def __init__(self, name=None, optimizer_alias=None):
        """
        Initialize a pass.
        
        Args:
            name: Human-readable pass name (defaults to class name)
            optimizer_alias: Short alias for node naming (e.g., 'pack_hoist', 'concat_fuse')
                           If not provided, defaults to a simplified version of name
        """
        self.name = name or self.__class__.__name__
        self.optimizer_alias = optimizer_alias or self._generate_default_alias()
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
        
        This is an alias for _extract_base_name for compatibility with subclasses.
        
        Args:
            input_name: Input name (may contain :port or ^ prefix)
            
        Returns:
            str: Cleaned base node name
            
        Examples:
            'node:0' -> 'node'
            '^control_dep' -> 'control_dep'
            'node:1' -> 'node'
            'node' -> 'node'
        """
        return input_name.split(':')[0].lstrip('^')
    
    @staticmethod
    def extract_key_attrs(attrs, op_type=None):
        """
        提取关键属性用于节点签名（排除形状等运行时属性）。
        
        跳过的属性：_output_shapes, _class
        对于大多数节点，T 和 dtype 通常是推断出来的，不影响语义等价性。
        但对于 Const 节点，dtype 是关键属性，必须包含在签名中。
        
        Args:
            attrs: 属性字典（AttrValue对象）
            op_type: 节点操作类型（用于特殊处理某些操作）
            
        Returns:
            tuple: 属性签名元组 (attr_name, type, value)
            
        Examples:
            {'axis': AttrValue(i=0), 'T': AttrValue(type=DT_FLOAT)}
            -> (('axis', 'i', 0),)
            
            Const 节点会包含 dtype:
            {'value': tensor, 'dtype': DT_INT32}
            -> (('dtype', 'type', DT_INT32), ('value', 'tensor', <bytes>))
        """
        key_attrs = []
        # 基础跳过属性
        skip_attrs = {'_output_shapes', '_class'}
        
        # 对于非 Const 节点，额外跳过 T 和 dtype
        if op_type != 'Const':
            skip_attrs.update({'T', 'dtype'})
        
        for attr_name in sorted(attrs.keys()):
            if attr_name in skip_attrs:
                continue
            
            attr_value = attrs[attr_name]
            # 简化属性值表示
            if attr_value.HasField('i'):
                key_attrs.append((attr_name, 'i', attr_value.i))
            elif attr_value.HasField('f'):
                key_attrs.append((attr_name, 'f', attr_value.f))
            elif attr_value.HasField('b'):
                key_attrs.append((attr_name, 'b', attr_value.b))
            elif attr_value.HasField('s'):
                key_attrs.append((attr_name, 's', attr_value.s))
            elif attr_value.HasField('type'):
                # 数据类型（对 Const 节点很重要）
                key_attrs.append((attr_name, 'type', attr_value.type))
            elif attr_value.HasField('tensor'):
                # 对于 tensor 类型（Const 节点的 value 属性）
                # 将 tensor 序列化为字节串作为签名，确保相同值的常量有相同签名
                tensor = attr_value.tensor
                tensor_bytes = tensor.SerializeToString()
                key_attrs.append((attr_name, 'tensor', tensor_bytes))
        
        return tuple(key_attrs)
    
    def create_node_signature(self, node, sort_inputs=True):
        """
        创建节点签名用于识别语义相同的节点。
        
        签名包括：操作类型、输入列表、关键属性
        
        对于 Const 节点，会包含 dtype 和 value 进行比较。
        
        Args:
            node: tf.NodeDef 节点
            sort_inputs: 是否排序输入（对于可交换操作可以设为True）
            
        Returns:
            tuple: (op_type, inputs_tuple, key_attrs)
            
        Examples:
            Pack([a, b, c], axis=0) -> ('Pack', ('a', 'b', 'c'), (('axis', 'i', 0),))
            Const(value=16, dtype=int32) -> ('Const', (), (('dtype', 'type', DT_INT32), ('value', 'tensor', <bytes>)))
        """
        # 清理输入名称
        cleaned_inputs = [self.clean_input_name(inp) for inp in node.input]
        
        # 根据需要排序输入
        if sort_inputs:
            inputs_tuple = tuple(sorted(cleaned_inputs))
        else:
            inputs_tuple = tuple(cleaned_inputs)
        
        # 提取关键属性（传入 op_type 以便特殊处理）
        key_attrs = self.extract_key_attrs(node.attr, op_type=node.op)
        
        return (node.op, inputs_tuple, key_attrs)
    
    def get_or_create_cached_node(self, op_type, inputs, attrs, root_node_name, 
                                   context_desc="", create_func=None):
        """
        获取或创建缓存节点的统一接口（用于节点去重）。
        
        工作原理：
        1. 根据 (op_type, inputs, attrs) 创建签名
        2. 如果签名已存在于缓存，返回缓存的节点名
        3. 否则创建新节点，加入缓存，返回新节点
        
        Args:
            op_type: 操作类型
            inputs: 输入列表（节点名称列表）
            attrs: 属性字典（AttrValue对象）
            root_node_name: 根节点名称（用于生成唯一名称）
            context_desc: 上下文描述（用于日志）
            create_func: 可选的节点创建函数 func(name, inputs, attrs) -> NodeDef
                        如果为None，则返回None作为new_node
            
        Returns:
            tuple: (node_name, is_new_node, node_def_or_none)
                - node_name: 节点名称（缓存命中时是已存在的名称）
                - is_new_node: 是否是新创建的节点
                - node_def_or_none: 如果是新节点返回NodeDef，否则返回None
        """
        from .utils import create_node
        
        # 创建节点签名（输入保持顺序）
        inputs_tuple = tuple(inputs)
        key_attrs = self.extract_key_attrs(attrs, op_type=op_type)
        node_signature = (op_type, inputs_tuple, key_attrs)
        
        # 检查缓存
        if node_signature in self._node_cache:
            cached_name = self._node_cache[node_signature]
            if context_desc:
                logging.info(f"[{self.name}] {context_desc}: REUSING cached {op_type} node {cached_name}")
            return cached_name, False, None
        
        # 创建新节点
        new_name = self.make_unique_node_name(root_node_name, op_type)
        
        if create_func:
            new_node = create_func(new_name, inputs, attrs)
        else:
            new_node = create_node(op_type, new_name, inputs=inputs, attr=attrs)
        
        # 缓存节点
        self._node_cache[node_signature] = new_name
        if context_desc:
            logging.debug(f"[{self.name}] {context_desc}: created new {op_type} node {new_name}")
        
        return new_name, True, new_node
    
    def build_deduplication_map(self, optimizer, skip_ops=None, protected_nodes=None):
        """
        构建全局去重映射，识别图中语义相同的重复节点。
        
        Args:
            optimizer: GraphOptimizer 实例
            skip_ops: 要跳过的操作类型集合（这些节点不应去重）
                     默认跳过：Placeholder, Variable, VariableV2, Identity
                     注意：Const 节点默认不跳过，可以根据 value 和 dtype 去重
            protected_nodes: 受保护的节点集合（这些节点不会被删除，但可以作为规范节点）
            
        Returns:
            dict: {duplicate_node_name -> canonical_node_name}
            
        Examples:
            如果图中有两个完全相同的 MatMul 节点：
            - MatMul_1(a, b) 
            - MatMul_2(a, b)  # 重复
            返回: {'MatMul_2': 'MatMul_1'}
        """
        from collections import defaultdict
        
        if skip_ops is None:
            skip_ops = {'Placeholder', 'Variable', 'VariableV2', 'Identity'}
        
        protected_set = set(protected_nodes or [])
        
        # 按节点签名分组
        nodes_by_signature = defaultdict(list)
        
        for node in optimizer.graph_def.node:
            # 跳过某些不应该去重的节点类型
            if node.op in skip_ops:
                continue
            
            # 创建签名时保留控制依赖标记（对于CSE很重要）
            signature = self._create_cse_signature(node)
            nodes_by_signature[signature].append(node.name)
        
        # 构建去重映射
        dedup_map = {}
        
        for signature, node_names in nodes_by_signature.items():
            if len(node_names) <= 1:
                continue
            
            # 选择规范节点：优先选择受保护的节点，然后是名字最短的
            protected_candidates = [n for n in node_names if n in protected_set]
            
            if protected_candidates:
                # 如果有受保护的节点，从中选择名字最短的作为规范节点
                canonical = min(protected_candidates, key=lambda n: (len(n), n))
            else:
                # 否则选择名字最短的作为规范节点
                canonical = min(node_names, key=lambda n: (len(n), n))
            
            # 将所有非规范节点（且非受保护节点）映射到规范节点
            for node_name in node_names:
                if node_name != canonical and node_name not in protected_set:
                    dedup_map[node_name] = canonical
        
        return dedup_map
    
    def _create_cse_signature(self, node):
        """
        为 CSE 创建节点签名，保留控制依赖标记。
        
        与 create_node_signature 的区别：
        - create_node_signature: 去除端口和控制依赖标记，用于一般模式匹配
        - _create_cse_signature: 保留控制依赖标记，用于 CSE 精确去重
        
        这样可以区分：
        - Add(a, b) 和 Add(a, b, ^ctrl) 是不同的节点
        - Add(a, b, ^ctrl1) 和 Add(a, b, ^ctrl2) 是不同的节点
        
        Args:
            node: tf.NodeDef 节点
            
        Returns:
            tuple: (op_type, inputs_tuple_with_ctrl_deps, key_attrs)
        """
        # 对于每个输入，只去除端口后缀，保留控制依赖前缀
        cleaned_inputs = []
        for inp in node.input:
            # 保留控制依赖前缀 ^
            if inp.startswith('^'):
                cleaned_inputs.append(inp)  # 保留 ^node
            elif ':' in inp:
                base_name = inp.split(':', 1)[0]  # 去除端口
                cleaned_inputs.append(base_name)
            else:
                cleaned_inputs.append(inp)
        
        # 输入保持原始顺序（不排序）
        inputs_tuple = tuple(cleaned_inputs)
        
        # 提取关键属性
        key_attrs = self.extract_key_attrs(node.attr, op_type=node.op)
        
        return (node.op, inputs_tuple, key_attrs)
    
    def apply_deduplication_map(self, optimizer, dedup_map):
        """
        应用去重映射：更新所有节点的输入引用，删除重复节点。
        
        工作流程：
        1. 遍历所有节点，更新输入引用（将重复节点替换为规范节点）
        2. 删除映射中的重复节点
        3. 重建优化器的内部索引
        
        Args:
            optimizer: GraphOptimizer 实例
            dedup_map: 去重映射 {duplicate_node_name -> canonical_node_name}
        """
        removed_nodes = set(dedup_map.keys())
        
        # 更新所有节点的输入引用
        for node in optimizer.graph_def.node:
            if node.name in removed_nodes:
                continue
            
            new_inputs = []
            for inp in node.input:
                # 提取基础名称（去除端口和控制依赖标记）
                port_suffix = ''
                control_prefix = ''
                
                if inp.startswith('^'):
                    control_prefix = '^'
                    inp = inp[1:]
                
                if ':' in inp:
                    base_name, port = inp.split(':', 1)
                    port_suffix = ':' + port
                else:
                    base_name = inp
                
                # 应用映射
                if base_name in dedup_map:
                    new_base = dedup_map[base_name]
                    new_inp = control_prefix + new_base + port_suffix
                    new_inputs.append(new_inp)
                else:
                    new_inputs.append(control_prefix + base_name + port_suffix)
            
            # 更新节点的输入列表
            del node.input[:]
            node.input.extend(new_inputs)
        
        # 删除重复节点（为每个节点添加日志）
        for node in optimizer.graph_def.node:
            if node.name in removed_nodes:
                canonical = dedup_map[node.name]
                logging.info(f"[{self.name}] Removing duplicate node: {node.name} (op: {node.op}) -> keeping {canonical}")
        
        new_nodes = [n for n in optimizer.graph_def.node if n.name not in removed_nodes]
        del optimizer.graph_def.node[:]
        optimizer.graph_def.node.extend(new_nodes)
        
        # 重建优化器内部状态
        optimizer.nodes = {node.name: node for node in optimizer.graph_def.node}
        optimizer._rebuild_consumer_index()

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

    def __init__(self, pattern, rewriter, name=None, optimizer_alias=None):
        super().__init__(name, optimizer_alias)
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
        # Reset node counters at the start of each transform
        self.reset_counters()
        
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
