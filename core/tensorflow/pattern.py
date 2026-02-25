from typing import Optional, TYPE_CHECKING
import tensorflow.compat.v1 as tf

from ...utils.logger import log_match
from .matcher import MatchContext
from .graph import get_attr_value

if TYPE_CHECKING:
    from .tf_optimizer import TFGraphOptimizer
    from .matcher import MatchContext


def Op(
    op_type,
    *inputs,
    alias=None,
    attrs=None,
    shape=None,
    consumer_count=None,
    commutative=False,
):
    pattern = OpPattern(
        op_type,
        list(inputs),
        attrs=attrs,
        shape=shape,
        alias=alias,
        commutative=commutative,
    )
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


def CommutativeOp(
    op_type, p1, p2, alias=None, attrs=None, shape=None, consumer_count=None
):
    """Matches an Op where the order of the first two inputs doesn't matter."""
    pattern = OpPattern(
        op_type, list([p1, p2]), attrs=attrs, shape=shape, alias=alias, commutative=True
    )
    pattern.consumer_count = consumer_count
    return pattern


def ConstValue(value, alias=None):
    """Matches a Const node with a specific value."""

    def check_value(unwrapped_value):
        return unwrapped_value == value

    return Op("Const", attrs={"value": check_value}, alias=alias)


class Pattern:
    def __init__(self, alias=None):
        self.alias = alias
        self.consumer_count = None  # Expected number of consumers (None = any)

    @log_match
    def match(
        self,
        node: tf.NodeDef,
        optimizer: "TFGraphOptimizer",
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
        if self.alias and self.alias in context.matched_nodes:
            # If alias is already bound, this node must be the identical node
            bound_node = context.matched_nodes[self.alias]
            if isinstance(bound_node, list):
                # Variadic accumulation (handled differently, but just in case)
                if node not in bound_node:
                    return False
            else:
                if node.name != bound_node.name:
                    return False

        res = self._do_match(node, optimizer, context)
        if res:
            context.all_matched_nodes.add(node.name)
            # Accumulate ALL control dependencies from all matched nodes
            for input_name in node.input:
                if input_name.startswith("^"):
                    context.control_inputs.add(input_name)

            if self.alias and self.alias not in context.matched_nodes:
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
    def __init__(
        self,
        op_type,
        inputs=None,
        attrs=None,
        shape=None,
        alias=None,
        commutative=False,
    ):
        super().__init__(alias)
        self.op_type = op_type
        self.inputs = inputs or []  # List of Pattern
        self.attrs = attrs or {}  # Map of attr_name -> attr_value (or predicate)
        self.shape = shape  # Expected output shape (list of ints or None for wildcard)
        self.consumer_count = None  # Expected number of consumers
        self.commutative = commutative

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
                actual = get_attr_value(actual_attr)

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

                if self.commutative and len(data_inputs) == 2 and len(self.inputs) == 2:
                    # Backup context before trying first permutation
                    ctx_backup = MatchContext()
                    ctx_backup.matched_nodes = dict(context.matched_nodes)
                    ctx_backup.all_matched_nodes = set(context.all_matched_nodes)
                    ctx_backup.control_inputs = set(context.control_inputs)

                    match_orig = self._match_single_input(
                        data_inputs[0], self.inputs[0], optimizer, context
                    ) and self._match_single_input(
                        data_inputs[1], self.inputs[1], optimizer, context
                    )

                    if not match_orig:
                        # Restore context and try swapped permutation
                        context.matched_nodes = ctx_backup.matched_nodes
                        context.all_matched_nodes = ctx_backup.all_matched_nodes
                        context.control_inputs = ctx_backup.control_inputs

                        match_swapped = self._match_single_input(
                            data_inputs[0], self.inputs[1], optimizer, context
                        ) and self._match_single_input(
                            data_inputs[1], self.inputs[0], optimizer, context
                        )
                        if not match_swapped:
                            return False
                else:
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


class OptionalPattern(Pattern):
    """
    Matches a pattern if present, otherwise skips it and matches its first input pattern.
    Assumes the wrapped pattern wraps a single main data flow path.
    """

    def __init__(self, pattern, alias=None):
        super().__init__(alias)
        self.pattern = pattern

    def _do_match(self, node, optimizer, context):
        ctx_backup = MatchContext()
        ctx_backup.matched_nodes = dict(context.matched_nodes)
        ctx_backup.all_matched_nodes = set(context.all_matched_nodes)
        ctx_backup.control_inputs = set(context.control_inputs)

        # Try matching the wrapped pattern
        if self.pattern._match_internal(node, optimizer, context):
            return True

        # Restore context and try to skip this node
        context.matched_nodes = ctx_backup.matched_nodes
        context.all_matched_nodes = ctx_backup.all_matched_nodes
        context.control_inputs = ctx_backup.control_inputs

        # Fallback to matching the node against the first input pattern
        if hasattr(self.pattern, "inputs") and len(self.pattern.inputs) > 0:
            return self.pattern.inputs[0]._match_internal(node, optimizer, context)

        return False


class MultiOutputPattern:
    """
    Matches a subgraph with multiple output nodes (sinks).
    """

    def __init__(self, output_patterns, alias=None):
        self.output_patterns = output_patterns
        self.alias = alias

    def get_indexed_op_type(self):
        """Index by the first output pattern's op type to anchor the search."""
        return self.output_patterns[0].get_indexed_op_type()

    def match(self, node, optimizer):
        context = MatchContext()
        # Anchor the match on the first output pattern
        anchor_pattern = self.output_patterns[0]
        if not anchor_pattern._match_internal(node, optimizer, context):
            return None

        # Determine the matched inner nodes mapping, then verify remaining outputs
        # To do this robustly without arbitrary graph matching,
        # we expect the anchor match to have bound aliases for internal shared nodes.
        # Then, we can evaluate the remaining output patterns starting from their known aliases.

        # If there are NO other outputs, returning now
        if len(self.output_patterns) == 1:
            if self.alias:
                context.matched_nodes[self.alias] = node
            return context

        # For remaining outputs, they must either be bound by an alias in context,
        # or we must be able to find them by tracing from a bound alias forward.
        # A simple constraint: remaining output_patterns must be alias bounds evaluated lazily,
        # or we sweep consumer edges. For now, require they have an alias that was bound
        # during the anchor match, and we just verify the node matches.

        for i in range(1, len(self.output_patterns)):
            pattern = self.output_patterns[i]
            if pattern.alias and pattern.alias in context.matched_nodes:
                continue

            # Optimized search: only look at descendants of already matched nodes.
            # A multi-output subgraph must have connected components.
            candidates = set()
            search_queue = list(context.all_matched_nodes)
            visited = set(search_queue)

            # BFS to find all connected nodes in the consumer direction
            while search_queue:
                curr_name = search_queue.pop(0)
                for consumer_name in optimizer.consumers.get(curr_name, []):
                    if consumer_name not in visited:
                        visited.add(consumer_name)
                        candidates.add(consumer_name)
                        # Limit search depth or total candidates if necessary?
                        # For now, following all descendants is $O(\text{Subgraph})$.
                        search_queue.append(consumer_name)

            matched_sink = False
            # Sort candidates to ensure deterministic matching order if multiple match
            for cand_name in sorted(candidates):
                cand_node = optimizer.nodes[cand_name]
                ctx_backup = MatchContext()
                ctx_backup.matched_nodes = dict(context.matched_nodes)
                ctx_backup.all_matched_nodes = set(context.all_matched_nodes)
                ctx_backup.control_inputs = set(context.control_inputs)

                if pattern._match_internal(cand_node, optimizer, context):
                    matched_sink = True
                    break
                else:
                    context.matched_nodes = ctx_backup.matched_nodes
                    context.all_matched_nodes = ctx_backup.all_matched_nodes
                    context.control_inputs = ctx_backup.control_inputs

            if not matched_sink:
                return None

        if self.alias:
            context.matched_nodes[self.alias] = node
        return context


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
