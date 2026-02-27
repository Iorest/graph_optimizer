import tensorflow.compat.v1 as tf
import collections
from typing import Dict, List, Optional, Any as AnyType, Tuple, TYPE_CHECKING

from ...utils.logger import tf_logger as logging

if TYPE_CHECKING:
    from .tf_optimizer import TFGraphOptimizer
    from .pattern import Pattern


class MatchContext:
    def __init__(self):
        self.matched_nodes = {}  # alias -> NodeDef or list of NodeDef
        self.all_matched_nodes = set()  # set of node names
        self.control_inputs = set()  # set of "^node_name"


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
        node_mapping: Optional[Dict[str, str]] = None,
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
    - Graph state management (handled by TFGraphOptimizer)
    """

    def __init__(self):
        self.pattern_index: Dict[str, List[Tuple["Pattern", AnyType]]] = (
            collections.defaultdict(list)
        )
        self.wildcard_patterns: List[Tuple["Pattern", AnyType]] = []

        # Stability: Ops that should block most simplifications/foldings
        self.stateful_ops = {
            "Variable",
            "VariableV2",
            "TemporaryVariable",
            "RandomUniform",
            "RandomStandardNormal",
            "TruncatedNormal",
            "Assign",
            "AssignAdd",
            "AssignSub",
            "Placeholder",
            "PlaceholderV2",
            "PlaceholderWithDefault",
            "IteratorGetNext",
            "QueueDequeue",
            "QueueDequeueV2",
        }
        self.control_flow_ops = {"Switch", "Merge", "Enter", "Exit", "NextIteration"}

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
        optimizer: "TFGraphOptimizer",
        pass_name: str = None,
        auto_cleanup: bool = True,
        protected_nodes: set = None,
    ):
        """
        Run a single iteration of pattern matching.

        Args:
            optimizer: TFGraphOptimizer with current graph state
            pass_name: Pass name for logging
            auto_cleanup: If True, prune dead nodes after matching
            protected_nodes: Nodes that should not be pruned

        Returns:
            tuple: (new_graph_def, changes_count)
        """
        protected_nodes = set(protected_nodes or [])
        optimizer.protected_nodes = protected_nodes
        optimizer.current_pass_name = pass_name  # Set for logging in Pattern.match

        nodes_before = len(optimizer.nodes)
        prefix = f"[{pass_name}] " if pass_name else ""

        new_nodes = []
        replaced_node_names = set()
        added_node_names = []  # Track newly added nodes for logging
        global_node_mapping = {}
        hoisted_controls_map = collections.defaultdict(
            set
        )  # node_name -> set(controls)
        modified = False

        for node in optimizer.graph_def.node:
            if node.name in replaced_node_names:
                continue

            candidates = self.pattern_index.get(node.op, []) + self.wildcard_patterns

            found_match = False
            for pattern, rewriter in candidates:
                # Stability: Don't fold stateful ops unless explicitly handled
                if node.op in self.stateful_ops and pass_name not in ["CSE", "prune"]:
                    continue

                match = pattern.match(node, optimizer)
                if match:
                    rewriter_output = rewriter(match, optimizer)
                    if rewriter_output is not None:
                        result = RewriteResult.from_nodes(rewriter_output)
                        if self._process_match_result(
                            node,
                            match,
                            result,
                            optimizer,
                            prefix,
                            replaced_node_names,
                            new_nodes,
                            added_node_names,
                            global_node_mapping,
                            hoisted_controls_map,
                        ):
                            found_match = True
                            modified = True
                        break

            if not found_match:
                new_nodes.append(node)

        if global_node_mapping:
            modified = True

        if not modified:
            return optimizer.graph_def.node, 0

        # Log newly added nodes
        for node_name, node_op in added_node_names:
            logging.info(f"{prefix}Added: {node_name} (op: {node_op})")

        self._apply_mappings_and_controls(
            new_nodes, global_node_mapping, hoisted_controls_map, optimizer
        )

        # Return the raw node list. The optimizer will rebuild the GraphDef.
        # Log iteration summary
        modes_after = len(new_nodes)
        node_diff = nodes_before - modes_after
        logging.info(
            f"{prefix}Summary: {nodes_before} -> {modes_after} nodes "
            f"(replaced: {len(replaced_node_names)}, added: {len(added_node_names)}, diff: -{node_diff})"
        )

        return new_nodes, len(replaced_node_names)

    def _process_match_result(
        self,
        node: tf.NodeDef,
        match: "MatchContext",
        result: RewriteResult,
        optimizer: "TFGraphOptimizer",
        prefix: str,
        replaced_node_names: set,
        new_nodes: list,
        added_node_names: list,
        global_node_mapping: dict,
        hoisted_controls_map: dict,
    ) -> bool:
        """Processes the result of a successful pattern match."""
        # Centralized protection: ensure model outputs/protected nodes are preserved
        is_protected = node.name in optimizer.protected_nodes
        if is_protected:
            swallowed = self._handle_protected_nodes(node, result, new_nodes)
            if swallowed:
                # Still want to update mappings so consumers use the base node!
                if result.node_mapping:
                    global_node_mapping.update(result.node_mapping)
                return False

        self._handle_control_dependencies(node, match, result, hoisted_controls_map)

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
            internal_names = match.all_matched_nodes
            all_replaced = {node.name} | set(result.replaced_nodes)
            nodes_with_ext_consumers = optimizer.check_external_consumers(
                result.replaced_nodes, all_replaced, internal_names
            )

            if nodes_with_ext_consumers:
                optimizer.log_external_consumer_warning(nodes_with_ext_consumers)
                safe_to_replace = [
                    name
                    for name in result.replaced_nodes
                    if name not in [n for n, _ in nodes_with_ext_consumers]
                ]
                replaced_node_names.update(safe_to_replace)
            else:
                replaced_node_names.update(result.replaced_nodes)

        return True

    def _handle_control_dependencies(
        self,
        node: tf.NodeDef,
        match: "MatchContext",
        result: RewriteResult,
        hoisted_controls_map: dict,
    ):
        """Preserve external control dependencies from matched nodes."""
        internal_names = match.all_matched_nodes
        relevant_controls = [
            ci for ci in match.control_inputs if ci.lstrip("^") not in internal_names
        ]

        if not relevant_controls:
            return

        mapped_target = result.node_mapping.get(node.name)
        new_node_names = {n.name for n in result.new_nodes}

        # Case: consumers remapped to something else (existing or new)
        if mapped_target and mapped_target not in new_node_names:
            hoisted_controls_map[node.name].update(relevant_controls)

        # Case: new nodes created (may or may not be remapped)
        if result.new_nodes:
            target_node = result.new_nodes[0]
            for new_node in result.new_nodes:
                if new_node.name == node.name:
                    target_node = new_node
                    break
            if target_node:
                existing = set(target_node.input)
                for ci in relevant_controls:
                    # Prevention: do not add self-control-dependency cycle
                    if ci.lstrip("^") == target_node.name:
                        continue
                    if ci not in existing:
                        target_node.input.append(ci)
                        existing.add(ci)
        elif not mapped_target:
            # Fallback
            hoisted_controls_map[node.name].update(relevant_controls)

    def _apply_mappings_and_controls(
        self,
        new_nodes: list,
        global_node_mapping: dict,
        hoisted_controls_map: dict,
        optimizer: "TFGraphOptimizer",
    ):
        """Applies node mappings and hoisted control dependencies to the new graph."""
        if not global_node_mapping and not hoisted_controls_map:
            return

        logging.debug(
            f"Applying node mapping and control hoisting: "
            f"{len(global_node_mapping)} remappings, {len(hoisted_controls_map)} hoisted"
        )
        from graph_optimizer.utils.tf.graph_utils import extract_base_name

        for node in new_nodes:
            node_hoisted = set()
            if hoisted_controls_map:
                for input_name in node.input:
                    base_name = extract_base_name(input_name)
                    if base_name in hoisted_controls_map:
                        node_hoisted.update(hoisted_controls_map[base_name])

            optimizer.update_node_inputs(
                node, global_node_mapping, node_hoisted or None
            )

    def _handle_protected_nodes(
        self, node: tf.NodeDef, result: RewriteResult, new_nodes_list: list
    ) -> bool:
        """
        Ensures protected nodes (e.g. outputs) preserve their identity after rewrite.
        Returns: True if the rewrite was 'swallowed' (no change caused), False otherwise.
        """
        mapped_target = result.node_mapping.get(node.name)

        # Case 1: Node replaced by new nodes -> the first new node must inherit the protected name
        if result.new_nodes:
            primary_new = result.new_nodes[0]
            if primary_new.name != node.name:
                # Update renaming map for subsequent nodes if necessary
                result.node_mapping[primary_new.name] = (
                    primary_new.name
                )  # Identity fallback
                primary_new.name = node.name
                # Ensure the remapping points to this renamed node
                result.node_mapping[node.name] = node.name
            return False

        # Case 2: Node remapped to an existing node (e.g. Identity elimination)
        # We must insert an Identity node to preserve the protected name
        elif mapped_target and mapped_target != node.name:
            # Check if this node is already an Identity pointing to the same target.
            # If so, the simplify rule matched it but we shouldn't re-inject the same identity.
            if node.op == "Identity" and node.input and node.input[0] == mapped_target:
                # We KEEP the mapping to help consumers, but we don't re-inject
                # a redundant identity, nor do we count this as a change for the loop.
                return True

            from graph_optimizer.utils.tf.graph_utils import (
                create_node,
                make_type_attr,
                get_node_dtype,
            )

            dtype = get_node_dtype(node)
            attr = {"T": make_type_attr(dtype)}
            proxy = create_node(
                "Identity", node.name, inputs=[mapped_target], attr=attr
            )
            result.new_nodes.append(proxy)
            result.node_mapping[node.name] = node.name
            return False

        return False
