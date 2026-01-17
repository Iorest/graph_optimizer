"""
Pack Hoisting Optimization Pass.

Transforms: Input -> Split -> [Op1, Op2, ...] -> Pack -> NextOp
Into: Input -> Op (batched) -> NextOp

Key insight: Many operations are naturally batched:
    Pack([Op(x1), Op(x2), ...], axis=k) = Op(Pack([x1, x2, ...], axis=k))
"""

from ..core import PatternRewritePass, PassRegistry, RewriteResult
from ..utils import create_node
from ..utils.logger import logger as logging
from tensorflow.core.framework import tensor_shape_pb2, attr_value_pb2

# Element-wise operations (dimension-agnostic)
ELEMENTWISE_OPS = {
    'Add', 'AddV2', 'Sub', 'Mul', 'Div', 'RealDiv', 'Maximum', 'Minimum',
    'Relu', 'Relu6', 'Elu', 'Selu', 'Sigmoid', 'Tanh', 'Softplus', 'Erf',
    'Square', 'Sqrt', 'Rsqrt', 'Exp', 'Log', 'Neg', 'Abs',
    'Cast', 'Identity',
}

# Operations that can handle batched inputs
BATCH_AWARE_OPS = {
    'BiasAdd',
    'MatMul',
    'BatchMatMul', 'BatchMatMulV2',
}

# All hoistable operations
HOISTABLE_OPS = ELEMENTWISE_OPS | BATCH_AWARE_OPS

# Operations that prevent hoisting
BLOCKING_OPS = {
    'Reshape', 'Transpose', 'ExpandDims', 'Squeeze',
    'ReduceMean', 'ReduceSum', 'ReduceMax', 'ReduceMin',
    'ArgMax', 'ArgMin', 'TopK',
    'Concat', 'ConcatV2', 'Split', 'SplitV',
}


class OpConverter:
    """
    Unified operator conversion handler for Pack hoisting.
    
    Handles conversions like:
    - MatMul -> BatchMatMulV2 (when Pack adds batch dimension)
    - BiasAdd -> AddV2 (when bias varies across branches)
    """
    
    # Conversion rules: (original_op, target_op, attr_mapping, attrs_to_remove)
    CONVERSIONS = {
        'MatMul': {
            'target': 'BatchMatMulV2',
            'attr_map': {
                'transpose_a': 'adj_x',
                'transpose_b': 'adj_y',
            },
            'remove_attrs': {'transpose_a', 'transpose_b'},
            'pack_input_idx': 1,  # Weight input index that can be packed
        },
        'BiasAdd': {
            'target': 'AddV2',
            'attr_map': {},
            'remove_attrs': {'data_format'},
            'pack_input_idx': 1,  # Bias input index that can be packed
        },
    }
    
    @classmethod
    def needs_conversion(cls, op_type, has_varying_inputs):
        """Check if op needs conversion when hoisting through Pack."""
        return op_type in cls.CONVERSIONS and has_varying_inputs
    
    @classmethod
    def get_target_op(cls, op_type, has_varying_inputs=False):
        """Get target op type after conversion (or original if no conversion needed)."""
        if has_varying_inputs and op_type in cls.CONVERSIONS:
            return cls.CONVERSIONS[op_type]['target']
        return op_type
    
    @classmethod
    def get_pack_input_idx(cls, op_type):
        """Get the input index that can be packed for this op type."""
        if op_type in cls.CONVERSIONS:
            return cls.CONVERSIONS[op_type].get('pack_input_idx')
        return None
    
    @classmethod
    def convert_attrs(cls, op_type, original_attrs, target_op_type=None):
        """
        Convert attributes from original op to target op.
        
        Args:
            op_type: Original operation type
            original_attrs: Original attributes dict
            target_op_type: Target op type (if None, determined by op_type)
            
        Returns:
            New attributes dict suitable for target op
        """
        if op_type not in cls.CONVERSIONS:
            # No conversion needed, just copy (excluding _output_shapes)
            return {k: v for k, v in original_attrs.items() if k != '_output_shapes'}
        
        conv = cls.CONVERSIONS[op_type]
        new_attrs = {}
        
        for key, value in original_attrs.items():
            if key == '_output_shapes':
                continue
            if key in conv['remove_attrs']:
                # Map to new attr name if needed
                if key in conv['attr_map']:
                    new_key = conv['attr_map'][key]
                    new_attrs[new_key] = attr_value_pb2.AttrValue(b=value.b)
            else:
                new_attrs[key] = value
        
        return new_attrs
    
    @classmethod
    def is_hoistable_through(cls, op_type):
        """
        Check if Pack can be hoisted through this op type.
        
        MatMul is NOT safe for hoisting (changes output dimensionality),
        but IS safe for elimination path.
        """
        if op_type == 'MatMul':
            return False  # Only safe in elimination path
        if op_type in BLOCKING_OPS:
            return False
        if op_type not in HOISTABLE_OPS:
            return False
        return True


@PassRegistry.register("pack_hoisting", opt_level=5, priority=60)
class PackHoisting(PatternRewritePass):
    """
    Pack hoisting optimization pass.
    
    Recursively hoists Pack operations through element-wise and batch-aware operations,
    or eliminates them entirely when branches originate from Split/StridedSlice.
    """
    
    def __init__(self):
        from ..core import Op, Variadic, Any
        pattern = Op("Pack", Variadic(Any(), alias="pack_inputs"), alias="pack")
        super().__init__(
            pattern, 
            self.rewrite, 
            name="PackHoisting",
            optimizer_alias="pack_hoist"
        )

    
    def rewrite(self, match, optimizer):
        """Rewrite matched pattern."""
        pack_node = match.matched_nodes["pack"]
        pack_inputs = match.matched_nodes["pack_inputs"]
        
        # Skip if Pack has < 2 inputs
        if len(pack_inputs) < 2:
            return None
        
        logging.debug(f"[PackHoisting] Analyzing {pack_node.name} ({len(pack_inputs)} inputs)")
        
        # Try to hoist the pack (with recursive hoisting until can't continue)
        result = self._try_hoist_pack_recursive(pack_node, pack_inputs, optimizer)
        
        if result:
            hoist_path = result['hoist_path']
            hoist_desc = " -> ".join(hoist_path) if hoist_path else "eliminated"
            logging.info(f"[PackHoisting] Hoisted: {pack_node.name} ({len(hoist_path)} layers: {hoist_desc})")
            
            # Return RewriteResult with replaced_nodes and node_mapping if specified
            if 'replaced_nodes' in result or 'node_mapping' in result:
                return RewriteResult(
                    new_nodes=result['new_nodes'],
                    replaced_nodes=result.get('replaced_nodes', []),
                    node_mapping=result.get('node_mapping', {})
                )
            else:
                return result['new_nodes']
        
        return None
    
    def _try_hoist_pack_recursive(self, pack_node, pack_inputs, optimizer):
        """
        Recursively hoist pack upward until elimination or no more hoisting possible.
        
        Returns:
            dict with 'hoist_path' and 'new_nodes', or None if cannot hoist
        """
        branches = pack_inputs
        
        # Check if all branches are the same operation
        if not self._all_same_op(branches):
            return None
        
        first_op = branches[0].op
        
        # Check if operation is hoistable
        if first_op in BLOCKING_OPS or first_op not in HOISTABLE_OPS:
            return None
        
        # Try elimination first (preferred)
        elimination_result = self._try_eliminate_with_split(pack_node, branches, optimizer)
        if elimination_result:
            return elimination_result
        
        # Try hoisting through current operation
        hoisting_result = self._hoist_through_op(pack_node, branches, first_op, optimizer)
        if not hoisting_result:
            return None
        
        # Recursively hoist newly created Pack nodes
        return self._continue_hoisting(hoisting_result, optimizer)
    
    def _all_same_op(self, branches):
        """Check if all branches have the same operation type."""
        return len(set(b.op for b in branches)) == 1
    
    def _continue_hoisting(self, hoisting_result, optimizer):
        """Continue hoisting recursively for newly created Pack nodes."""
        all_nodes = []
        hoist_path = [hoisting_result['op_type']]
        node_mapping = {}
        replaced_nodes = []
        
        for new_pack in hoisting_result['new_packs']:
            new_pack_inputs = self._get_pack_inputs(new_pack, optimizer)
            
            if not new_pack_inputs or len(new_pack_inputs) < 2:
                all_nodes.append(new_pack)
                continue
            
            # Try recursive hoisting
            recursive_result = self._try_hoist_pack_recursive(new_pack, new_pack_inputs, optimizer)
            
            if recursive_result:
                all_nodes.extend(recursive_result['new_nodes'])
                hoist_path.extend(recursive_result['hoist_path'])
                # 合并递归结果的映射和替换信息
                if 'node_mapping' in recursive_result:
                    node_mapping.update(recursive_result['node_mapping'])
                if 'replaced_nodes' in recursive_result:
                    replaced_nodes.extend(recursive_result['replaced_nodes'])
            else:
                all_nodes.append(new_pack)
        
        all_nodes.append(hoisting_result['hoisted_op'])
        
        # 添加当前层的映射信息
        if 'original_pack_name' in hoisting_result:
            node_mapping[hoisting_result['original_pack_name']] = hoisting_result['hoisted_op'].name
            replaced_nodes.append(hoisting_result['original_pack_name'])
        
        result = {'hoist_path': hoist_path, 'new_nodes': all_nodes}
        if node_mapping:
            result['node_mapping'] = node_mapping
        if replaced_nodes:
            result['replaced_nodes'] = replaced_nodes
        
        return result
    
    def _get_pack_inputs(self, pack_node, optimizer):
        """Get input nodes for a Pack node."""
        inputs = []
        for inp_name in pack_node.input:
            inp_name_clean = self.clean_input_name(inp_name)
            inp_node = optimizer.nodes.get(inp_name_clean)
            if inp_node:
                inputs.append(inp_node)
        return inputs if len(inputs) == len(pack_node.input) else []
    
    def _try_eliminate_with_split(self, pack_node, branches, optimizer):
        """Try to eliminate Pack when branches come from common Split/StridedSlice source."""
        # Find Split/StridedSlice sources for all branches
        split_sources = [
            self._find_split_or_slice(branch, optimizer, max_depth=20)
            for branch in branches
        ]
        
        if not all(split_sources):
            return None
        
        # Get common source node
        common_source = self._get_common_source(split_sources)
        if not common_source:
            return None
        
        # Verify operations can be batched (same weights/biases)
        if not self._can_batch_operations(branches, optimizer):
            return None
        
        logging.debug(f"[PackHoisting] {pack_node.name}: can eliminate with {split_sources[0]['type']}")
        return self._generate_batched_ops(pack_node, branches, common_source, optimizer)
    
    def _get_common_source(self, split_sources):
        """Extract common source from split nodes."""
        common_source = None
        for split_node in split_sources:
            try:
                if split_node['type'] == 'StridedSlice':
                    if len(split_node['node'].input) < 1:
                        return None
                    source = self.clean_input_name(split_node['node'].input[0])
                elif split_node['type'] == 'Split':
                    # Split: input[0] is split_dim, input[1] is the value to split
                    if len(split_node['node'].input) < 2:
                        return None
                    source = self.clean_input_name(split_node['node'].input[1])
                else:
                    return None
                
                if common_source is None:
                    common_source = source
                elif common_source != source:
                    return None
            except (AttributeError, TypeError, KeyError) as e:
                logging.warning(f"[PackHoisting] Error extracting source: {e}")
                return None
        
        return common_source
    
    def _can_batch_operations(self, branches, optimizer):
        """Check if operations can be batched.
        
        Rules:
        - For most ops: different inputs must trace to Split/StridedSlice
        - MatMul/BiasAdd with different weights/biases are NOT supported in elimination path
          (they require special handling that changes the graph structure significantly)
        """
        if not branches:
            return False
        
        first_op = branches[0]
        op_type = first_op.op
        num_inputs = len(first_op.input)
        
        for input_idx in range(num_inputs):
            inputs_at_idx = [
                self.clean_input_name(branch.input[input_idx])
                for branch in branches
                if input_idx < len(branch.input)
            ]
            
            if len(inputs_at_idx) != len(branches):
                return False  # Input count mismatch
            
            # If input varies across branches
            if len(set(inputs_at_idx)) > 1:
                # For other cases, verify it traces to Split/StridedSlice
                for inp_name in set(inputs_at_idx):
                    inp_node = optimizer.nodes.get(inp_name)
                    if inp_node and inp_node.op not in ('StridedSlice', 'Split', 'SplitV'):
                        if not self._find_split_or_slice(inp_node, optimizer, max_depth=20):
                            return False
        
        return True
    
    def _find_split_or_slice(self, node, optimizer, max_depth=20):
        """Find Split or StridedSlice in backward path."""
        current = node
        for _ in range(max_depth):
            if current.op == 'StridedSlice':
                return {'type': 'StridedSlice', 'node': current}
            if current.op in ('Split', 'SplitV'):
                return {'type': 'Split', 'node': current}
            
            if not current.input:
                break
            
            input_name = self.clean_input_name(current.input[0])
            current = optimizer.nodes.get(input_name)
            if not current:
                break
        
        return None
    
    def _check_dimension_compatibility(self, op_type, pack_node, branches):
        """Check if the operation can handle dimension increase from Pack."""
        return OpConverter.is_hoistable_through(op_type)
    
    def _update_output_shape_for_hoisted_op(self, hoisted_op, pack_node, branches, op_type):
        """Update _output_shapes for hoisted operation (adds pack dimension)."""
        pack_axis_value = pack_node.attr.get('axis').i if pack_node.attr.get('axis') else 0
        pack_size = len(branches)
        
        if branches and '_output_shapes' in branches[0].attr:
            original_shape_list = branches[0].attr['_output_shapes'].list.shape
            if original_shape_list and original_shape_list[0].dim:  # Check dim is not empty
                new_shape = self._create_packed_shape(
                    original_shape_list[0], pack_axis_value, pack_size
                )
                output_shapes_attr = attr_value_pb2.AttrValue()
                output_shapes_attr.list.shape.add().CopyFrom(new_shape)
                hoisted_op.attr['_output_shapes'].CopyFrom(output_shapes_attr)
                return
        
        # Don't set _output_shapes if unknown - let TF infer it
        # Setting an empty shape [[]] causes validation errors
        # Just remove the attribute if it exists with empty value
        if '_output_shapes' in hoisted_op.attr:
            del hoisted_op.attr['_output_shapes']
    
    def _create_packed_shape(self, original_shape, pack_axis, pack_size):
        """Create shape with pack dimension inserted at specified axis."""
        new_shape = tensor_shape_pb2.TensorShapeProto()
        
        # Copy dimensions before pack axis
        for i in range(pack_axis):
            if i < len(original_shape.dim):
                new_dim = new_shape.dim.add()
                new_dim.size = original_shape.dim[i].size
        
        # Insert pack dimension
        pack_dim = new_shape.dim.add()
        pack_dim.size = pack_size
        
        # Copy remaining dimensions
        for i in range(pack_axis, len(original_shape.dim)):
            new_dim = new_shape.dim.add()
            new_dim.size = original_shape.dim[i].size
        
        return new_shape
    
    def _update_pack_output_shape(self, pack_node, input_names, optimizer):
        """Update _output_shapes for Pack node."""
        pack_axis_value = pack_node.attr.get('axis').i if pack_node.attr.get('axis') else 0
        pack_size = len(input_names)
        
        if input_names and optimizer.nodes:
            first_input_name = self.clean_input_name(input_names[0])
            first_input_node = optimizer.nodes.get(first_input_name)
            
            if first_input_node and '_output_shapes' in first_input_node.attr:
                input_shape_list = first_input_node.attr['_output_shapes'].list.shape
                if input_shape_list and input_shape_list[0].dim:  # Check dim is not empty
                    new_shape = self._create_packed_shape(
                        input_shape_list[0], pack_axis_value, pack_size
                    )
                    output_shapes_attr = attr_value_pb2.AttrValue()
                    output_shapes_attr.list.shape.add().CopyFrom(new_shape)
                    pack_node.attr['_output_shapes'].CopyFrom(output_shapes_attr)
                    return
        
        # Don't set _output_shapes if unknown - let TF infer it
        if '_output_shapes' in pack_node.attr:
            del pack_node.attr['_output_shapes']
    
    def _hoist_through_op(self, pack_node, branches, op_type, optimizer):
        """Hoist Pack through the operation.
        
        Transform: Pack([Op(x1, w), Op(x2, w), ...]) -> Op(Pack([x1, x2, ...]), w)
        
        Uses OpConverter for handling special conversions (BiasAdd->AddV2, MatMul->BatchMatMulV2).
        
        IMPORTANT: Cannot hoist if Pack or branches have external consumers, as hoisting
        changes the output shape semantics (hoisted op output != original Pack output).
        """
        if not self._check_dimension_compatibility(op_type, pack_node, branches):
            return None
        
        # CRITICAL: Check if Pack node has external consumers
        # Hoisting changes output shape, so external consumers would receive wrong shape
        pack_consumers = optimizer.consumers.get(pack_node.name, [])
        if len(pack_consumers) > 0:
            logging.debug(f"[PackHoisting] {pack_node.name}: Pack has {len(pack_consumers)} external consumers, skip hoisting")
            return None
        
        # CRITICAL: Check if branch nodes have external consumers
        # If branches are consumed by nodes other than Pack, hoisting breaks those paths
        for branch in branches:
            consumers = optimizer.consumers.get(branch.name, [])
            for consumer in consumers:
                consumer_name = self.clean_input_name(consumer)
                if consumer_name != pack_node.name:
                    logging.debug(f"[PackHoisting] {pack_node.name}: branch {branch.name} has external consumer {consumer_name}, skip hoisting")
                    return None
        
        num_inputs = len(branches[0].input)
        data_inputs = []
        shared_inputs = []
        
        for input_idx in range(num_inputs):
            inputs_at_idx = []
            for branch in branches:
                if input_idx < len(branch.input):
                    inp = self.clean_input_name(branch.input[input_idx])
                    inputs_at_idx.append(inp)
                else:
                    return None
            
            if len(set(inputs_at_idx)) == 1:
                shared_inputs.append((input_idx, branches[0].input[input_idx]))
            else:
                data_inputs.append((input_idx, inputs_at_idx))
        
        # Check if hoisting is beneficial
        if len(branches) <= len(data_inputs) + 1:
            return None
        
        # Check if conversion is needed based on varying inputs
        has_varying_packable_input = any(
            idx == OpConverter.get_pack_input_idx(op_type) 
            for idx, _ in data_inputs
        )
        actual_op_type = OpConverter.get_target_op(op_type, has_varying_packable_input)
        
        if actual_op_type != op_type:
            logging.debug(f"[PackHoisting] Converting {op_type} to {actual_op_type}")
        
        # Create Pack nodes for data inputs
        new_packs = []
        hoisted_inputs = [''] * num_inputs
        
        for pack_idx, (data_input_idx, data_input_names) in enumerate(data_inputs):
            pack_attrs = dict(pack_node.attr)
            
            # For converted ops (BatchMatMulV2, AddV2), use axis=0 for batch dimension
            if actual_op_type in ('BatchMatMulV2', 'AddV2'):
                pack_attrs['axis'].i = 0
            
            # Use unified cache interface
            pack_name, is_new, new_pack_node = self.get_or_create_cached_node(
                "Pack",
                data_input_names,
                pack_attrs,
                pack_node.name,
                ""
            )
            
            hoisted_inputs[data_input_idx] = pack_name
            
            if is_new and new_pack_node:
                self._update_pack_output_shape(new_pack_node, data_input_names, optimizer)
                new_packs.append(new_pack_node)
        
        for shared_idx, shared_input in shared_inputs:
            hoisted_inputs[shared_idx] = shared_input
        
        # Use OpConverter for attribute conversion
        hoisted_op_attr = OpConverter.convert_attrs(op_type, dict(branches[0].attr), actual_op_type)
        
        new_name = self.make_unique_node_name(pack_node.name, actual_op_type) 
        hoisted_op = create_node(
            actual_op_type,
            new_name,
            inputs=hoisted_inputs,
            attr=hoisted_op_attr
        )
        
        self._update_output_shape_for_hoisted_op(hoisted_op, pack_node, branches, actual_op_type)
        
        return {
            'op_type': actual_op_type,
            'new_packs': new_packs,
            'hoisted_op': hoisted_op,
            'original_pack_name': pack_node.name
        }
    
    def _generate_batched_ops(self, pack_node, branches, common_source, optimizer, max_trace_depth=50):
        """Generate batched operations when Pack can be eliminated with Split/StridedSlice."""
        all_paths = []
        
        for branch in branches:
            path = []
            current = branch
            depth = 0
            
            while depth < max_trace_depth and current:
                if current.op in ('StridedSlice', 'Split', 'SplitV'):
                    break
                path.append(current)
                
                if not current.input:
                    break
                
                input_name = self.clean_input_name(current.input[0])
                current = optimizer.nodes.get(input_name)
                depth += 1
            
            all_paths.append(path)
        
        if not all_paths or not all_paths[0]:
            new_name = self.make_unique_node_name(pack_node.name, 'Identity')
            identity_node = create_node(
                "Identity",
                new_name,
                inputs=[common_source]
            )
            # Copy _output_shapes from pack_node to identity_node (only if valid)
            if '_output_shapes' in pack_node.attr:
                pack_shapes = pack_node.attr['_output_shapes'].list.shape
                if pack_shapes and pack_shapes[0].dim:
                    identity_node.attr['_output_shapes'].CopyFrom(pack_node.attr['_output_shapes'])
            logging.debug(f"[PackHoisting] {pack_node.name}: eliminated (direct connection)")
            return {
                'hoist_path': ['eliminated'],
                'new_nodes': [identity_node]
            }
        
        path_length = len(all_paths[0])
        if not all(len(p) == path_length for p in all_paths):
            logging.warning(f"[PackHoisting] {pack_node.name}: branches have different path lengths")
            return None
        
        for i in range(path_length):
            ops_at_level = [p[i].op for p in all_paths]
            if not all(op == ops_at_level[0] for op in ops_at_level):
                logging.warning(f"[PackHoisting] {pack_node.name}: different ops at level {i}")
                return None
        
        # CRITICAL: Check if branch nodes have external consumers (outside the Pack pattern)
        # If any branch node is consumed by nodes other than Pack and nodes within this path,
        # elimination would break those external consumers
        for path in all_paths:
            for node in path:
                consumers = optimizer.consumers.get(node.name, [])
                for consumer in consumers:
                    consumer_name = self.clean_input_name(consumer)
                    # Consumer is the Pack node being eliminated - OK
                    if consumer_name == pack_node.name:
                        continue
                    # Consumer is in the same path (later in the chain) - OK
                    path_names = {n.name for n in path}
                    if consumer_name in path_names:
                        continue
                    # External consumer found - cannot safely eliminate
                    logging.debug(f"[PackHoisting] {pack_node.name}: branch node {node.name} has external consumer {consumer_name}, skip elimination")
                    return None
        
        new_nodes = []
        batched_node_map = {common_source: common_source}
        # Track created nodes for _output_shapes update
        created_nodes = {}
        
        for level_idx in range(path_length - 1, -1, -1):
            ops_at_level = [p[level_idx] for p in all_paths]
            first_op = ops_at_level[0]
            op_type = first_op.op
            
            # For dedup: track processed shared inputs -> actual node name
            seen_shared_inputs = {}
            batched_inputs = []
            needs_pack_inputs = []  # Track inputs that need Pack (for MatMul->BatchMatMulV2)
            
            for inp_idx in range(len(first_op.input)):
                inputs_at_position = []
                for op in ops_at_level:
                    if inp_idx < len(op.input):
                        inp = self.clean_input_name(op.input[inp_idx])
                        inputs_at_position.append(inp)
                    else:
                        logging.warning(f"[PackHoisting] {pack_node.name}: input count mismatch at level {level_idx}")
                        return None
                
                unique_inputs = set(inputs_at_position)
                
                if len(unique_inputs) == 1:
                    shared_input_name = inputs_at_position[0]
                    
                    # Check if this shared input has been processed (dedup)
                    if shared_input_name in seen_shared_inputs:
                        # Reuse already processed node name
                        batched_inputs.append(seen_shared_inputs[shared_input_name])
                    elif shared_input_name in batched_node_map:
                        # Use batched version
                        mapped_name = batched_node_map[shared_input_name]
                        batched_inputs.append(mapped_name)
                        seen_shared_inputs[shared_input_name] = mapped_name
                    else:
                        # Use original input
                        batched_inputs.append(first_op.input[inp_idx])
                        seen_shared_inputs[shared_input_name] = first_op.input[inp_idx]
                else:
                    # Different inputs at this position
                    # Check if all inputs are mapped to the same batched node
                    mapped_inputs = [batched_node_map.get(inp) for inp in inputs_at_position]
                    if all(m is not None for m in mapped_inputs) and len(set(mapped_inputs)) == 1:
                        # All inputs map to the same batched node (e.g., all from same Split source)
                        batched_inputs.append(mapped_inputs[0])
                    # Check if this is a packable input for convertible ops (MatMul, BiasAdd)
                    elif inp_idx == OpConverter.get_pack_input_idx(op_type):
                        # Pack the different inputs (weights for MatMul, biases for BiasAdd)
                        varying_inputs = [op.input[inp_idx] for op in ops_at_level]
                        pack_attrs = dict(pack_node.attr)
                        pack_attrs['axis'].i = 0  # Batch dimension first
                        
                        suffix = "weight" if op_type == 'MatMul' else "bias" if op_type == 'BiasAdd' else "param"
                        pack_name, is_new, pack_node_new = self.get_or_create_cached_node(
                            "Pack",
                            varying_inputs,
                            pack_attrs,
                            pack_node.name,
                            suffix
                        )
                        
                        if is_new and pack_node_new:
                            new_nodes.append(pack_node_new)
                        
                        batched_inputs.append(pack_name)
                        needs_pack_inputs.append(inp_idx)
                    # For data inputs (inp_idx==0), check if they're from StridedSlice and map to common_source
                    elif inp_idx == 0:
                        # Check if all inputs come from StridedSlice pointing to common_source
                        all_from_split = True
                        for inp in inputs_at_position:
                            inp_node = optimizer.nodes.get(inp)
                            if inp_node and inp_node.op in ('StridedSlice', 'Split', 'SplitV'):
                                # Map this input to common_source
                                batched_node_map[inp] = common_source
                            else:
                                all_from_split = False
                                break
                        
                        if all_from_split:
                            batched_inputs.append(common_source)
                        else:
                            logging.debug(f"[PackHoisting] {pack_node.name}: different inputs at level {level_idx} pos {inp_idx}")
                            return None
                    else:
                        logging.debug(f"[PackHoisting] {pack_node.name}: different inputs at level {level_idx} pos {inp_idx}")
                        return None
            
            # Use OpConverter for type conversion and attribute handling
            has_varying_packable = len(needs_pack_inputs) > 0
            actual_op_type = OpConverter.get_target_op(op_type, has_varying_packable)
            
            if actual_op_type != op_type:
                logging.debug(f"[PackHoisting] Converting {op_type} to {actual_op_type} in elimination path")
            
            # Use OpConverter for attribute conversion
            batched_attr = OpConverter.convert_attrs(op_type, dict(first_op.attr), actual_op_type)
            
            # Use unified cache interface to create batch node
            batched_name, is_new, batched_node = self.get_or_create_cached_node(
                actual_op_type,
                batched_inputs,
                batched_attr,
                pack_node.name,
                ""
            )
            
            if is_new and batched_node:
                # Update _output_shapes for the batched node
                # The output shape should be the pack's output shape (since pack is eliminated)
                if level_idx == 0:
                    # This is the final op before pack, its output shape = pack's output shape
                    if '_output_shapes' in pack_node.attr:
                        pack_shapes = pack_node.attr['_output_shapes'].list.shape
                        if pack_shapes and pack_shapes[0].dim:  # Has actual dimensions
                            batched_node.attr['_output_shapes'].CopyFrom(pack_node.attr['_output_shapes'])
                        elif '_output_shapes' in batched_node.attr:
                            del batched_node.attr['_output_shapes']
                else:
                    # For intermediate ops, compute shape from first_op with pack dimension added
                    self._update_output_shape_for_batched_op(batched_node, pack_node, first_op)
                
                new_nodes.append(batched_node)
                created_nodes[batched_name] = batched_node
            
            # Update mapping
            for op in ops_at_level:
                batched_node_map[op.name] = batched_name
        # Create final Identity node with proper _output_shapes and T attr
        new_name = self.make_unique_node_name(pack_node.name, 'Identity')
        identity_node = create_node(
            "Identity",
            new_name,
            inputs=[batched_name]
        )
        # Copy T attr from pack_node (type attribute)
        if 'T' in pack_node.attr:
            identity_node.attr['T'].CopyFrom(pack_node.attr['T'])
        # Copy _output_shapes from pack_node (Identity output = Pack output)
        # Only copy if pack_node has valid shapes (not empty)
        if '_output_shapes' in pack_node.attr:
            pack_shapes = pack_node.attr['_output_shapes'].list.shape
            if pack_shapes and pack_shapes[0].dim:  # Has actual dimensions
                identity_node.attr['_output_shapes'].CopyFrom(pack_node.attr['_output_shapes'])
        new_nodes.append(identity_node)
        
        old_branch_nodes = []
        for path in all_paths:
            for node in path:
                old_branch_nodes.append(node.name)
        
        # Add mapping for the Pack node itself -> Identity node
        batched_node_map[pack_node.name] = identity_node.name
        
        logging.debug(f"[PackHoisting] {pack_node.name}: eliminated with {path_length} batched ops")
        
        return {
            'hoist_path': ['eliminated'],
            'new_nodes': new_nodes,
            'replaced_nodes': old_branch_nodes,
            'node_mapping': batched_node_map  # Add the mapping for consumer updates
        }
    
    def _update_output_shape_for_batched_op(self, batched_node, pack_node, original_op):
        """Update _output_shapes for batched operation in elimination path.
        
        When eliminating Split -> Ops -> Pack, the batched ops should have
        their output shapes updated to include the pack dimension.
        """
        pack_axis_value = pack_node.attr.get('axis').i if pack_node.attr.get('axis') else 0
        pack_n = pack_node.attr.get('N').i if pack_node.attr.get('N') else len(pack_node.input)
        
        if '_output_shapes' in original_op.attr:
            original_shape_list = original_op.attr['_output_shapes'].list.shape
            if original_shape_list and original_shape_list[0].dim:  # Has actual dimensions
                # Create shape with pack dimension inserted
                new_shape = self._create_packed_shape(
                    original_shape_list[0], pack_axis_value, pack_n
                )
                output_shapes_attr = attr_value_pb2.AttrValue()
                output_shapes_attr.list.shape.add().CopyFrom(new_shape)
                batched_node.attr['_output_shapes'].CopyFrom(output_shapes_attr)
                return
        
        # Unknown shape fallback - copy pack_node's output shapes if valid
        if '_output_shapes' in pack_node.attr:
            pack_shapes = pack_node.attr['_output_shapes'].list.shape
            if pack_shapes and pack_shapes[0].dim:  # Has actual dimensions
                batched_node.attr['_output_shapes'].CopyFrom(pack_node.attr['_output_shapes'])
            elif '_output_shapes' in batched_node.attr:
                # Remove empty _output_shapes
                del batched_node.attr['_output_shapes']
