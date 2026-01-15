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


@PassRegistry.register("pack_hoisting", opt_level=3, priority=30)
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
            logging.debug(f"Pack {pack_node.name}: skipped (< 2 inputs)")
            return None
        
        logging.debug(f"Analyzing Pack node: {pack_node.name} with {len(pack_inputs)} inputs")
        
        # Try to hoist the pack (with recursive hoisting until can't continue)
        result = self._try_hoist_pack_recursive(pack_node, pack_inputs, optimizer)
        
        if result:
            hoist_path = result['hoist_path']
            hoist_desc = " -> ".join(hoist_path) if hoist_path else "eliminated"
            logging.info(f"[PACK_HOISTING] Pack {pack_node.name}: fully hoisted ({len(hoist_path)} layers: {hoist_desc})")
            
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
            logging.debug(f"Pack {pack_node.name}: branches have different ops")
            return None
        
        first_op = branches[0].op
        
        # Check if operation is hoistable
        if first_op in BLOCKING_OPS or first_op not in HOISTABLE_OPS:
            logging.debug(f"Pack {pack_node.name}: op {first_op} not hoistable")
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
            logging.debug(f"Pack {pack_node.name}: operations cannot be batched (different parameters)")
            return None
        
        logging.info(f"[PACK_HOISTING] Pack {pack_node.name}: can eliminate with {split_sources[0]['type']}")
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
                        logging.debug(f"Split node {split_node['node'].name} has insufficient inputs: {len(split_node['node'].input)}")
                        return None
                    source = self.clean_input_name(split_node['node'].input[1])
                else:
                    return None
                
                if common_source is None:
                    common_source = source
                elif common_source != source:
                    return None
            except (AttributeError, TypeError, KeyError) as e:
                logging.warning(f"Error extracting source from split_node: {e}, split_node={split_node}")
                return None
        
        return common_source
    
    def _can_batch_operations(self, branches, optimizer):
        """Check if operations can be batched (same non-data inputs across branches)."""
        if not branches:
            return False
        
        num_inputs = len(branches[0].input)
        
        for input_idx in range(num_inputs):
            inputs_at_idx = [
                self.clean_input_name(branch.input[input_idx])
                for branch in branches
                if input_idx < len(branch.input)
            ]
            
            if len(inputs_at_idx) != len(branches):
                return False  # Input count mismatch
            
            # If input varies across branches, verify it traces to Split/StridedSlice
            if len(set(inputs_at_idx)) > 1:
                for inp_name in set(inputs_at_idx):
                    inp_node = optimizer.nodes.get(inp_name)
                    if inp_node and inp_node.op not in ('StridedSlice', 'Split', 'SplitV'):
                        if not self._find_split_or_slice(inp_node, optimizer, max_depth=20):
                            logging.debug(f"Cannot batch: input {input_idx} differs ({inp_name})")
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
        if op_type in ('Identity', 'BiasAdd', 'MatMul', 'BatchMatMul', 'BatchMatMulV2'):
            return True
        
        if op_type in ELEMENTWISE_OPS:
            return True
        
        logging.debug(f"Pack {pack_node.name}: uncertain dimension compatibility for {op_type}")
        return False
    
    def _update_output_shape_for_hoisted_op(self, hoisted_op, pack_node, branches, op_type):
        """Update _output_shapes for hoisted operation (adds pack dimension)."""
        pack_axis_value = pack_node.attr.get('axis').i if pack_node.attr.get('axis') else 0
        pack_size = len(branches)
        
        if branches and '_output_shapes' in branches[0].attr:
            original_shape_list = branches[0].attr['_output_shapes'].list.shape
            if original_shape_list:
                new_shape = self._create_packed_shape(
                    original_shape_list[0], pack_axis_value, pack_size
                )
                output_shapes_attr = attr_value_pb2.AttrValue()
                output_shapes_attr.list.shape.add().CopyFrom(new_shape)
                hoisted_op.attr['_output_shapes'].CopyFrom(output_shapes_attr)
                return
        
        # Unknown shape fallback
        output_shapes_attr = attr_value_pb2.AttrValue()
        output_shapes_attr.list.shape.add()
        hoisted_op.attr['_output_shapes'].CopyFrom(output_shapes_attr)
    
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
                if input_shape_list:
                    new_shape = self._create_packed_shape(
                        input_shape_list[0], pack_axis_value, pack_size
                    )
                    output_shapes_attr = attr_value_pb2.AttrValue()
                    output_shapes_attr.list.shape.add().CopyFrom(new_shape)
                    pack_node.attr['_output_shapes'].CopyFrom(output_shapes_attr)
                    return
        
        # Unknown shape fallback
        output_shapes_attr = attr_value_pb2.AttrValue()
        output_shapes_attr.list.shape.add()
        pack_node.attr['_output_shapes'].CopyFrom(output_shapes_attr)
    
    def _hoist_through_op(self, pack_node, branches, op_type, optimizer):
        """Hoist Pack through the operation.
        
        Transform: Pack([Op(x1, w), Op(x2, w), ...]) -> Op(Pack([x1, x2, ...]), w)
        """
        if not self._check_dimension_compatibility(op_type, pack_node, branches):
            logging.debug(f"Pack {pack_node.name}: {op_type} not dimension-compatible")
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
                    logging.debug(f"Pack {pack_node.name}: branches have different input counts")
                    return None
            
            if len(set(inputs_at_idx)) == 1:
                shared_inputs.append((input_idx, branches[0].input[input_idx]))
            else:
                data_inputs.append((input_idx, inputs_at_idx))
        
        # Check if hoisting is beneficial
        if len(branches) <= len(data_inputs) + 1:
            logging.debug(f"Pack {pack_node.name}: skipping hoist - not beneficial")
            return None
        
        # Create Pack nodes for data inputs (with caching to avoid duplicates)
        new_packs = []
        hoisted_inputs = [''] * num_inputs
        
        for pack_idx, (data_input_idx, data_input_names) in enumerate(data_inputs):
            # 使用统一的缓存接口
            pack_name, is_new, new_pack_node = self.get_or_create_cached_node(
                "Pack",
                data_input_names,
                pack_node.attr,
                pack_node.name,
                f"Hoisting Pack {pack_node.name}"
            )
            
            hoisted_inputs[data_input_idx] = pack_name
            
            if is_new and new_pack_node:
                self._update_pack_output_shape(new_pack_node, data_input_names, optimizer)
                new_packs.append(new_pack_node)
        
        for shared_idx, shared_input in shared_inputs:
            hoisted_inputs[shared_idx] = shared_input
        
        hoisted_op_type = op_type
        hoisted_op_attr = {}
        for key, value in branches[0].attr.items():
            if key != '_output_shapes':
                hoisted_op_attr[key] = value
        new_name = self.make_unique_node_name(pack_node.name, hoisted_op_type) 
        hoisted_op = create_node(
            hoisted_op_type,
            new_name,
            inputs=hoisted_inputs,
            attr=hoisted_op_attr
        )
        
        self._update_output_shape_for_hoisted_op(hoisted_op, pack_node, branches, op_type)
        
        return {
            'op_type': hoisted_op_type,
            'new_packs': new_packs,
            'hoisted_op': hoisted_op,
            'original_pack_name': pack_node.name  # 记录原始Pack节点名称，用于映射
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
            logging.info(f"[PACK_HOISTING] Pack {pack_node.name}: eliminated (direct connection)")
            return {
                'hoist_path': ['eliminated'],
                'new_nodes': [identity_node]
            }
        
        path_length = len(all_paths[0])
        if not all(len(p) == path_length for p in all_paths):
            logging.warning(f"Pack {pack_node.name}: branches have different path lengths")
            return None
        
        for i in range(path_length):
            ops_at_level = [p[i].op for p in all_paths]
            if not all(op == ops_at_level[0] for op in ops_at_level):
                logging.warning(f"Pack {pack_node.name}: branches have different ops at level {i}")
                return None
        
        new_nodes = []
        batched_node_map = {common_source: common_source}
        
        for level_idx in range(path_length - 1, -1, -1):
            ops_at_level = [p[level_idx] for p in all_paths]
            first_op = ops_at_level[0]
            
            # 用于去重：记录已经处理过的共享输入节点 -> 实际使用的节点名
            seen_shared_inputs = {}
            batched_inputs = []
            
            for inp_idx in range(len(first_op.input)):
                inputs_at_position = []
                for op in ops_at_level:
                    if inp_idx < len(op.input):
                        inp = self.clean_input_name(op.input[inp_idx])
                        inputs_at_position.append(inp)
                    else:
                        logging.warning(f"Pack {pack_node.name}: input count mismatch at level {level_idx}")
                        return None
                
                unique_inputs = set(inputs_at_position)
                
                if len(unique_inputs) == 1:
                    shared_input_name = inputs_at_position[0]
                    
                    # 检查这个共享输入是否已经被处理过（去重）
                    if shared_input_name in seen_shared_inputs:
                        # 复用已经处理过的节点名
                        batched_inputs.append(seen_shared_inputs[shared_input_name])
                        logging.debug(f"Pack {pack_node.name} level {level_idx}: REUSING input {shared_input_name} at position {inp_idx}")
                    elif shared_input_name in batched_node_map:
                        # 使用批处理后的版本
                        mapped_name = batched_node_map[shared_input_name]
                        batched_inputs.append(mapped_name)
                        seen_shared_inputs[shared_input_name] = mapped_name
                        logging.debug(f"Pack {pack_node.name} level {level_idx} pos {inp_idx}: using batched {mapped_name} for {shared_input_name}")
                    else:
                        # 使用原始输入
                        batched_inputs.append(first_op.input[inp_idx])
                        seen_shared_inputs[shared_input_name] = first_op.input[inp_idx]
                        logging.debug(f"Pack {pack_node.name} level {level_idx} pos {inp_idx}: using original {first_op.input[inp_idx]}")
                else:
                    logging.warning(f"Pack {pack_node.name} level {level_idx} pos {inp_idx}: branches have different inputs {list(unique_inputs)[:3]}...")
                    return None
            
            # 提取关键属性（排除_output_shapes）
            batched_attr = {k: v for k, v in first_op.attr.items() if k != '_output_shapes'}
            
            # 使用统一的缓存接口创建batch节点
            batched_name, is_new, batched_node = self.get_or_create_cached_node(
                first_op.op,
                batched_inputs,
                batched_attr,
                pack_node.name,
                f"Pack {pack_node.name} level {level_idx}"
            )
            
            if is_new and batched_node:
                new_nodes.append(batched_node)
            
            # 更新映射
            for op in ops_at_level:
                batched_node_map[op.name] = batched_name
        new_name = self.make_unique_node_name(pack_node.name, 'Identity')
        identity_node = create_node(
            "Identity",
            new_name,
            inputs=[batched_name]
        )
        new_nodes.append(identity_node)
        
        old_branch_nodes = []
        for path in all_paths:
            for node in path:
                old_branch_nodes.append(node.name)
        
        logging.info(f"[PACK_HOISTING] Pack {pack_node.name}: eliminated with {path_length} batched ops")
        
        return {
            'hoist_path': ['eliminated'],
            'new_nodes': new_nodes,
            'replaced_nodes': old_branch_nodes,
            'node_mapping': batched_node_map  # Add the mapping for consumer updates
        }
