"""
Common Subexpression Elimination (CSE) Pass

消除公共子表达式：识别并删除图中语义相同的重复节点。

工作原理：
1. 扫描图中所有节点
2. 为每个节点计算签名（基于 op、inputs、attrs）
3. 识别具有相同签名的重复节点
4. 将所有引用重定向到规范节点（保留第一个）
5. 删除重复节点
6. 迭代执行步骤 1-5，直到没有新的重复节点（收敛）

迭代的必要性：
    消除一组重复节点后，可能会暴露出新的重复模式。
    例如：a = Add(x, y) 和 b = Add(x, y) 被合并后，
    依赖它们的上游节点 c = Mul(a, z) 和 d = Mul(b, z) 
    可能变成新的重复节点。

Const 节点去重规则：
    Const 节点必须同时满足以下条件才会被认为是重复的：
    1. value 完全相同（张量的所有值相同）
    2. dtype 相同（数据类型相同）
    
    例如：
    - Const(value=1, dtype=int32) 和 Const(value=1, dtype=int32) → 重复，合并
    - Const(value=1, dtype=int32) 和 Const(value=1, dtype=float32) → 不同，保留

不可去重的操作类型：
    1. 输入节点：Placeholder（每个输入都是独立的）
    2. 有状态操作：Variable, VarHandleOp, ReadVariableOp（每次访问可能得到不同值）
    3. 随机操作：RandomUniform, RandomNormal（每次调用结果不同）
    4. 副作用操作：Print, Assert（执行次数有意义）
    5. 控制流操作：Switch, Merge, Enter, Exit（控制流结构不可合并）
    6. Identity 操作：由专门的 identity_removal pass 处理
    7. 队列/栈/数组操作：QueueDequeueV2, StackPopV2, TensorArrayReadV3（有状态）

控制依赖的处理：
    控制依赖（^node）会被包含在节点签名中，因此：
    - Add(a, b, ^ctrl1) 和 Add(a, b, ^ctrl2) → 不同的控制依赖，不会合并
    - Add(a, b, ^ctrl1) 和 Add(a, b, ^ctrl1) → 相同的控制依赖，会合并

示例：
    原图：
        a = MatMul(x, w)
        b = MatMul(x, w)  # 重复的计算
        y1 = Add(a, c1)
        y2 = Add(b, c2)
    
    第一次迭代后：
        a = MatMul(x, w)
        y1 = Add(a, c1)
        y2 = Add(a, c2)  # b 被 a 替换并删除
    
    （如果 y1 和 y2 也是重复的，会在下一次迭代中被发现）
"""

from collections import defaultdict
from ..utils.logger import logger as logging
from ..core import PassRegistry, BasePass


def extract_key_attrs(attrs, op_type=None):
    """
    提取关键属性用于节点签名（排除形状等运行时属性）。
    
    跳过的属性：_output_shapes, _class（这些是推断出来的元数据）
    对于大多数节点，T 和 dtype 通常是推断出来的，不影响语义等价性。
    但对于 Const 节点，dtype 是关键属性，必须包含在签名中。
    
    Args:
        attrs: 属性字典（AttrValue对象）
        op_type: 节点操作类型（用于特殊处理某些操作）
        
    Returns:
        tuple: 属性签名元组 (attr_name, type, value)
    """
    key_attrs = []
    # 基础跳过属性（这些是推断出来的元数据，不影响语义）
    skip_attrs = {'_output_shapes', '_class'}
    
    # 对于非 Const 节点，额外跳过 T 和 dtype（这些通常是类型推断的结果）
    if op_type != 'Const':
        skip_attrs.update({'T', 'dtype', 'Tshape', 'Tidx', 'Taxis', 'Tpaddings'})
    
    for attr_name in sorted(attrs.keys()):
        if attr_name in skip_attrs:
            continue
        
        attr_value = attrs[attr_name]
        
        # 按优先级检查属性类型
        if attr_value.HasField('i'):
            key_attrs.append((attr_name, 'i', attr_value.i))
        elif attr_value.HasField('f'):
            key_attrs.append((attr_name, 'f', attr_value.f))
        elif attr_value.HasField('b'):
            key_attrs.append((attr_name, 'b', attr_value.b))
        elif attr_value.HasField('s'):
            key_attrs.append((attr_name, 's', attr_value.s))
        elif attr_value.HasField('type'):
            key_attrs.append((attr_name, 'type', attr_value.type))
        elif attr_value.HasField('shape'):
            # shape 属性（如 Placeholder 的 shape）
            shape_dims = tuple(d.size for d in attr_value.shape.dim)
            key_attrs.append((attr_name, 'shape', shape_dims))
        elif attr_value.HasField('tensor'):
            # tensor 类型（Const 节点的 value 属性）
            # 序列化为字节串确保相同值的常量有相同签名
            tensor_bytes = attr_value.tensor.SerializeToString()
            key_attrs.append((attr_name, 'tensor', tensor_bytes))
        elif attr_value.HasField('func'):
            # 函数引用（如 While 循环的 body/cond）
            key_attrs.append((attr_name, 'func', attr_value.func.name))
        elif attr_value.HasField('placeholder'):
            key_attrs.append((attr_name, 'placeholder', attr_value.placeholder))
        elif attr_value.list.i:
            # list of int
            key_attrs.append((attr_name, 'list_i', tuple(attr_value.list.i)))
        elif attr_value.list.f:
            # list of float
            key_attrs.append((attr_name, 'list_f', tuple(attr_value.list.f)))
        elif attr_value.list.b:
            # list of bool
            key_attrs.append((attr_name, 'list_b', tuple(attr_value.list.b)))
        elif attr_value.list.s:
            # list of string
            key_attrs.append((attr_name, 'list_s', tuple(attr_value.list.s)))
        elif attr_value.list.type:
            # list of type
            key_attrs.append((attr_name, 'list_type', tuple(attr_value.list.type)))
        elif attr_value.list.shape:
            # list of shape
            shapes = tuple(
                tuple(d.size for d in shape.dim) 
                for shape in attr_value.list.shape
            )
            key_attrs.append((attr_name, 'list_shape', shapes))
        elif attr_value.list.tensor:
            # list of tensor
            tensors = tuple(t.SerializeToString() for t in attr_value.list.tensor)
            key_attrs.append((attr_name, 'list_tensor', tensors))
        elif attr_value.list.func:
            # list of func
            funcs = tuple(f.name for f in attr_value.list.func)
            key_attrs.append((attr_name, 'list_func', funcs))
    
    return tuple(key_attrs)


def create_cse_signature(node):
    """
    为 CSE 创建节点签名，保留控制依赖标记和端口号。
    
    这样可以区分：
    - Add(a, b) 和 Add(a, b, ^ctrl) 是不同的节点
    - Add(a, b, ^ctrl1) 和 Add(a, b, ^ctrl2) 是不同的节点
    - Add(split:0, split:0) 和 Add(split:0, split:1) 是不同的节点
    
    Args:
        node: tf.NodeDef 节点
        
    Returns:
        tuple: (op_type, inputs_tuple_with_ctrl_deps_and_ports, key_attrs)
    """
    # 保留完整输入（包括控制依赖前缀和端口号）
    inputs_tuple = tuple(node.input)
    
    # 提取关键属性
    key_attrs = extract_key_attrs(node.attr, op_type=node.op)
    
    return (node.op, inputs_tuple, key_attrs)


def build_deduplication_map(optimizer, skip_ops, protected_nodes=None):
    """
    构建全局去重映射，识别图中语义相同的重复节点。
    
    使用两阶段分组优化性能：
    1. 第一阶段：按 (op, inputs) 快速分组（成本低）
    2. 第二阶段：只有潜在重复的组才计算完整属性签名（避免无用计算）
    
    Args:
        optimizer: GraphOptimizer 实例
        skip_ops: 要跳过的操作类型集合（这些节点不应去重）
        protected_nodes: 受保护的节点集合（这些节点不会被删除，但可以作为规范节点）
        
    Returns:
        dict: {duplicate_node_name -> canonical_node_name}
    """
    protected_set = set(protected_nodes or [])
    
    # === 第一阶段：按 (op, inputs) 快速分组 ===
    # 这个分组成本很低，只需要访问 op 和 input 字段
    quick_groups = defaultdict(list)
    
    for node in optimizer.graph_def.node:
        # 跳过不应该去重的节点类型
        if node.op in skip_ops:
            continue
        
        # 快速签名：只用 op 和 inputs（不计算属性）
        quick_sig = (node.op, tuple(node.input))
        quick_groups[quick_sig].append(node)
    
    # === 第二阶段：只对潜在重复的组计算完整签名 ===
    nodes_by_signature = defaultdict(list)
    
    for quick_sig, nodes in quick_groups.items():
        if len(nodes) == 1:
            # 只有一个节点，不可能有重复，跳过属性计算
            continue
        
        # 有多个节点具有相同的 (op, inputs)，需要检查属性是否也相同
        for node in nodes:
            # 现在才计算完整签名（包括属性）
            full_signature = create_cse_signature(node)
            nodes_by_signature[full_signature].append(node.name)
    
    # === 构建去重映射 ===
    dedup_map = {}
    
    for signature, node_names in nodes_by_signature.items():
        if len(node_names) <= 1:
            continue
        
        # 选择规范节点：优先选择受保护的节点，然后是名字最短的
        protected_candidates = [n for n in node_names if n in protected_set]
        
        if protected_candidates:
            canonical = min(protected_candidates, key=lambda n: (len(n), n))
        else:
            canonical = min(node_names, key=lambda n: (len(n), n))
        
        # 将所有非规范节点（且非受保护节点）映射到规范节点
        for node_name in node_names:
            if node_name != canonical and node_name not in protected_set:
                dedup_map[node_name] = canonical
    
    return dedup_map


def apply_deduplication_map(optimizer, dedup_map, pass_name="CSE"):
    """
    应用去重映射：更新所有节点的输入引用，删除重复节点。
    
    Args:
        optimizer: GraphOptimizer 实例
        dedup_map: 去重映射 {duplicate_node_name -> canonical_node_name}
        pass_name: Pass 名称（用于日志）
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
    
    # Delete duplicate nodes - log each deletion
    if removed_nodes:
        logging.info(f"[{pass_name}] Removing {len(removed_nodes)} duplicate nodes")
        for node_name in removed_nodes:
            canonical = dedup_map.get(node_name, "unknown")
            logging.debug(f"[{pass_name}] Deleted: {node_name}, reason: duplicate of {canonical}")
    
    new_nodes = [n for n in optimizer.graph_def.node if n.name not in removed_nodes]
    del optimizer.graph_def.node[:]
    optimizer.graph_def.node.extend(new_nodes)
    
    # Refresh optimizer state after in-place modification
    optimizer.refresh_state()


@PassRegistry.register("common_subexpression_elimination", opt_level=1, priority=20)
class CommonSubexpressionElimination(BasePass):
    """
    Common Subexpression Elimination Pass.
    
    Eliminates duplicate nodes with identical operations, inputs, and attributes.
    Uses BasePass's iterative mode to repeatedly scan until convergence.
    """
    
    # 不应去重的操作类型
    # 这些操作要么有状态、有副作用、或者每次调用结果不同
    SKIP_OPS = {
        # === 输入节点 ===
        'Placeholder',
        'PlaceholderV2',
        'PlaceholderWithDefault',
        
        # === 变量操作（有状态）===
        'Variable',
        'VariableV2',
        'VarHandleOp',
        'ReadVariableOp',      # 每次读取可能得到不同值
        'ResourceGather',      # 从变量读取
        'ResourceGatherNd',
        'AssignVariableOp',    # 有副作用
        'AssignAddVariableOp',
        'AssignSubVariableOp',
        
        # === Identity 操作 ===
        # 由专门的 identity_removal pass 处理
        'Identity',
        'IdentityN',
        
        # === 随机操作（每次结果不同）===
        'RandomUniform',
        'RandomUniformInt',
        'RandomNormal',
        'RandomStandardNormal',
        'TruncatedNormal',
        'Multinomial',
        'RandomShuffle',
        'RandomGamma',
        'RandomPoisson',
        'StatelessRandomUniform',
        'StatelessRandomNormal',
        
        # === 副作用操作 ===
        'Print',
        'PrintV2',
        'Assert',
        'NoOp',
        
        # === 控制流操作 ===
        'Switch',
        'Merge',
        'Enter',
        'Exit',
        'NextIteration',
        'LoopCond',
        'RefSwitch',
        'RefMerge',
        'RefEnter',
        'RefExit',
        'RefNextIteration',
        
        # === 队列/栈/数组操作（有状态）===
        'QueueDequeueV2',
        'QueueEnqueueV2',
        'QueueEnqueueManyV2',
        'QueueDequeueManyV2',
        'QueueDequeueUpToV2',
        'QueueCloseV2',
        'QueueSizeV2',
        'StackPushV2',
        'StackPopV2',
        'StackCloseV2',
        'TensorArrayV3',
        'TensorArrayReadV3',
        'TensorArrayWriteV3',
        'TensorArrayGatherV3',
        'TensorArrayScatterV3',
        'TensorArraySizeV3',
        'TensorArrayCloseV3',
        
        # === 数据集操作（有状态）===
        'Iterator',
        'IteratorV2',
        'IteratorGetNext',
        'IteratorGetNextSync',
        'MakeIterator',
        
        # === 其他有状态操作 ===
        'HashTableV2',
        'LookupTableFindV2',
        'LookupTableInsertV2',
        'LookupTableSizeV2',
        'InitializeTableV2',
        'InitializeTableFromTextFileV2',
        
        # === Summary 操作（有副作用）===
        'ScalarSummary',
        'HistogramSummary',
        'ImageSummary',
        'AudioSummary',
        'MergeSummary',
        'WriteScalarSummary',
        'WriteHistogramSummary',
        'WriteImageSummary',
        'WriteAudioSummary',
        
        # === 文件操作 ===
        'WriteFile',
        'ReadFile',
        'MatchingFiles',
    }

    def __init__(self):
        super().__init__(
            name="CSE",
            optimizer_alias="cse",
            iterative=True,  # 迭代执行直到收敛
            max_iterations=100
        )
    
    def transform_once(self, optimizer, auto_cleanup=True, protected_nodes=None):
        """
        Execute a single CSE iteration.
        
        Returns:
            int: Number of duplicate nodes eliminated in this iteration
        """
        protected_set = protected_nodes or set()
        
        # Build deduplication map
        dedup_map = build_deduplication_map(optimizer, self.SKIP_OPS, protected_set)
        
        if not dedup_map:
            return 0  # No duplicates found
        
        dedup_count = len(dedup_map)
        logging.info(f"[{self.name}] Found {dedup_count} duplicate nodes")
        
        # Apply deduplication map
        apply_deduplication_map(optimizer, dedup_map, self.name)
        
        return dedup_count
