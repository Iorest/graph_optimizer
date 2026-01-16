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

from ..utils.logger import logger as logging
from ..core import PassRegistry, BasePass


@PassRegistry.register("common_subexpression_elimination", opt_level=1, priority=20)
class CommonSubexpressionElimination(BasePass):
    """
    Common Subexpression Elimination Pass.
    
    Eliminates duplicate nodes with identical operations, inputs, and attributes.
    Iteratively scans the graph until convergence (no more duplicates found).
    """
    
    def __init__(self):
        super().__init__(
            name="common_subexpression_elimination",
            optimizer_alias="cse"
        )
        # 不应去重的操作类型
        self.skip_ops = {
            # 注意：Const 节点可以去重！具有相同值的常量是真正的重复节点
            'Placeholder',   # 输入节点不应去重
            'Variable',      # 变量节点有状态，不应去重
            'VariableV2',
            'VarHandleOp',   # TF2.x 变量句柄
            'ReadVariableOp', # 读取变量的操作（每次读取可能得到不同值）
            'AssignVariableOp', # 变量赋值有副作用
            'AssignAddVariableOp', # 变量累加有副作用
            'AssignSubVariableOp', # 变量减法有副作用
            'Identity',      # Identity 节点由专门的 pass 处理
            'NoOp',          # 控制流节点
            'Assert',        # 断言节点
            'Print',         # 打印操作有副作用
            'RandomUniform', # 随机操作每次调用结果不同
            'RandomNormal',
            'RandomStandardNormal',
            'TruncatedNormal',
            'Multinomial',
            'QueueDequeueV2', # 队列操作有状态
            'QueueEnqueueV2',
            'StackPushV2',    # 栈操作有状态
            'StackPopV2',
            'TensorArrayReadV3',  # 数组操作有状态
            'TensorArrayWriteV3',
            'TensorArrayGatherV3',
            'TensorArrayScatterV3',
            # 控制流操作
            'Switch',
            'Merge',
            'Enter',
            'Exit',
            'NextIteration',
            'LoopCond',
        }
    
    def transform(self, optimizer, step=None, debug_dir=None, auto_cleanup=True, protected_nodes=None):
        """
        执行公共子表达式消除（迭代执行直到收敛）。
        
        Args:
            optimizer: GraphOptimizer 实例
            step: 当前优化步骤（用于日志）
            debug_dir: 调试输出目录
            auto_cleanup: 是否自动清理死节点
            protected_nodes: 受保护的节点集合
            
        Returns:
            GraphDef: 优化后的图定义（与其他 Pass 保持一致）
        """
        import time
        from ..utils import save_graph
        
        logging.info(f"[{self.name}] Starting (iterative until convergence)...")
        original_node_count = len(optimizer.nodes)
        start_time = time.time()
        
        total_removed = 0
        iteration = 0
        max_iterations = 100  # 安全限制，防止无限循环
        
        # 准备受保护节点集合
        protected_set = set(protected_nodes or [])
        
        # Iterate until no more duplicates
        while True:
            iteration += 1
            
            # Check iteration limit
            if iteration > max_iterations:
                logging.warning(
                    f"[{self.name}] Reached maximum iterations ({max_iterations}), stopping"
                )
                break
            
            current_node_count = len(optimizer.nodes)
            
            logging.debug(f"[{self.name}] Iteration {iteration}: scanning {current_node_count} nodes")
            
            # Build deduplication map
            dedup_map = self.build_deduplication_map(optimizer, self.skip_ops, protected_set)
            
            if not dedup_map:
                logging.debug(f"[{self.name}] Iteration {iteration}: no duplicates found, converged")
                break
            
            dedup_count = len(dedup_map)
            total_removed += dedup_count
            # INFO: 找到的 pattern 数量
            logging.info(f"[{self.name}] Iteration {iteration}: found {dedup_count} duplicate nodes")
            
            # Apply deduplication map
            self.apply_deduplication_map(optimizer, dedup_map)
        
        # Report final results
        final_node_count = len(optimizer.nodes)
        duration = time.time() - start_time
        
        logging.info(
            f"[{self.name}] Completed in {duration:.3f}s ({iteration} iterations). "
            f"Nodes: {original_node_count} -> {final_node_count} "
            f"(removed {total_removed} duplicates)"
        )
        
        # Save debug info if needed
        if debug_dir and step is not None:
            import os
            filename = f"{step:02d}_{self.name}.pb"
            file_path = os.path.join(debug_dir, filename)
            save_graph(optimizer.graph_def, file_path)
            logging.debug(f"[{self.name}] Saved debug graph to {file_path}")
        
        # 返回 GraphDef（与其他 Pass 保持一致）
        return optimizer.graph_def
