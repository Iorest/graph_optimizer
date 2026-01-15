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


@PassRegistry.register("common_subexpression_elimination", opt_level=1, priority=60)
class CommonSubexpressionElimination(BasePass):
    """
    Common Subexpression Elimination Pass.
    
    Eliminates duplicate nodes with identical operations, inputs, and attributes.
    Iteratively scans the graph until convergence (no more duplicates found).
    """
    
    def __init__(self):
        super().__init__(
            name="COMMON_SUBEXPRESSION_ELIMINATION",
            optimizer_alias="cse"
        )
        # 不应去重的操作类型
        self.skip_ops = {
            # 注意：Const 节点可以去重！具有相同值的常量是真正的重复节点
            'Placeholder',   # 输入节点不应去重
            'Variable',      # 变量节点有状态，不应去重
            'VariableV2',
            'Identity',      # Identity 节点由专门的 pass 处理
            'NoOp',          # 控制流节点
            'Assert',        # 断言节点
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
        
        logging.info(f"[{self.name}] Starting graph optimization pass (iterative until convergence)...")
        original_node_count = len(optimizer.nodes)
        start_time = time.time()
        
        total_removed = 0
        iteration = 0
        
        # 迭代执行直到没有新的重复节点
        while True:
            iteration += 1
            current_node_count = len(optimizer.nodes)
            
            logging.info(f"[{self.name}] Iteration {iteration}: scanning {current_node_count} nodes...")
            
            # 构建去重映射
            dedup_map = self.build_deduplication_map(optimizer, self.skip_ops)
            
            if not dedup_map:
                logging.info(f"[{self.name}] Iteration {iteration}: no duplicates found, converged.")
                break
            
            dedup_count = len(dedup_map)
            total_removed += dedup_count
            logging.info(f"[{self.name}] Iteration {iteration}: found {dedup_count} duplicate nodes")
            
            # 显示一些示例（仅在第一次迭代或有显著发现时）
            if iteration == 1 or dedup_count > 10:
                sample_size = min(3, dedup_count)
                sample_items = list(dedup_map.items())[:sample_size]
                for dup_name, canonical_name in sample_items:
                    dup_node = optimizer.nodes.get(dup_name)
                    if dup_node:
                        logging.info(
                            f"[{self.name}]   {dup_name} (op: {dup_node.op}) -> {canonical_name}"
                        )
                
                if dedup_count > sample_size:
                    logging.info(f"[{self.name}]   ... and {dedup_count - sample_size} more")
            
            # 应用去重映射
            self.apply_deduplication_map(optimizer, dedup_map)
        
        # 报告最终结果
        final_node_count = len(optimizer.nodes)
        duration = time.time() - start_time
        
        if total_removed > 0:
            logging.info(
                f"[{self.name}] Optimization finished in {duration:.3f}s after {iteration} iterations. "
                f"Nodes: {original_node_count} -> {final_node_count} "
                f"(removed {total_removed} duplicates)"
            )
        else:
            logging.info(
                f"[{self.name}] Optimization finished in {duration:.3f}s. "
                f"Nodes: {original_node_count} -> {final_node_count} (no duplicates found)"
            )
        
        # 保存调试信息（如果需要）
        if debug_dir and step is not None:
            import os
            filename = f"{step:02d}_{self.name}.pb"
            file_path = os.path.join(debug_dir, filename)
            save_graph(optimizer.graph_def, file_path)
        
        # 返回 GraphDef（与其他 Pass 保持一致）
        return optimizer.graph_def
