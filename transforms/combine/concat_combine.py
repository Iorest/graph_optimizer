"""
Concat Combine Pass (concat_combine)

================================================================================
Pass 注册信息
================================================================================
Registration:
    Name: "concat_combine"
    Optimization Level: 3 (aggressive optimization)
    Priority: 40
    Iterative: Yes (runs until convergence)

Class:
    ConcatCombinePass (inherits from PatternRewritePass)

================================================================================
目的 (Purpose)
================================================================================
融合多级嵌套的 ConcatV2 操作，减少中间张量的创建和内存拷贝。
当多个 ConcatV2 在相同轴上串联时，可以合并为单个 ConcatV2 操作。

================================================================================
算法 (Algorithm)
================================================================================
识别并融合具有相同 axis 的嵌套 ConcatV2：

原始模式:
    a ─────────────────────┐
    b ─────────────────────┤
    ConcatV2([c, d], axis) ┼→ ConcatV2([a, b, inner, e], axis) → out
    e ─────────────────────┘

优化后:
    a ──┐
    b ──┤
    c ──┼→ ConcatV2([a, b, c, d, e], axis) → out
    d ──┤
    e ──┘

关键步骤:
1. 匹配外层 ConcatV2 节点
2. 检查每个输入是否也是 ConcatV2
3. 验证内层 ConcatV2 的 axis 与外层相同
4. 验证 shape 兼容性（非 concat 维度必须匹配）
5. 展开内层 ConcatV2 的输入到外层

================================================================================
融合条件 (Fusion Conditions)
================================================================================
内层 ConcatV2 可以被融合当且仅当:

1. Axis 相同:
   - 内层和外层的 concat axis 必须相同
   - 使用规范化后的 axis（处理负数索引）

2. Shape 兼容:
   - 非 concat 维度的大小必须相同
   - 动态维度 (-1) 视为兼容

3. 无副作用:
   - 内层 ConcatV2 只被外层使用（否则不能删除）

================================================================================
复杂度 (Complexity)
================================================================================
Time: O(N * M)
    - N 是 ConcatV2 节点数量
    - M 是每个 ConcatV2 的平均输入数量

Space: O(M)
    - 需要存储新的输入列表

================================================================================
示例 (Examples)
================================================================================
示例 1 - 基本融合:
    原图:
        inner = ConcatV2([a, b], axis=1)
        outer = ConcatV2([c, inner, d], axis=1)
    
    优化后:
        outer = ConcatV2([c, a, b, d], axis=1)

示例 2 - 多级融合:
    原图:
        level1 = ConcatV2([a, b], axis=0)
        level2 = ConcatV2([level1, c], axis=0)
        level3 = ConcatV2([level2, d], axis=0)
    
    第一次迭代:
        level2 = ConcatV2([a, b, c], axis=0)
        level3 = ConcatV2([level2, d], axis=0)
    
    第二次迭代:
        level3 = ConcatV2([a, b, c, d], axis=0)

示例 3 - 不融合（axis 不同）:
    原图:
        inner = ConcatV2([a, b], axis=0)  # axis=0
        outer = ConcatV2([c, inner], axis=1)  # axis=1 ≠ 0
    
    不变（axis 不匹配）

================================================================================
与其他 Pass 的关系
================================================================================
- 应该在 CSE 之后运行（可能暴露更多融合机会）
- 应该在 Pack Vectorize 之前运行（减少节点数量）
- 可以与 Identity Elimination 配合（移除中间 Identity）
"""

from ...core import Op, Any, Variadic, PassRegistry, PatternRewritePass
from ...utils import create_node
from ...utils.logger import logger as logging


@PassRegistry.register("concat_combine", opt_level=3, priority=40)
class ConcatCombinePass(PatternRewritePass):
    """
    Fuses multi-level ConcatV2 operations with the same axis.
    
    Transform: ConcatV2([..., ConcatV2([a, b], axis), ...], axis)
    Into: ConcatV2([..., a, b, ...], axis)
    """

    def __init__(self):
        pattern = Op(
            "ConcatV2",
            Variadic(Any(), alias="inputs"),
            Op("Const", alias="axis"),
            alias="root",
        )
        super().__init__(
            pattern, 
            self._combine_concat, 
            name="ConcatCombine",
            optimizer_alias="concat_combine"
        )

    def _combine_concat(self, match, optimizer):
        root = match.matched_nodes["root"]
        axis_node = match.matched_nodes["axis"]

        root_rank = optimizer.get_node_rank(root)
        raw_axis = optimizer.get_node_attr(axis_node, "value")
        axis_val = optimizer.canonicalize_axis(raw_axis, root_rank)

        if axis_val is None:
            return None

        # Separate root inputs into data, axis, and control
        root_all_inputs = root.input
        root_control_inputs = [i for i in root_all_inputs if i.startswith("^")]
        root_data_inputs = [i for i in root_all_inputs if not i.startswith("^")]
        
        # ConcatV2 inputs are [values..., axis]. 
        # Last data input is axis.
        if not root_data_inputs:
             return None
             
        root_axis_input = root_data_inputs[-1]
        root_values_inputs = root_data_inputs[:-1]

        new_values_inputs = []
        new_control_inputs = set(root_control_inputs) # Use set to avoid duplicates
        changed = False

        for input_name in root_values_inputs:
            base_name = self.clean_input_name(input_name)
            should_fuse = False
            
            if base_name in optimizer.nodes:
                input_node = optimizer.nodes[base_name]
                if input_node.op == "ConcatV2":
                    logging.debug(f"[ConcatCombine] Found inner ConcatV2: {input_node.name}")
                    
                    # Analyze inner inputs
                    inner_all_inputs = input_node.input
                    inner_control_inputs = [i for i in inner_all_inputs if i.startswith("^")]
                    inner_data_inputs = [i for i in inner_all_inputs if not i.startswith("^")]
                    
                    if len(inner_data_inputs) >= 2:
                        inner_axis_name = self.clean_input_name(inner_data_inputs[-1])
                        inner_axis_node = optimizer.nodes.get(inner_axis_name)
                        
                        if inner_axis_node and inner_axis_node.op == "Const":
                            inner_rank = optimizer.get_node_rank(input_node)
                            raw_inner_axis = optimizer.get_node_attr(inner_axis_node, "value")
                            inner_axis_val = optimizer.canonicalize_axis(raw_inner_axis, inner_rank)

                            if inner_axis_val == axis_val:
                                # Shape check
                                root_shape = optimizer.get_node_shape(root)
                                inner_shape = optimizer.get_node_shape(input_node)
                                
                                match_fail = False
                                if root_shape and inner_shape:
                                    if len(root_shape) != len(inner_shape):
                                        match_fail = True
                                    else:
                                        for i in range(len(root_shape)):
                                            if i != axis_val:
                                                if (
                                                    root_shape[i] != inner_shape[i]
                                                    and root_shape[i] != -1
                                                    and inner_shape[i] != -1
                                                ):
                                                    match_fail = True
                                                    break
                                
                                if not match_fail:
                                    should_fuse = True
                                    # Add inner values (exclude axis)
                                    new_values_inputs.extend(inner_data_inputs[:-1])
                                    # Collect inner control deps
                                    for ctrl in inner_control_inputs:
                                        new_control_inputs.add(ctrl)
            
            if should_fuse:
                changed = True
            else:
                new_values_inputs.append(input_name)

        if not changed:
            return None

        # Reconstruct inputs: [new_values..., axis, new_controls...]
        final_inputs = new_values_inputs + [root_axis_input] + sorted(list(new_control_inputs))
        
        new_node = create_node("ConcatV2", root.name, inputs=final_inputs, attr=root.attr)
        new_node.attr["N"].i = len(new_values_inputs)
        
        logging.info(f"[ConcatCombine] Combined: {root.name} ({len(root_values_inputs)} -> {len(new_values_inputs)} value inputs)")
        return [new_node]



