"""
Pack Vectorize Optimization Pass

================================================================================
Pass 注册信息
================================================================================
Registration:
    Name: "pack_vectorize"
    Optimization Level: 2 (aggressive optimization)
    Priority: 50
    Iterative: Yes (runs until convergence)
    Max Iterations: 100

Class:
    PackVectorizePass (inherits from PatternRewritePass)

================================================================================
目的 (Purpose)
================================================================================
将 Pack 节点向上"穿透"其输入 Op，减少重复计算，合并相同操作。
通过批量化操作来提升计算效率。

================================================================================
算法 (Algorithm)
================================================================================
核心思想：将 Pack 节点向上"穿透"其输入 Op，减少重复计算。

原始模式：
    Input_0 → Op_0 → ┐
    Input_1 → Op_1 → ├→ Pack → Consumer
    ...              │
    Input_N → Op_N → ┘

优化后：
    Input_0 → ┐
    Input_1 → ├→ Pack_new → Op_batched → Consumer
    ...       │
    Input_N → ┘

关键步骤：
1. 识别模式：Pack 节点的所有输入都来自相同类型的 Op
2. 验证可行性：
   - 检查 Op 类型是否支持上浮
   - 检查其他输入是否可以 broadcast 或也需要 Pack
   - 检查属性是否兼容
3. 执行变换：
   - 为每个输入创建新的 Pack 节点
   - 调整 Op 的输出形状（增加一个维度）
   - 可能需要类型转换（如 MatMul → BatchMatMulV2）
4. 清理：删除原始的单独 Op 节点

================================================================================
复杂度 (Complexity)
================================================================================
Time: O(N * M)
    - N 是节点数量
    - M 是每个节点的输入数量
    - 每次迭代需要匹配所有 Pack 模式

Space: O(N)
    - 需要存储匹配的模式和生成的新节点

================================================================================
支持的操作类型 (Supported Operations)
================================================================================

1. 一元逐元素操作 (Unary Elementwise):
   - Relu, Relu6, Elu, Selu, Sigmoid, Tanh
   - Exp, Log, Sqrt, Rsqrt, Square
   - Abs, Neg, Sign
   - 规则：直接上浮，输出形状增加一个维度

2. 二元逐元素操作 (Binary Elementwise):
   - Add, Sub, Mul, Div, RealDiv, FloorDiv
   - Maximum, Minimum
   - 规则：其他输入支持 broadcast 或也需要 Pack

3. 矩阵操作 (Matrix Operations):
   - MatMul → BatchMatMulV2 (需要类型转换)
   - BatchMatMul, BatchMatMulV2
   - 规则：检查 transpose 属性兼容性
   - 共享权重：直接 broadcast
   - 不同权重：Pack(axis=0) + BatchMatMulV2 + Transpose（需额外 Transpose 节点）

4. 维度操作 (Dimension Operations):
   - Reshape, ExpandDims, Squeeze
   - Transpose (需要调整 perm)
   - 规则：需要调整维度相关参数
   - 注意：ADJUST 策略为预留功能，当前未实现参数调整逻辑

5. 切片操作 (Slice Operations):
   - StridedSlice
   - 规则：可通过消除优化（Pack + StridedSlice → 直接引用输入）
   - 注意：非消除路径的参数调整为预留功能，当前未实现

================================================================================
限制与约束 (Limitations)
================================================================================

不支持的情况：
1. Pack 的输入是 Placeholder（输入节点不应被优化）
2. Op 有多个消费者且不都是可合并的 Pack
3. Op 的其他输入无法 broadcast 且值不相同（无法统一处理）
4. 控制流操作（Switch, Merge, Enter, Exit）- 通过白名单隐式拒绝
5. 有状态操作（Variable, ReadVariableOp）- 通过白名单隐式拒绝
6. 需要维护执行顺序的操作（Print, Assert）- 通过白名单隐式拒绝

属性兼容性：
- 所有被 Pack 的 Op 必须有相同的关键属性（如 transpose, dtype, _output_shapes 等）
- 跳过比对的属性：`name`, `_class`, `_device`（实例特定的元数据）

形状要求：
- Pack 节点的所有输入必须有兼容的形状（能够在指定 axis 上 Pack）
- 上浮后会在指定的 axis 维度上增加 batch 维度（支持任意合法的 axis 值）

================================================================================
示例 (Examples)
================================================================================

Example 1: Relu 上浮
    原图：
        a = MatMul(x1, w)
        b = MatMul(x2, w)
        c = MatMul(x3, w)
        ra = Relu(a)
        rb = Relu(b)
        rc = Relu(c)
        pack = Pack([ra, rb, rc], axis=0)  # shape: [3, D]

    优化后：
        a = MatMul(x1, w)
        b = MatMul(x2, w)
        c = MatMul(x3, w)
        pack_input = Pack([a, b, c], axis=0)  # shape: [3, D]
        relu_batched = Relu(pack_input)       # shape: [3, D]

    节点减少：7 → 5 (减少29%)
    计算优化：3次独立Relu → 1次批量Relu

Example 2: Add + broadcast 上浮
    原图：
        a1 = Add(x1, bias)
        a2 = Add(x2, bias)  # 相同的 bias
        pack = Pack([a1, a2], axis=0)

    优化后：
        pack_x = Pack([x1, x2], axis=0)      # shape: [2, D]
        add_batched = Add(pack_x, bias)      # bias 自动 broadcast

    节点减少：3 → 2 (减少33%)

Example 3: MatMul → BatchMatMulV2 上浮
    原图：
        m1 = MatMul(x1, w)
        m2 = MatMul(x2, w)  # 相同的权重
        pack = Pack([m1, m2], axis=0)

    优化后：
        pack_x = Pack([x1, x2], axis=0)         # shape: [2, D1, D2]
        bmm = BatchMatMulV2(pack_x, w)          # shape: [2, D1, D3]
        # TensorFlow 自动将 w (D2, D3) broadcast 到 (2, D2, D3)

    节点减少：3 → 2 (减少33%，且计算效率提升)
    类型转换：MatMul → BatchMatMulV2

    注意：依赖 TensorFlow 的自动 broadcasting，不创建额外的 ExpandDims 节点

Example 4: 无法上浮的情况
    原图：
        a = Add(x1, bias1)  # 不同的 bias
        b = Add(x2, bias2)
        pack = Pack([a, b], axis=0)

    无法优化：bias1 != bias2，无法 broadcast

================================================================================
预期效果 (Expected Results)
================================================================================
- 节点减少：20-40% (在有大量相似操作的模型中)
- 运行时性能提升：10-30% (通过批量化操作)
- 内存使用：可能略微增加（中间 Pack 节点）
- 编译时间：增加 10-20%

适用场景：
- 多塔模型（用户塔/物品塔结构相同）
- 多任务学习（共享相同的特征处理）
- Ensemble 模型（多个相同结构的子模型）

================================================================================
相关 Pass (Related Passes)
================================================================================
- common_subexpression_elimination: CSE 可以先消除重复计算，为 Pack hoisting 创造条件
- constant_folding: 常量折叠可以简化 Pack 的输入
- dead_code_elimination: 清理上浮后不再使用的节点

建议执行顺序：
1. CSE (消除重复，可以发现更多可以的优化)
2. Pack Hoisting (批量化)
3. DCE (清理死代码)

================================================================================
"""

from ...core import (
    PatternRewritePass,
    PassRegistry,
    RewriteResult,
    Op,
    Any,
    Variadic,
    get_attr_value,
)
from ...utils import create_node
from ...utils.logger import logger as logging
from tensorflow.core.framework import tensor_shape_pb2, attr_value_pb2, types_pb2
import tensorflow.compat.v1 as tf
from typing import Dict, List, Set, Optional, Tuple, Any as AnyType
from enum import Enum, auto


# ============================================================================
# Op 配置：定义每种 Op 的穿透规则
# ============================================================================


class InputStrategy(Enum):
    """输入处理策略"""

    PACK = auto()  # 主输入：总是 Pack
    BROADCAST = auto()  # 其他输入：共享时直接 broadcast
    BROADCAST_OR_PACK = auto()  # 其他输入：共享时 broadcast，不同时 Pack
    CONST = auto()  # 常量输入：必须相同值
    ADJUST = auto()  # 需要调整的输入（如维度操作的 axis）


class OpCategory(Enum):
    """Op 类别"""

    UNARY_ELEMENTWISE = auto()  # 一元逐元素：Relu, Sigmoid 等
    BINARY_ELEMENTWISE = auto()  # 二元逐元素：Add, Mul 等
    MATMUL = auto()  # 矩阵乘法：MatMul, BatchMatMul
    SLICE = auto()  # 切片操作：StridedSlice
    DIMENSION = auto()  # 维度操作：Reshape, Transpose, Squeeze, ExpandDims


class OpHoistConfig:
    """
    单个 Op 的穿透配置。

    将所有穿透策略统一到此类中：
    - 输入处理策略（Pack/Broadcast/Const）
    - 算子转换规则
    - Shape 处理规则
    - 特殊优化（如 StridedSlice 消除）
    """

    def __init__(
        self,
        op_type: str,
        category: OpCategory,
        # 输入配置
        main_input_idx: int = 0,
        other_inputs: Dict[int, InputStrategy] = None,  # {idx: strategy}
        # 算子转换
        target_op: str = None,
        attr_transform: Dict[str, str] = None,  # {old_attr: new_attr}
        remove_attrs: Set[str] = None,
        # Shape 相关
        shape_preserving: bool = True,  # 输出 shape 是否与主输入相同
        other_input_pack_axis: int = 0,  # 其他输入 Pack 时使用的 axis
        skip_broadcast_check: bool = False,  # 是否跳过 broadcast 兼容性检查（如 MatMul）
        # 特殊处理
        can_eliminate: bool = False,  # 是否可消除（如 StridedSlice）
        needs_axis_adjust: bool = False,  # 是否需要调整 axis 属性
        axis_attr_name: str = None,  # axis 属性名（如 'squeeze_dims', 'axis'）
    ):
        self.op_type = op_type
        self.category = category
        self.main_input_idx = main_input_idx
        self.other_inputs = other_inputs or {}
        self.target_op = target_op or op_type
        self.attr_transform = attr_transform or {}
        self.remove_attrs = remove_attrs or set()
        self.shape_preserving = shape_preserving
        self.other_input_pack_axis = other_input_pack_axis
        self.skip_broadcast_check = skip_broadcast_check
        self.can_eliminate = can_eliminate
        self.needs_axis_adjust = needs_axis_adjust
        self.axis_attr_name = axis_attr_name


# ============================================================================
# Op 配置注册表
# ============================================================================


def _create_unary_config(op_type: str) -> OpHoistConfig:
    """创建一元逐元素操作配置"""
    return OpHoistConfig(
        op_type=op_type,
        category=OpCategory.UNARY_ELEMENTWISE,
        main_input_idx=0,
        shape_preserving=True,
    )


def _create_binary_config(op_type: str) -> OpHoistConfig:
    """创建二元逐元素操作配置"""
    return OpHoistConfig(
        op_type=op_type,
        category=OpCategory.BINARY_ELEMENTWISE,
        main_input_idx=0,
        other_inputs={1: InputStrategy.BROADCAST_OR_PACK},
        shape_preserving=True,
    )


# 支持穿透的 Op 配置
HOIST_CONFIGS: Dict[str, OpHoistConfig] = {
    # ========== 一元逐元素操作 ==========
    # 特点：只有主输入，输出 shape 与输入相同
    # 处理：直接穿透，无需额外处理
    "Relu": _create_unary_config("Relu"),
    "Relu6": _create_unary_config("Relu6"),
    "Sigmoid": _create_unary_config("Sigmoid"),
    "Tanh": _create_unary_config("Tanh"),
    "Exp": _create_unary_config("Exp"),
    "Log": _create_unary_config("Log"),
    "Neg": _create_unary_config("Neg"),
    "Abs": _create_unary_config("Abs"),
    "Square": _create_unary_config("Square"),
    "Sqrt": _create_unary_config("Sqrt"),
    "Rsqrt": _create_unary_config("Rsqrt"),
    "Softplus": _create_unary_config("Softplus"),
    "Elu": _create_unary_config("Elu"),
    "Selu": _create_unary_config("Selu"),
    "Erf": _create_unary_config("Erf"),
    "Cast": _create_unary_config("Cast"),
    "Identity": _create_unary_config("Identity"),
    # ========== 二元逐元素操作 ==========
    # 特点：两个输入，支持 broadcast，输出 shape 由 broadcast 规则决定
    # 处理：主输入 Pack，其他输入根据是否共享决定 broadcast 或 Pack
    "Add": _create_binary_config("Add"),
    "AddV2": _create_binary_config("AddV2"),
    "Sub": _create_binary_config("Sub"),
    "Mul": _create_binary_config("Mul"),
    "Div": _create_binary_config("Div"),
    "RealDiv": _create_binary_config("RealDiv"),
    "Maximum": _create_binary_config("Maximum"),
    "Minimum": _create_binary_config("Minimum"),
    "Pow": _create_binary_config("Pow"),
    "FloorDiv": _create_binary_config("FloorDiv"),
    "Mod": _create_binary_config("Mod"),
    "SquaredDifference": _create_binary_config("SquaredDifference"),
    # ========== BiasAdd ==========
    # 特点：BiasAdd 要求特定输入格式，增维后需转换为 AddV2
    # 处理：转换算子 + 可能需要 reshape bias
    "BiasAdd": OpHoistConfig(
        op_type="BiasAdd",
        category=OpCategory.BINARY_ELEMENTWISE,
        main_input_idx=0,
        other_inputs={1: InputStrategy.BROADCAST_OR_PACK},
        target_op="AddV2",
        remove_attrs={"data_format"},
        shape_preserving=True,
    ),
    # ========== MatMul 系列 ==========
    # 特点：(M,K) @ (K,N) -> (M,N)，shape 会改变
    # 处理：MatMul -> BatchMatMulV2，属性转换
    # 注意：权重输入不需要 broadcast 检查，因为 MatMul 的两个输入有不同的 shape 语义
    "MatMul": OpHoistConfig(
        op_type="MatMul",
        category=OpCategory.MATMUL,
        main_input_idx=0,
        other_inputs={1: InputStrategy.BROADCAST_OR_PACK},
        target_op="BatchMatMulV2",
        attr_transform={"transpose_a": "adj_x", "transpose_b": "adj_y"},
        remove_attrs={"transpose_a", "transpose_b"},
        shape_preserving=False,
        skip_broadcast_check=True,
    ),
    "BatchMatMul": OpHoistConfig(
        op_type="BatchMatMul",
        category=OpCategory.MATMUL,
        main_input_idx=0,
        other_inputs={1: InputStrategy.BROADCAST_OR_PACK},
        shape_preserving=False,
        skip_broadcast_check=True,
    ),
    "BatchMatMulV2": OpHoistConfig(
        op_type="BatchMatMulV2",
        category=OpCategory.MATMUL,
        main_input_idx=0,
        other_inputs={1: InputStrategy.BROADCAST_OR_PACK},
        shape_preserving=False,
        skip_broadcast_check=True,
    ),
    # ========== StridedSlice ==========
    # 特点：可能完全消除（当做连续单元素切片时）
    # 处理：尝试消除，否则正常穿透
    "StridedSlice": OpHoistConfig(
        op_type="StridedSlice",
        category=OpCategory.SLICE,
        main_input_idx=0,
        other_inputs={
            1: InputStrategy.CONST,  # begin
            2: InputStrategy.CONST,  # end
            3: InputStrategy.CONST,  # strides
        },
        shape_preserving=False,
        can_eliminate=True,
    ),
    # ========== 维度操作 ==========
    # 特点：需要调整 axis/shape 等参数
    # 注意：ADJUST 策略为预留功能，当前未实现参数调整逻辑，
    #       这些 Op 会因 is_shared 检查失败而被跳过
    "Reshape": OpHoistConfig(
        op_type="Reshape",
        category=OpCategory.DIMENSION,
        main_input_idx=0,
        other_inputs={1: InputStrategy.ADJUST},  # shape 需要调整（预留）
        shape_preserving=False,
    ),
    "Squeeze": OpHoistConfig(
        op_type="Squeeze",
        category=OpCategory.DIMENSION,
        main_input_idx=0,
        shape_preserving=False,
        needs_axis_adjust=True,  # 预留功能，当前未实现
        axis_attr_name="squeeze_dims",
    ),
    "ExpandDims": OpHoistConfig(
        op_type="ExpandDims",
        category=OpCategory.DIMENSION,
        main_input_idx=0,
        other_inputs={1: InputStrategy.ADJUST},  # axis 需要调整（预留）
        shape_preserving=False,
    ),
    "Transpose": OpHoistConfig(
        op_type="Transpose",
        category=OpCategory.DIMENSION,
        main_input_idx=0,
        other_inputs={1: InputStrategy.ADJUST},  # perm 需要调整（预留）
        shape_preserving=False,
    ),
}


# ============================================================================
# Broadcast 工具函数
# ============================================================================


def can_broadcast(shape_a: List[int], shape_b: List[int]) -> bool:
    """
    检查 shape_a 是否可以广播到与 shape_b 兼容。

    广播规则（从右向左对齐）：
    - 每个维度必须相等，或其中一个是 1
    - 较短的 shape 会在前面补 1
    - 动态维度 (-1) 总是兼容
    """
    if shape_a is None or shape_b is None:
        return True  # 未知 shape 假设可以 broadcast

    a = list(reversed(shape_a))
    b = list(reversed(shape_b))

    for i in range(min(len(a), len(b))):
        dim_a, dim_b = a[i], b[i]
        # 动态维度总是可以广播
        if dim_a == -1 or dim_b == -1:
            continue
        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return False

    return True


def try_reshape_for_broadcast(
    input_shape: List[int], target_shape: List[int], pack_axis: int
) -> Optional[List[int]]:
    """
    尝试找到一个 reshape 后的 shape 使其能广播到 target_shape。

    当 Pack 增加了一个维度后，原本的输入可能需要 reshape 才能正确广播。
    返回 None 表示不需要 reshape（已经可以 broadcast），
    返回 shape 表示需要 reshape 到该 shape。
    """
    if input_shape is None or target_shape is None:
        return None

    # 如果已经可以广播，不需要 reshape
    if can_broadcast(input_shape, target_shape):
        return None

    # 尝试在不同位置插入维度 1
    for insert_pos in range(len(input_shape) + 1):
        new_shape = list(input_shape)
        new_shape.insert(insert_pos, 1)
        if can_broadcast(new_shape, target_shape):
            return new_shape

    # 尝试插入多个维度 1
    diff = len(target_shape) - len(input_shape)
    if diff > 0:
        new_shape = list(input_shape)
        for _ in range(diff):
            insert_pos = min(pack_axis + 1, len(new_shape))
            new_shape.insert(insert_pos, 1)
        if can_broadcast(new_shape, target_shape):
            return new_shape

    return None


def insert_batch_dim(
    shape: List[int], axis: int, batch_size: int
) -> Optional[List[int]]:
    """在指定 axis 位置插入 batch 维度"""
    if shape is None:
        return None

    new_shape = list(shape)
    if axis < 0:
        axis = len(new_shape) + 1 + axis
    axis = max(0, min(axis, len(new_shape)))
    new_shape.insert(axis, batch_size)
    return new_shape


def compute_effective_pack_axis(shape: List[int], pack_axis: int) -> int:
    """计算有效的 pack axis，处理标量和越界情况"""
    if shape is None:
        return pack_axis

    rank = len(shape)
    if rank == 0:  # 标量只能在 axis=0 Pack
        return 0
    if pack_axis < 0:
        return max(0, rank + 1 + pack_axis)
    return min(pack_axis, rank)


# ============================================================================
# 输入分析
# ============================================================================


class InputAnalysisResult:
    """单个输入的分析结果"""

    def __init__(self):
        self.strategy: InputStrategy = None
        # actions: 'pack', 'broadcast', 'reshape_broadcast', 'const', 'pack_transpose',
        #          'pack_axis0', 'transpose_broadcast', 'skip' ('adjust' 预留)
        self.action: str = None
        self.nodes: List[str] = []
        self.shape: List[int] = None
        self.pack_shape: List[int] = None  # Pack 后的 shape
        self.reshape_target: List[int] = None  # 需要 reshape 到的目标 shape
        self.is_shared: bool = False  # 所有分支输入是否相同
        self.const_value = None  # 常量值（如果是 CONST 策略）
        self.transpose_perm: List[int] = None  # 用于 transpose_broadcast


class HoistAnalysis:
    """Pack 穿透分析结果"""

    def __init__(self):
        self.can_hoist: bool = False
        self.reason: str = None
        self.config: OpHoistConfig = None
        self.branch_ops: List[tf.NodeDef] = []
        self.n_branches: int = 0
        self.pack_axis: int = 0
        self.effective_pack_axis: int = 0
        # 从后往前数的 axis（负索引），在上浮过程中保持不变
        # 例如 axis=-2 表示倒数第二个维度
        self.pack_axis_from_end: int = 0
        self.inputs: Dict[int, InputAnalysisResult] = {}

        # ========== 统一的 shape 推导结果 ==========
        # 原始 Pack 节点的输出 shape（倒推法的基准）
        self.original_pack_output_shape: List[int] = None
        # 主输入的原始 shape（单个分支的输入 shape）
        self.main_input_shape: List[int] = None
        # 主输入 Pack 后的 shape（用于其他输入的 broadcast 检查）
        self.batched_main_input_shape: List[int] = None
        # 上浮后操作的输出 shape（最终结果，用于创建新节点）
        self.batched_output_shape: List[int] = None
        # 上浮后的 batch axis（n_branches 所在的维度）
        self.output_batch_axis: int = 0

        # 特殊优化
        self.can_eliminate: bool = False  # StridedSlice 消除
        self.eliminate_info: Dict = None
        # Pack 合并
        self.merge_packs: Set[str] = None  # 需要合并的其他 Pack 节点
        # MatMul 不同权重：需要最终 Transpose（BMM 输出后）
        self.needs_final_transpose: bool = False
        self.final_transpose_perm: List[int] = None
        # MatMul 不同权重：主输入需要前置 Transpose（当已是 [batch, N, K] 格式时）
        self.needs_pre_transpose: bool = False
        self.pre_transpose_perm: List[int] = None


# ============================================================================
# 节点创建工具
# ============================================================================


def make_output_shapes_attr(shapes: List[List[int]]) -> attr_value_pb2.AttrValue:
    """创建 _output_shapes 属性"""
    attr = attr_value_pb2.AttrValue()
    for shape in shapes:
        shape_proto = attr.list.shape.add()
        for dim in shape:
            shape_proto.dim.add().size = dim
    return attr


def make_int_attr(value: int) -> attr_value_pb2.AttrValue:
    """创建整数属性"""
    attr = attr_value_pb2.AttrValue()
    attr.i = value
    return attr


def make_type_attr(dtype) -> attr_value_pb2.AttrValue:
    """创建类型属性"""
    attr = attr_value_pb2.AttrValue()
    attr.type = dtype
    return attr


def make_tensor_attr_from_list(
    values: List[int], dtype=types_pb2.DT_INT32
) -> attr_value_pb2.AttrValue:
    """从列表创建 tensor 属性"""
    import numpy as np
    from tensorflow.python.framework import tensor_util

    attr = attr_value_pb2.AttrValue()
    np_array = np.array(values, dtype=np.int32)
    attr.tensor.CopyFrom(tensor_util.make_tensor_proto(np_array))
    return attr


def create_const_node(name: str, values: List[int], dtype=types_pb2.DT_INT32):
    """创建常量节点（用于 Transpose 的 perm 等）"""

    import numpy as np
    from tensorflow.python.framework import tensor_util
    from tensorflow.core.framework import node_def_pb2

    node = node_def_pb2.NodeDef()
    node.op = "Const"
    node.name = name

    # dtype 属性
    node.attr["dtype"].type = dtype

    # value 属性
    np_array = np.array(
        values, dtype=np.int32 if dtype == types_pb2.DT_INT32 else np.float32
    )
    node.attr["value"].tensor.CopyFrom(tensor_util.make_tensor_proto(np_array))

    return node


def get_dtype_from_node(optimizer, node_name: str):
    """获取节点的数据类型"""
    if not node_name:
        return types_pb2.DT_FLOAT

    base_name = node_name.split(":")[0].lstrip("^")
    node = optimizer.nodes.get(base_name)
    if not node:
        return types_pb2.DT_FLOAT

    type_attrs = ["T", "dtype", "output_type", "DstT", "SrcT", "Tparams"]
    for attr_name in type_attrs:
        if attr_name in node.attr:
            dtype = node.attr[attr_name].type
            if dtype and dtype != types_pb2.DT_INVALID:
                return dtype

    if node.op == "Const" and "value" in node.attr:
        tensor = node.attr["value"].tensor
        if tensor.dtype and tensor.dtype != types_pb2.DT_INVALID:
            return tensor.dtype

    return types_pb2.DT_FLOAT


# ============================================================================
# Pack 上浮优化器
# ============================================================================


@PassRegistry.register("pack_vectorize", opt_level=3, priority=50)
class PackVectorizePass(PatternRewritePass):
    """
    Pack 上浮优化 Pass。

    根据 OpHoistConfig 配置决定如何处理各种输入：
    1. 主输入：总是 Pack
    2. 其他输入：根据策略 broadcast/pack/const
    3. 特殊优化：StridedSlice 消除等
    """

    def __init__(self):
        pattern = Op("Pack", Variadic(Any(), alias="inputs"), alias="pack")
        super().__init__(
            pattern=pattern,
            rewriter=self._hoist_pack_rewriter,
            name="PackVectorize",
            optimizer_alias="pack_vectorize_pass",
        )
        self._hoisted_cache = {}
        # 记录需要后处理的 shape_preserving 节点
        self._shape_preserving_nodes = []
        # 记录每个创建的 Pack 节点的 pack_axis_from_end（从后往前数的 axis）
        # 这样在后续分析时可以继承这个值，而不是重新计算
        self._pack_axis_from_end_map: Dict[str, int] = {}

    def _hoist_pack_rewriter(self, match, optimizer):
        """主入口：分析 -> 执行 -> 返回结果"""
        pack_node = match.matched_nodes["pack"]
        protected = getattr(optimizer, "protected_nodes", set())

        if pack_node.name in protected:
            return None

        # 分析
        analysis = self._analyze_hoist(optimizer, pack_node, protected)
        if not analysis.can_hoist:
            return None

        # 执行
        if analysis.can_eliminate:
            result = self._execute_elimination(optimizer, pack_node, analysis)
        else:
            result = self._execute_hoist(optimizer, pack_node, analysis)

        if not result:
            return None

        # 构建映射
        node_mapping = {pack_node.name: result["output_node"]}
        if analysis.merge_packs:
            for p in analysis.merge_packs:
                node_mapping[p] = result["output_node"]

        return RewriteResult(
            new_nodes=result["new_nodes"],
            replaced_nodes=[n for n in result["replaced_nodes"] if n != pack_node.name],
            node_mapping=node_mapping,
        )

    # ========================================================================
    # 工具方法
    # ========================================================================

    @staticmethod
    def _clean_input_name(name: str) -> str:
        return name.split(":")[0].lstrip("^")

    # ========================================================================
    # 分析阶段
    # ========================================================================

    def _analyze_hoist(self, optimizer, pack_node, protected_set) -> HoistAnalysis:
        """
        分析 Pack 是否可以上浮。
        流程：验证分支 → 计算 shape → 检查消除 → 分析输入
        """
        ha = HoistAnalysis()
        ha.pack_axis = get_attr_value(pack_node.attr.get("axis")) or 0
        data_inputs = [inp for inp in pack_node.input if not inp.startswith("^")]
        ha.n_branches = len(data_inputs)

        if ha.n_branches < 2:
            return self._fail(ha, "Pack has less than 2 inputs")

        # 1. 验证分支 Op
        err = self._validate_branches(
            optimizer, pack_node, data_inputs, protected_set, ha
        )
        if err:
            return self._fail(ha, err)

        # 2. 计算 shape
        cfg = ha.config
        main_name = self._clean_input_name(ha.branch_ops[0].input[cfg.main_input_idx])
        ha.main_input_shape = optimizer.get_node_shape(main_name)
        ha.effective_pack_axis = compute_effective_pack_axis(
            ha.main_input_shape, ha.pack_axis
        )
        ha.original_pack_output_shape = optimizer.get_node_shape(pack_node.name)

        # 计算从后往前数的 axis（负索引）
        # 优先从 _pack_axis_from_end_map 获取（继承上一轮创建的值）
        # 这样在上浮过程中 pack_axis_from_end 保持不变
        if pack_node.name in self._pack_axis_from_end_map:
            ha.pack_axis_from_end = self._pack_axis_from_end_map[pack_node.name]
            logging.debug(
                f"[{self.name}] Inherited pack_axis_from_end={ha.pack_axis_from_end} for {pack_node.name}"
            )
        elif ha.original_pack_output_shape:
            ndim = len(ha.original_pack_output_shape)
            ha.pack_axis_from_end = ha.pack_axis - ndim  # 转为负索引
        else:
            # 如果没有 shape 信息，假设 3 维
            ha.pack_axis_from_end = ha.pack_axis - 3

        self._infer_shape(optimizer, ha)

        # 3. 检查 StridedSlice 消除（可提前返回）
        if cfg.can_eliminate and cfg.op_type == "StridedSlice":
            elim = self._check_elimination(optimizer, ha.branch_ops, ha.n_branches)
            if elim:
                ha.can_eliminate, ha.eliminate_info, ha.can_hoist = True, elim, True
                return ha

        # 4. 分析输入
        err = self._analyze_inputs(optimizer, ha)
        if err:
            return self._fail(ha, err)

        ha.can_hoist = True
        return ha

    def _fail(self, ha: HoistAnalysis, reason: str) -> HoistAnalysis:
        """设置失败原因并返回"""
        ha.reason = reason
        return ha

    def _validate_branches(
        self, optimizer, pack_node, data_inputs, protected_set, ha: HoistAnalysis
    ) -> Optional[str]:
        """验证所有分支 Op，返回错误信息或 None"""
        op_type, first_node = None, None
        # 不需要检查一致性的属性（元数据或输出相关）
        skip_attrs = {"name", "_class", "_device"}

        for inp in data_inputs:
            name = self._clean_input_name(inp)
            node = optimizer.nodes.get(name)

            # 基本验证
            if not node or node.op == "Placeholder" or node.op not in HOIST_CONFIGS:
                return f"Invalid input: {name}"
            if name in protected_set:
                return f"Protected: {name}"

            # Op 类型一致性
            if op_type is None:
                op_type, first_node = node.op, node
            elif node.op != op_type:
                return "Mixed op types"
            else:
                # 检查关键属性一致性（如 transpose, dtype 等）
                for attr in set(first_node.attr.keys()) - skip_attrs:
                    if first_node.attr.get(attr) != node.attr.get(attr):
                        return f"Attribute mismatch: {attr}"

            # 多消费者检查：允许消费者都是 Pack 的情况
            others = [
                c for c in optimizer.consumers.get(name, []) if c != pack_node.name
            ]
            if others:
                if self._can_merge_packs(optimizer, pack_node, others, data_inputs):
                    ha.merge_packs = (ha.merge_packs or set()) | set(others)
                else:
                    # 检查是否所有消费者都是 Pack（可通过后续复用机制处理）
                    all_packs = all(
                        optimizer.nodes.get(c) and optimizer.nodes.get(c).op == "Pack"
                        for c in others
                    )
                    if not all_packs:
                        return f"Has other consumers: {name}"

            ha.branch_ops.append(node)

        ha.config = HOIST_CONFIGS[op_type]
        return None

    def _analyze_inputs(self, optimizer, ha: HoistAnalysis) -> Optional[str]:
        """分析所有输入，返回错误信息或 None"""
        cfg = ha.config

        # 主输入
        main = InputAnalysisResult()
        main.strategy = InputStrategy.PACK
        main.nodes = [
            self._clean_input_name(op.input[cfg.main_input_idx]) for op in ha.branch_ops
        ]
        main.shape = ha.main_input_shape
        raw = [op.input[cfg.main_input_idx].lstrip("^") for op in ha.branch_ops]
        main.is_shared = len(set(raw)) == 1
        main.action = "broadcast" if main.is_shared else "pack"
        main.pack_shape = None if main.is_shared else ha.batched_main_input_shape
        ha.inputs[cfg.main_input_idx] = main

        # 其他输入
        for idx, strategy in cfg.other_inputs.items():
            info = self._analyze_other_input(optimizer, ha, idx, strategy)
            if info.action == "incompatible":
                return f"Input {idx} incompatible"
            ha.inputs[idx] = info

            # pack_transpose 策略：不同权重 MatMul
            if info.action == "pack_transpose":
                # BMM 要求：[N, batch, K] @ [N, K, D] -> [N, batch, D]
                #
                # 核心思想：保持原始 Pack axis，通过 Transpose 适配 BMM
                # - Pack 仍然使用原始 axis（如 axis=1），输出 [batch, N, K]
                # - 在 BMM 前添加 Transpose: [batch, N, K] -> [N, batch, K]
                # - BMM 输出 [N, batch, D]
                # - 在 BMM 后添加 Transpose: [N, batch, D] -> [batch, N, D]
                #
                # 这样的好处：
                # 1. StridedSlice 消除可以正常工作（slice_axis == pack_axis）
                # 2. 连续的 Transpose 可以相互抵消

                pack_axis = ha.pack_axis  # 使用原始 Pack axis

                if main.is_shared:
                    # 主输入是 shared，检查其 shape 格式
                    main_shape = main.shape
                    if main_shape and len(main_shape) == 3:
                        n_pos = None
                        for i, dim in enumerate(main_shape):
                            if dim == ha.n_branches:
                                n_pos = i
                                break

                        if n_pos is not None and n_pos != 0:
                            # N 不在 axis=0，需要前置 Transpose
                            ha.needs_pre_transpose = True
                            ndim = len(main_shape)
                            # 把 n_pos 移到 axis=0: [batch, N, K] -> [N, batch, K]
                            ha.pre_transpose_perm = [n_pos] + [
                                i for i in range(ndim) if i != n_pos
                            ]
                            main.action = "transpose_broadcast"
                            main.transpose_perm = ha.pre_transpose_perm
                        elif n_pos == 0:
                            # 已经是 [N, batch, K] 格式
                            main.action = "broadcast"
                        else:
                            # 无法确定 N 的位置，需要 Pack + Transpose
                            main.action = "pack_then_transpose"
                    else:
                        main.action = "pack_then_transpose"
                else:
                    # 主输入不是 shared，使用原始 axis Pack，然后 Transpose
                    main.action = "pack_then_transpose"

                # 设置前置 Transpose（Pack 后，BMM 前）
                if main.action == "pack_then_transpose" and pack_axis != 0:
                    ha.needs_pre_transpose = True
                    # 从 [batch, N, K] (N at pack_axis) -> [N, batch, K]
                    ndim = (
                        len(ha.batched_main_input_shape)
                        if ha.batched_main_input_shape
                        else 3
                    )
                    ha.pre_transpose_perm = [pack_axis] + [
                        i for i in range(ndim) if i != pack_axis
                    ]

                # 设置后置 Transpose（BMM 后，恢复原始格式）
                if pack_axis != 0:
                    ha.needs_final_transpose = True
                    # 从 [N, batch, D] -> [batch, N, D] (N 移回 pack_axis)
                    ndim = (
                        len(ha.batched_output_shape) if ha.batched_output_shape else 3
                    )
                    # 把 axis=0 移到 pack_axis 位置
                    perm = list(range(ndim))
                    perm.remove(0)
                    perm.insert(pack_axis, 0)
                    ha.final_transpose_perm = perm

        return None

    def _infer_shape(self, optimizer, analysis: HoistAnalysis):
        """统一的 shape 推导

        核心原则：使用从后往前数的 axis（负索引）
        - pack_axis_from_end 在上浮过程中保持不变
        - output_batch_axis 根据输出维度和 pack_axis_from_end 计算
        """
        config, n = analysis.config, analysis.n_branches
        pack_shape = analysis.original_pack_output_shape
        main_shape = analysis.main_input_shape

        # 从原始 Pack shape 验证并推导
        if pack_shape:
            # 计算期望的 batch axis（从后往前数）
            expected_batch_axis = len(pack_shape) + analysis.pack_axis_from_end
            expected_batch_axis = max(0, min(expected_batch_axis, len(pack_shape) - 1))

            # 验证 N 维度在正确位置
            # 如果 pack_axis 和 expected_batch_axis 不一致，说明原始 Pack 的 axis 与继承的 pack_axis_from_end 不匹配
            # 这时候应该使用正向推导，而不是倒推
            if (
                analysis.pack_axis < len(pack_shape)
                and pack_shape[analysis.pack_axis] == n
            ):
                if analysis.pack_axis == expected_batch_axis:
                    # axis 一致，可以直接使用倒推
                    analysis.batched_output_shape = list(pack_shape)
                    analysis.output_batch_axis = expected_batch_axis
                    analysis.batched_main_input_shape = (
                        list(pack_shape)
                        if config.shape_preserving
                        else insert_batch_dim(main_shape, analysis.output_batch_axis, n)
                        if main_shape
                        else None
                    )
                    return
                # axis 不一致，使用正向推导（下面的代码会处理）

        # 正向推导
        branch_shape = optimizer.get_node_shape(analysis.branch_ops[0].name)
        if branch_shape:
            # 根据分支输出维度计算新的 pack axis
            ndim = len(branch_shape) + 1  # Pack 后维度 +1
            analysis.output_batch_axis = ndim + analysis.pack_axis_from_end
            # 确保 axis 有效
            analysis.output_batch_axis = max(
                0, min(analysis.output_batch_axis, ndim - 1)
            )

            analysis.batched_output_shape = insert_batch_dim(
                branch_shape, analysis.output_batch_axis, n
            )
            analysis.batched_main_input_shape = (
                insert_batch_dim(main_shape, analysis.output_batch_axis, n)
                if main_shape
                else None
            )
        else:
            # 无法推导，使用原始 axis
            analysis.output_batch_axis = analysis.pack_axis

    def _analyze_other_input(
        self, optimizer, ha: HoistAnalysis, idx: int, strategy: InputStrategy
    ) -> InputAnalysisResult:
        """分析非主输入"""
        result = InputAnalysisResult()
        result.strategy = strategy

        nodes, shapes = [], []
        for op in ha.branch_ops:
            if idx >= len(op.input):
                result.action = "incompatible"
                return result
            name = self._clean_input_name(op.input[idx])
            nodes.append(name)
            shapes.append(optimizer.get_node_shape(name))

        result.nodes = nodes
        result.shape = shapes[0]
        raw = [op.input[idx].lstrip("^") for op in ha.branch_ops]
        result.is_shared = len(set(raw)) == 1

        if strategy == InputStrategy.CONST:
            values = []
            for name in nodes:
                node = optimizer.nodes.get(name)
                if node and node.op == "Const":
                    v = get_attr_value(node.attr.get("value"))
                    values.append(tuple(v) if isinstance(v, (list, tuple)) else v)
                else:
                    values.append(None)
            result.action = (
                "const"
                if len(set(values)) == 1 and values[0] is not None
                else "incompatible"
            )

        elif strategy == InputStrategy.ADJUST:
            # ADJUST 策略为预留功能，当前未实现参数调整逻辑
            # 无论 is_shared 如何，都返回 incompatible 跳过这些 Op
            result.action = "incompatible"

        elif strategy in (InputStrategy.BROADCAST, InputStrategy.BROADCAST_OR_PACK):
            if result.is_shared:
                if ha.config.skip_broadcast_check or can_broadcast(
                    result.shape, ha.batched_main_input_shape
                ):
                    result.action = "broadcast"
                else:
                    reshape = try_reshape_for_broadcast(
                        result.shape, ha.batched_main_input_shape, ha.output_batch_axis
                    )
                    if reshape:
                        result.action = "reshape_broadcast"
                        result.reshape_target = reshape
                    else:
                        result.action = "incompatible"
            else:
                if strategy == InputStrategy.BROADCAST:
                    result.action = "incompatible"
                elif len(set(tuple(s) if s else None for s in shapes)) > 1:
                    result.action = "incompatible"
                # MatMul 类 Op：不同权重使用 pack_transpose 策略
                # 关键：Pack 仍使用原始 axis，然后 Transpose 到 [N, K, D] 格式
                elif ha.config.category == OpCategory.MATMUL:
                    result.action = "pack_transpose"
                    # Pack 使用原始 axis
                    result.pack_axis = ha.output_batch_axis
                    result.pack_shape = insert_batch_dim(
                        result.shape, ha.output_batch_axis, ha.n_branches
                    )
                    # 记录需要的 Transpose：从 [K, N, D]（N at pack_axis）到 [N, K, D]
                    if ha.output_batch_axis != 0:
                        # 需要 Transpose 把 N 移到 axis=0
                        ndim = len(result.pack_shape) if result.pack_shape else 3
                        result.transpose_perm = [ha.output_batch_axis] + [
                            i for i in range(ndim) if i != ha.output_batch_axis
                        ]
                else:
                    result.action = "pack"
                    # 选择 pack axis：
                    # 1. 对于 shape_preserving 操作，尝试找到能 broadcast 的 axis
                    # 2. 对于非 shape_preserving 操作，使用配置的 axis
                    if (
                        ha.config.shape_preserving
                        and result.shape
                        and ha.batched_main_input_shape
                    ):
                        # 尝试找到一个 axis 使得 pack 后的 shape 能 broadcast 到主输入 shape
                        best_axis = None
                        for try_axis in range(len(result.shape) + 1):
                            try_shape = insert_batch_dim(
                                result.shape, try_axis, ha.n_branches
                            )
                            if can_broadcast(try_shape, ha.batched_main_input_shape):
                                best_axis = try_axis
                                break
                        if best_axis is not None:
                            pack_axis = best_axis
                            result.pack_shape = insert_batch_dim(
                                result.shape, pack_axis, ha.n_branches
                            )
                        else:
                            # 没有合适的 axis，使用原始 pack axis
                            pack_axis = ha.output_batch_axis
                            result.pack_shape = insert_batch_dim(
                                result.shape, pack_axis, ha.n_branches
                            )
                            if not can_broadcast(
                                result.pack_shape, ha.batched_main_input_shape
                            ):
                                result.action = "incompatible"
                    else:
                        pack_axis = (
                            ha.output_batch_axis
                            if ha.config.shape_preserving
                            else ha.config.other_input_pack_axis
                        )
                        result.pack_shape = insert_batch_dim(
                            result.shape, pack_axis, ha.n_branches
                        )
                        if (
                            not ha.config.skip_broadcast_check
                            and ha.batched_main_input_shape
                            and result.pack_shape
                        ):
                            if not can_broadcast(
                                result.pack_shape, ha.batched_main_input_shape
                            ):
                                result.action = "incompatible"
                    # 记录选择的 pack axis
                    result.pack_axis = (
                        pack_axis if "pack_axis" in dir() else ha.output_batch_axis
                    )

        return result

    def _can_merge_packs(self, optimizer, pack_node, others, expected_inputs) -> bool:
        """检查其他消费者是否是可合并的 Pack"""
        axis = get_attr_value(pack_node.attr.get("axis")) or 0
        expected = [self._clean_input_name(i) for i in expected_inputs]
        for name in others:
            node = optimizer.nodes.get(name)
            if not node or node.op != "Pack":
                return False
            if (get_attr_value(node.attr.get("axis")) or 0) != axis:
                return False
            inputs = [
                self._clean_input_name(i) for i in node.input if not i.startswith("^")
            ]
            if inputs != expected:
                return False
        return True

    def _check_elimination(self, optimizer, branch_ops, n_branches) -> Optional[Dict]:
        """检查 StridedSlice 消除"""
        source = self._clean_input_name(branch_ops[0].input[0])
        for op in branch_ops[1:]:
            if self._clean_input_name(op.input[0]) != source:
                return None

        indices, slice_axis = [], None
        for op in branch_ops:
            begin_node = optimizer.nodes.get(self._clean_input_name(op.input[1]))
            if not begin_node or begin_node.op != "Const":
                return None
            begin = get_attr_value(begin_node.attr.get("value"))
            if hasattr(begin, "tolist"):
                begin = begin.tolist()
            shrink = get_attr_value(op.attr.get("shrink_axis_mask")) or 0
            for i, b in enumerate(begin if isinstance(begin, list) else [begin]):
                if shrink & (1 << i):
                    if slice_axis is None:
                        slice_axis = i
                    elif slice_axis != i:
                        return None
                    indices.append(b)
                    break

        if sorted(indices) != list(range(n_branches)):
            return None

        source_shape = optimizer.get_node_shape(source)
        if source_shape and slice_axis is not None and slice_axis < len(source_shape):
            if source_shape[slice_axis] == n_branches and indices == list(
                range(n_branches)
            ):
                return {
                    "eliminate": True,
                    "slice_axis": slice_axis,
                    "indices_ordered": True,
                }
        return None

    # ========================================================================
    # 执行阶段
    # ========================================================================

    def _execute_elimination(
        self, optimizer, pack_node, analysis: HoistAnalysis
    ) -> Optional[Dict]:
        """执行 StridedSlice 消除

        消除条件：
        1. 所有 StridedSlice 都从同一个 source 切分
        2. source 的 slice_axis 维度大小等于 n_branches
        3. slice_axis 从后往前数等于 pack_axis_from_end（确保消除后 shape 格式一致）
        """
        source = self._clean_input_name(analysis.branch_ops[0].input[0])
        for op in analysis.branch_ops[1:]:
            if self._clean_input_name(op.input[0]) != source:
                return None

        source_shape = optimizer.get_node_shape(source)
        info = analysis.eliminate_info
        if (
            source_shape
            and info.get("indices_ordered")
            and info["slice_axis"] < len(source_shape)
        ):
            slice_axis = info["slice_axis"]

            # 关键检查：将 slice_axis 转为从后往前数的索引，与 pack_axis_from_end 比较
            slice_axis_from_end = slice_axis - len(source_shape)
            if slice_axis_from_end != analysis.pack_axis_from_end:
                logging.debug(
                    f"[{self.name}] Cannot eliminate: slice_axis_from_end={slice_axis_from_end} != pack_axis_from_end={analysis.pack_axis_from_end}"
                )
                return None

            if source_shape[slice_axis] == analysis.n_branches:
                logging.info(
                    f"[{self.name}] Eliminated Pack + StridedSlice -> {source}"
                )
                return {
                    "new_nodes": [],
                    "output_node": source,
                    "replaced_nodes": [pack_node.name]
                    + [op.name for op in analysis.branch_ops],
                }
        return None

    def _execute_hoist(self, optimizer, pack_node, analysis: HoistAnalysis) -> Dict:
        """执行 Pack 上浮"""
        new_nodes = []
        config = analysis.config
        main_info = analysis.inputs[config.main_input_idx]
        dtype = get_dtype_from_node(optimizer, main_info.nodes[0])

        # 1. 准备主输入
        if main_info.action == "broadcast":
            main_input = main_info.nodes[0]
        elif main_info.action == "transpose_broadcast":
            # 主输入已经是 batched 但格式需要 Transpose
            main_input = self._create_transpose(
                pack_node.name,
                main_info.nodes[0],
                main_info.transpose_perm,
                dtype,
                main_info.shape,
                optimizer,
                new_nodes,
            )
        elif main_info.action == "pack_then_transpose":
            # MatMul 不同权重：先 Pack（保持原始 axis），再 Transpose
            # 1) Pack 使用原始 axis，传递 pack_axis_from_end 以便继承
            pack_output = self._create_pack(
                pack_node.name,
                main_info.nodes,
                analysis.output_batch_axis,
                analysis.n_branches,
                dtype,
                analysis.batched_main_input_shape,
                new_nodes,
                pack_axis_from_end=analysis.pack_axis_from_end,
            )
            # 2) 如果需要 Transpose（N 不在 axis=0）
            if analysis.needs_pre_transpose and analysis.pre_transpose_perm:
                transposed_shape = (
                    [
                        analysis.batched_main_input_shape[p]
                        for p in analysis.pre_transpose_perm
                    ]
                    if analysis.batched_main_input_shape
                    else None
                )
                main_input = self._create_transpose(
                    pack_node.name,
                    pack_output,
                    analysis.pre_transpose_perm,
                    dtype,
                    analysis.batched_main_input_shape,
                    optimizer,
                    new_nodes,
                )
            else:
                main_input = pack_output
        else:
            # 默认 Pack 操作，传递 pack_axis_from_end 以便继承
            main_input = self._create_pack(
                pack_node.name,
                main_info.nodes,
                analysis.output_batch_axis,
                analysis.n_branches,
                dtype,
                analysis.batched_main_input_shape,
                new_nodes,
                pack_axis_from_end=analysis.pack_axis_from_end,
            )

        # 2. 准备其他输入
        other_inputs = {}
        for idx, info in analysis.inputs.items():
            if idx == config.main_input_idx:
                continue
            if info.action == "broadcast":
                other_inputs[idx] = info.nodes[0]
            elif info.action == "reshape_broadcast":
                other_inputs[idx] = self._create_reshape(
                    pack_node.name,
                    info.nodes[0],
                    info.reshape_target,
                    optimizer,
                    new_nodes,
                )
            elif info.action == "pack":
                # 使用分析阶段确定的 pack_axis（考虑了 broadcast 兼容性）
                # 其他输入的 Pack 也使用同样的 pack_axis_from_end
                pack_axis = getattr(info, "pack_axis", analysis.output_batch_axis)
                other_inputs[idx] = self._create_pack(
                    pack_node.name,
                    info.nodes,
                    pack_axis,
                    analysis.n_branches,
                    get_dtype_from_node(optimizer, info.nodes[0]),
                    info.pack_shape,
                    new_nodes,
                    pack_axis_from_end=analysis.pack_axis_from_end,
                )
            elif info.action == "pack_transpose":
                # MatMul 不同权重：Pack 使用原始 axis，然后 Transpose 到 [N, K, D]
                pack_axis = getattr(info, "pack_axis", analysis.output_batch_axis)
                pack_output = self._create_pack(
                    pack_node.name,
                    info.nodes,
                    pack_axis,
                    analysis.n_branches,
                    get_dtype_from_node(optimizer, info.nodes[0]),
                    info.pack_shape,
                    new_nodes,
                    pack_axis_from_end=analysis.pack_axis_from_end,
                )
                # 如果需要 Transpose
                if hasattr(info, "transpose_perm") and info.transpose_perm:
                    other_inputs[idx] = self._create_transpose(
                        pack_node.name,
                        pack_output,
                        info.transpose_perm,
                        get_dtype_from_node(optimizer, info.nodes[0]),
                        info.pack_shape,
                        optimizer,
                        new_nodes,
                    )
                else:
                    other_inputs[idx] = pack_output
            else:
                # 'const'：所有分支使用相同常量，直接使用第一个节点
                other_inputs[idx] = info.nodes[0]

        # 3. 创建批量操作（检查缓存避免重复）
        first_op = analysis.branch_ops[0]
        batched_inputs, control_deps = [], []
        for i, inp in enumerate(first_op.input):
            if inp.startswith("^"):
                control_deps.append(inp)
            elif i == config.main_input_idx:
                batched_inputs.append(main_input)
            elif i in other_inputs:
                batched_inputs.append(other_inputs[i])
        batched_inputs.extend(control_deps)

        # 检查是否已存在相同的 batched op
        cache_key = (config.target_op, tuple(batched_inputs))
        if cache_key in self._hoisted_cache:
            batched_name = self._hoisted_cache[cache_key]
            logging.debug(f"[{self.name}] Reusing batched op {batched_name}")
        else:
            attrs = {}
            for name, val in first_op.attr.items():
                if name not in config.remove_attrs and name not in {
                    "_output_shapes",
                    "shape",
                }:
                    attrs[config.attr_transform.get(name, name)] = val

            # 对于 pack_transpose 模式，输出 shape 是 [N, batch, D]
            if analysis.needs_final_transpose and analysis.batched_output_shape:
                # 交换前两维：[batch, N, ...] -> [N, batch, ...]
                transposed_shape = [
                    analysis.batched_output_shape[1],
                    analysis.batched_output_shape[0],
                ] + list(analysis.batched_output_shape[2:])
                attrs["_output_shapes"] = make_output_shapes_attr([transposed_shape])
            elif analysis.batched_output_shape:
                attrs["_output_shapes"] = make_output_shapes_attr(
                    [analysis.batched_output_shape]
                )

            batched_name = self.make_unique_node_name(pack_node.name, config.target_op)
            new_nodes.append(
                create_node(config.target_op, batched_name, batched_inputs, attrs)
            )
            self._hoisted_cache[cache_key] = batched_name

        # 4. 如果需要，添加最终 Transpose
        output_name = batched_name
        if analysis.needs_final_transpose:
            transpose_name = self.make_unique_node_name(pack_node.name, "Transpose")
            perm = analysis.final_transpose_perm or [1, 0, 2]
            transpose_attrs = {
                "T": make_type_attr(dtype),
                "Tperm": make_type_attr(types_pb2.DT_INT32),
            }
            if analysis.batched_output_shape:
                transpose_attrs["_output_shapes"] = make_output_shapes_attr(
                    [analysis.batched_output_shape]
                )

            # 创建 perm 常量节点
            perm_name = self.make_unique_node_name(pack_node.name, "transpose_perm")
            perm_const = create_const_node(perm_name, perm, types_pb2.DT_INT32)
            new_nodes.append(perm_const)

            # 创建 Transpose 节点
            new_nodes.append(
                create_node(
                    "Transpose",
                    transpose_name,
                    [batched_name, perm_name],
                    transpose_attrs,
                )
            )
            output_name = transpose_name
            logging.info(f"[{self.name}] Added Transpose for MatMul different weights")

        logging.info(
            f"[{self.name}] Hoisted Pack through {analysis.n_branches} x {config.op_type} -> {config.target_op}"
        )

        replaced = [pack_node.name] + [op.name for op in analysis.branch_ops]
        if analysis.merge_packs:
            replaced.extend(analysis.merge_packs)
            logging.info(
                f"[{self.name}] Merged {len(analysis.merge_packs)} duplicate Pack nodes"
            )

        return {
            "new_nodes": new_nodes,
            "output_node": output_name,
            "replaced_nodes": replaced,
        }

    def _create_pack(
        self,
        root,
        nodes,
        axis,
        n,
        dtype,
        shape,
        new_nodes,
        pack_axis_from_end: int = None,
    ) -> str:
        """创建或复用 Pack 节点

        Args:
            pack_axis_from_end: 从后往前数的 axis（负索引），会被记录以便后续分析继承
        """
        key = (tuple(nodes), axis)
        if key in self._hoisted_cache:
            return self._hoisted_cache[key]

        name = self.make_unique_node_name(root, "Pack")
        attrs = {
            "N": make_int_attr(n),
            "axis": make_int_attr(axis),
            "T": make_type_attr(dtype),
        }
        if shape:
            attrs["_output_shapes"] = make_output_shapes_attr([shape])
        new_nodes.append(create_node("Pack", name, nodes, attrs))
        self._hoisted_cache[key] = name

        # 记录 pack_axis_from_end，以便后续分析继承
        if pack_axis_from_end is not None:
            self._pack_axis_from_end_map[name] = pack_axis_from_end
            logging.debug(
                f"[{self.name}] Created Pack {name} with axis={axis}, shape={shape}, pack_axis_from_end={pack_axis_from_end}"
            )
        else:
            logging.debug(
                f"[{self.name}] Created Pack {name} with axis={axis}, shape={shape}"
            )
        return name

    def _create_reshape(
        self, root, input_node, target_shape, optimizer, new_nodes
    ) -> str:
        """创建 Reshape 节点"""
        shape_name = self.make_unique_node_name(root, "Const")
        new_nodes.append(
            create_node(
                "Const",
                shape_name,
                attr={
                    "value": make_tensor_attr_from_list(target_shape),
                    "dtype": make_type_attr(types_pb2.DT_INT32),
                },
            )
        )

        reshape_name = self.make_unique_node_name(root, "Reshape")
        dtype = get_dtype_from_node(optimizer, input_node)
        attrs = {
            "T": make_type_attr(dtype),
            "Tshape": make_type_attr(types_pb2.DT_INT32),
        }
        if target_shape:
            attrs["_output_shapes"] = make_output_shapes_attr([target_shape])
        new_nodes.append(
            create_node("Reshape", reshape_name, [input_node, shape_name], attrs)
        )
        return reshape_name

    def _create_transpose(
        self, root, input_node, perm, dtype, input_shape, optimizer, new_nodes
    ) -> str:
        """创建 Transpose 节点（用于前置 Transpose）"""
        # 计算输出 shape
        output_shape = None
        if input_shape:
            output_shape = [input_shape[p] for p in perm]

        # 创建 perm 常量节点
        perm_name = self.make_unique_node_name(root, "pre_transpose_perm")
        perm_const = create_const_node(perm_name, perm, types_pb2.DT_INT32)
        new_nodes.append(perm_const)

        # 创建 Transpose 节点
        transpose_name = self.make_unique_node_name(root, "pre_Transpose")
        transpose_attrs = {
            "T": make_type_attr(dtype),
            "Tperm": make_type_attr(types_pb2.DT_INT32),
        }
        if output_shape:
            transpose_attrs["_output_shapes"] = make_output_shapes_attr([output_shape])

        new_nodes.append(
            create_node(
                "Transpose", transpose_name, [input_node, perm_name], transpose_attrs
            )
        )
        logging.info(
            f"[{self.name}] Added pre-Transpose for MatMul different weights (shared batched input)"
        )
        return transpose_name
