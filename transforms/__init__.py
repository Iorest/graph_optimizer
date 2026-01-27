"""
Graph Optimization Transforms
=============================

按照 LLVM 风格组织的图优化 Pass 集合。

目录结构：
transforms/
├── scalar/              # 标量/局部优化
│   └── cse.py              # 公共子表达式消除
│
├── combine/             # 组合/融合优化
│   └── concat_combine.py   # ConcatV2 融合
│
└── vectorize/           # 向量化/批处理优化
    └── pack_vectorize.py   # Pack 上浮

Pass 执行顺序建议：
1. constant_fold      (priority=5)   - 常量折叠
2. algebraic_simplify (priority=7)   - 代数恒等式化简（包括 Identity 折叠）
3. cse                (priority=20)  - 消除公共子表达式
4. concat_combine     (priority=40)  - 融合 Concat 操作
5. pack_vectorize     (priority=50)  - 批量化相同操作
"""

# Scalar transforms
from .scalar import (
    CSEPass,
    ConstantFoldPass,
    SimplifyAddPass,
    SimplifySubPass,
    SimplifyMulPass,
    SimplifyDivPass,
    SimplifyNegPass,
    SimplifyLogicalNotPass,
    SimplifyRedundantComparisonPass,
    SimplifySelectPass,
    BypassIdentityPass,
)

# Combine transforms
from .combine import (
    ConcatCombinePass,
)

# Vectorize transforms
from .vectorize import (
    PackVectorizePass,
)

__all__ = [
    # Scalar
    'CSEPass',
    'ConstantFoldPass',
    'SimplifyAddPass',
    'SimplifySubPass',
    'SimplifyMulPass',
    'SimplifyDivPass',
    'SimplifyNegPass',
    'SimplifyLogicalNotPass',
    'SimplifyRedundantComparisonPass',
    'SimplifySelectPass',
    'BypassIdentityPass',
    # Combine
    'ConcatCombinePass',
    # Vectorize
    'PackVectorizePass',
]
