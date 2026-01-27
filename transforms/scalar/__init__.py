"""
Scalar Transforms - 标量/局部优化
==================================

对单个节点或小范围节点进行局部变换，不改变计算的并行度。
类似 LLVM 的 InstCombine、DCE、CSE 等 Pass。

包含的 Pass：
- algebraic_simplify_rules.py : 代数恒等式化简规则
- cse.py                      : 公共子表达式消除（签名去重）
- constant_fold.py            : 常量折叠

特点：
- 低开销、高收益
- 通常作为 pipeline 的早期 Pass
- 不增加也不减少计算的并行度
"""

from .cse import CSEPass
from .constant_fold import ConstantFoldPass
from .algebraic_simplify_rules import (
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

__all__ = [
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
]
