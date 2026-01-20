"""
Combine Transforms - 组合/融合优化
===================================

将多个操作融合为单个操作，减少中间张量和内存拷贝。
类似 LLVM 的 InstCombine 中的 pattern combining。

包含的 Pass：
- concat_combine.py : ConcatV2 融合（相同 axis 的嵌套 Concat 展开）

特点：
- 减少中间节点数量
- 降低内存带宽需求
- 通常在 CSE 之后、向量化之前运行
"""

from .concat_combine import ConcatCombinePass

__all__ = [
    'ConcatCombinePass',
]
