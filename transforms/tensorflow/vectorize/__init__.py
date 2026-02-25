"""
Vectorize Transforms - 向量化/批处理优化
=========================================

将多个相同操作合并为单个批量操作，提高硬件利用率。
类似 LLVM 的 SLP Vectorizer 和 Loop Vectorizer。

包含的 Pass：
- pack_vectorize.py : Pack 上浮（将相同操作批量化）

特点：
- 提高计算并行度
- 利用 SIMD/GPU 的批量计算能力
- 通常是 pipeline 的后期 Pass
- 可能增加单次计算的内存需求
"""

from .pack_vectorize import PackVectorizePass

__all__ = [
    'PackVectorizePass',
]
