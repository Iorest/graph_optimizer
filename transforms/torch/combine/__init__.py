"""
PyTorch FX Combine (graph-level) optimization transforms.
"""

from .matmul_fuse import MatmulFusePass

__all__ = [
    "MatmulFusePass",
]
