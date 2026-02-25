"""
PyTorch FX Graph Optimization Transforms
"""

from .scalar.algebraic_simplify import TorchAlgebraicSimplifyPass
from .scalar.constant_fold import TorchConstantFoldPass
from .scalar.cse import TorchCSEPass
from .combine.matmul_fuse import MatmulFusePass

__all__ = [
    "TorchAlgebraicSimplifyPass",
    "TorchConstantFoldPass",
    "TorchCSEPass",
    "MatmulFusePass",
]
