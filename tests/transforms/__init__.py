"""
Transform Tests
===============
Organized by backend, mirroring the source layout::

    tests/transforms/
    ├── tensorflow/
    │   ├── scalar/        # AlgebraicSimplify, ConstantFold, CSE
    │   ├── combine/       # ConcatCombine
    │   └── vectorize/     # PackVectorize
    └── torch/
        ├── test_algebraic_simplify.py
        ├── test_constant_fold.py
        ├── test_cse.py
        ├── test_matmul_fuse.py
        └── test_reshape_eliminate.py
"""
