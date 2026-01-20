"""
Transform Tests - 优化 Pass 测试模块
=====================================

按照 transforms 目录结构组织的测试：

tests/transforms/
├── scalar/              # 标量优化测试
│   ├── test_identity_elim.py   # Identity 消除测试
│   └── test_cse.py             # CSE 测试
│
├── combine/             # 组合优化测试
│   └── test_concat_combine.py  # Concat 融合测试
│
└── vectorize/           # 向量化优化测试
    └── test_pack_vectorize.py  # Pack 上浮测试（暂无）
"""
