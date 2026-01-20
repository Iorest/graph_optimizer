"""
Graph Optimizer Test Suite
===========================

测试模块组织：

tests/
├── framework/           # 核心框架测试
│   ├── test_core.py              # 基础引擎测试（节点查找、模式匹配）
│   ├── test_control_dep_loss.py  # 控制依赖保留测试
│   ├── test_indexed_matching.py  # 索引化模式匹配测试
│   ├── test_infrastructure.py    # Pipeline 和回滚机制测试
│   ├── test_logging.py           # 日志系统测试
│   ├── test_pruning.py           # 图裁剪测试
│   └── test_variadic_alias.py    # Variadic 模式别名测试
│
└── transforms/          # 优化 Pass 测试
    ├── scalar/                   # 标量优化测试
    │   ├── test_identity_elim.py     # Identity 消除测试
    │   └── test_cse.py               # CSE 测试
    │
    ├── combine/                  # 组合优化测试
    │   └── test_concat_combine.py    # Concat 融合测试
    │
    └── vectorize/                # 向量化优化测试
        └── (预留)

运行测试：
    # 推荐使用项目脚本运行全部测试
    sh run_test.sh

    # 使用 pytest 运行全部
    python -m pytest tests/ -v

    # 运行特定模块
    python -m pytest tests/framework/ -v
    python -m pytest tests/transforms/ -v

    # 运行特定类别 Pass 测试
    python -m pytest tests/transforms/scalar/ -v
    python -m pytest tests/transforms/combine/ -v
"""
