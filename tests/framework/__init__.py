"""
Framework Tests - 核心框架测试模块
===================================

测试 GraphOptimizer 核心引擎的各项功能：

模块列表：
- test_core.py              : 基础功能（节点查找、交换律匹配、Variadic 匹配、控制依赖）
- test_core_advanced.py     : 高级功能（属性匹配、CSE 签名、去重映射）
- test_control_dep_loss.py  : 控制依赖在 rewrite 过程中的保留
- test_indexed_matching.py  : 基于 op_type 的索引化模式匹配优化
- test_infrastructure.py    : OptimizationPipeline、Pass 失败回滚
- test_logging.py           : 日志系统配置和级别控制
- test_pruning.py           : 死代码消除、引用计数、Placeholder 保留
- test_variadic_alias.py    : Variadic 模式的 alias 收集功能
"""
