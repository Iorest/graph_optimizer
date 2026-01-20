"""
Infrastructure Tests - 基础设施测试
====================================

测试内容：
1. test_pipeline_and_rollback - OptimizationPipeline 执行和 Pass 失败回滚

验证：
- Pass 执行失败时自动回滚到前一个稳定状态
- 后续 Pass 继续执行
- 最终输出图不包含失败 Pass 的改动

Mock Passes：
- MockFailingPass  : 模拟失败的 Pass（添加节点后抛异常）
- MockSuccessPass  : 模拟成功的 Pass（正常添加节点）
"""

import unittest
import os
import tensorflow.compat.v1 as tf
from graph_optimizer.core import (
    BasePass,
    PassRegistry,
)
from graph_optimizer.runner import OptimizationPipeline

tf.disable_v2_behavior()


# --- Mock passes for testing infrastructure ---
class MockFailingPass(BasePass):
    def transform(self, optimizer, step=None, debug_dir=None, **kwargs):
        optimizer.graph_def.node.extend([tf.NodeDef(name="BAD_NODE", op="NoOp")])
        optimizer.load_state(optimizer.graph_def)
        raise RuntimeError("Fail")


class MockSuccessPass(BasePass):
    def transform(self, optimizer, step=None, debug_dir=None, **kwargs):
        optimizer.graph_def.node.extend([tf.NodeDef(name="GOOD_NODE", op="NoOp")])
        optimizer.load_state(optimizer.graph_def)
        return optimizer.graph_def


class TestInfrastructure(unittest.TestCase):
    def setUp(self):
        if "mock_fail" not in PassRegistry._registered_passes:
            PassRegistry.register("mock_fail", opt_level=1, priority=10)(
                MockFailingPass
            )
        if "mock_success" not in PassRegistry._registered_passes:
            PassRegistry.register("mock_success", opt_level=1, priority=20)(
                MockSuccessPass
            )

        self.input_graph = "test_infra_input.pb"
        self.output_graph = "test_infra_output.pb"
        graph = tf.GraphDef()
        n = graph.node.add()
        n.name = "Input"
        n.op = "Placeholder"
        with open(self.input_graph, "wb") as f:
            f.write(graph.SerializeToString())

    def tearDown(self):
        for f in [self.input_graph, self.output_graph]:
            if os.path.exists(f):
                os.remove(f)

    def test_pipeline_and_rollback(self):
        pipeline = OptimizationPipeline(
            input_graph=self.input_graph,
            output_graph=self.output_graph,
            passes=["mock_fail", "mock_success"],
        )
        pipeline.run()

        graph = tf.GraphDef()
        with open(self.output_graph, "rb") as f:
            graph.ParseFromString(f.read())

        self.assertIn("GOOD_NODE", [n.name for n in graph.node])
        self.assertNotIn("BAD_NODE", [n.name for n in graph.node])


if __name__ == "__main__":
    unittest.main()
