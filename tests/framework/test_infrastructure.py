"""
Pipeline and registry infrastructure tests.

Covers: OptimizationPipeline execution, pass rollback on failure,
PassRegistry, and the fluent API with reporting.
"""

import json
import os
import pytest
import tensorflow.compat.v1 as tf
from graph_optimizer.core.tensorflow import BasePass
from graph_optimizer.core import PassRegistry
from graph_optimizer.runner import OptimizationPipeline
from graph_optimizer.utils.graph_utils import create_node, create_const_node
from graph_optimizer.utils.reporting import OptimizationReport

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# Mock passes (registered once at module level)
# ---------------------------------------------------------------------------


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


def _register_mock_passes():
    if "mock_fail" not in PassRegistry._registered_passes:
        PassRegistry.register("mock_fail", opt_level=1, priority=10)(MockFailingPass)
    if "mock_success" not in PassRegistry._registered_passes:
        PassRegistry.register("mock_success", opt_level=1, priority=20)(MockSuccessPass)


_register_mock_passes()


# ---------------------------------------------------------------------------
# Rollback on pass failure
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_pb(tmp_path):
    """Write a minimal GraphDef .pb, yield (input_path, output_path)."""
    inp = str(tmp_path / "input.pb")
    out = str(tmp_path / "output.pb")
    gd = tf.GraphDef()
    n = gd.node.add()
    n.name, n.op = "Input", "Placeholder"
    with open(inp, "wb") as f:
        f.write(gd.SerializeToString())
    return inp, out


def test_pipeline_rollback_on_failure(tmp_pb):
    """A failing pass must not leave its changes in the output graph."""
    inp, out = tmp_pb
    OptimizationPipeline(
        input_graph=inp, output_graph=out, passes=["mock_fail", "mock_success"]
    ).run()
    result = tf.GraphDef()
    with open(out, "rb") as f:
        result.ParseFromString(f.read())
    names = [n.name for n in result.node]
    assert "GOOD_NODE" in names
    assert "BAD_NODE" not in names


# ---------------------------------------------------------------------------
# Fluent API and reporting
# ---------------------------------------------------------------------------


def test_fluent_api_and_report():
    """OptimizationPipeline fluent API produces a valid report."""
    gd = tf.GraphDef()
    x = create_node("Placeholder", "x")
    zero = create_const_node("zero", 0.0, dtype="float32")
    add = create_node("Add", "add", inputs=["x", "zero"])
    gd.node.extend([x, zero, add])

    report = (
        OptimizationPipeline(graph_def=gd)
        .with_level(1)
        .add_pass("algebraic_simplify")
        .with_cleanup(True)
        .run()
    )

    assert report.initial_nodes == 3
    assert report.final_nodes < 3
    assert any("AlgebraicSimplify" in name for name in report.pass_stats)


def test_report_json_export(tmp_path):
    """Report.save_json() creates a readable JSON file."""
    gd = tf.GraphDef()
    x = create_node("Placeholder", "x")
    zero = create_const_node("zero", 0.0, dtype="float32")
    add = create_node("Add", "add", inputs=["x", "zero"])
    gd.node.extend([x, zero, add])

    json_path = str(tmp_path / "report.json")
    report = (
        OptimizationPipeline(graph_def=gd)
        .with_level(1)
        .add_pass("algebraic_simplify")
        .run()
    )
    report.save_json(json_path)
    assert os.path.exists(json_path)


# ---------------------------------------------------------------------------
# PassRegistry
# ---------------------------------------------------------------------------


def test_pass_registry_get_pass_returns_instance():
    import graph_optimizer.transforms.tensorflow.scalar  # noqa: F401

    tf_names = [
        n for n in PassRegistry._registered_passes if not n.startswith("torch_")
    ]
    assert len(tf_names) > 0
    assert PassRegistry.get_pass(tf_names[0]) is not None


def test_pass_registry_unknown_raises():
    with pytest.raises(ValueError, match="Unknown pass"):
        PassRegistry.get_pass("does_not_exist_xyz")


def test_pass_registry_sort_by_priority():
    import graph_optimizer.transforms.tensorflow.scalar  # noqa: F401

    tf_names = [
        n for n in PassRegistry._registered_passes if not n.startswith("torch_")
    ]
    assert PassRegistry.sort_passes(tf_names) == sorted(
        tf_names, key=PassRegistry.get_priority
    )


def test_pass_registry_get_passes_by_level():
    import graph_optimizer.transforms.tensorflow.scalar  # noqa: F401

    level1 = PassRegistry.get_passes_by_level(1)
    level2 = PassRegistry.get_passes_by_level(2)
    assert set(level1).issubset(set(level2)) or len(level1) <= len(level2)


# ---------------------------------------------------------------------------
# OptimizationReport
# ---------------------------------------------------------------------------


def test_report_to_dict_structure():
    """to_dict() must contain summary + passes keys with correct fields."""
    gd = tf.GraphDef()
    x = create_node("Placeholder", "x")
    zero = create_const_node("zero", 0.0, dtype="float32")
    add = create_node("Add", "add", inputs=["x", "zero"])
    gd.node.extend([x, zero, add])

    report = (
        OptimizationPipeline(graph_def=gd, level=1).add_pass("algebraic_simplify").run()
    )
    d = report.to_dict()
    assert "summary" in d
    assert "passes" in d
    assert d["summary"]["initial_nodes"] == 3
    assert d["summary"]["initial_nodes"] >= d["summary"]["final_nodes"]
    assert 0.0 <= d["summary"]["reduction_pct"] <= 100.0


def test_report_reduction_pct_zero_initial():
    """Edge case: 0 initial nodes must not cause ZeroDivisionError."""
    r = OptimizationReport(
        initial_nodes=0, final_nodes=0, total_time=0.0, pass_stats={}
    )
    assert r.reduction_pct == 0.0


def test_report_nodes_removed_computed():
    r = OptimizationReport(
        initial_nodes=10, final_nodes=7, total_time=0.1, pass_stats={}
    )
    assert r.nodes_removed == 3
    assert abs(r.reduction_pct - 30.0) < 0.01


def test_report_save_json_roundtrip(tmp_path):
    """save_json() then reading back must contain correct summary fields."""
    r = OptimizationReport(
        initial_nodes=20, final_nodes=15, total_time=0.5, pass_stats={}
    )
    p = tmp_path / "r.json"
    r.save_json(str(p))
    d = json.loads(p.read_text())
    assert d["summary"]["initial_nodes"] == 20
    assert d["summary"]["nodes_removed"] == 5
    assert abs(d["summary"]["reduction_pct"] - 25.0) < 0.01
