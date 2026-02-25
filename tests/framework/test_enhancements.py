import tensorflow.compat.v1 as tf
from graph_optimizer.runner import OptimizationPipeline
from graph_optimizer.utils.graph_utils import create_node, create_const_node


def test_fluent_api_and_reporting():
    # Build a simple graph
    graph_def = tf.GraphDef()
    x = create_node("Placeholder", "x")
    zero = create_const_node("zero", 0.0, dtype="float32")
    add = create_node("Add", "add", inputs=["x", "zero"])
    graph_def.node.extend([x, zero, add])

    # Use Fluent API
    pipeline = (
        OptimizationPipeline(graph_def=graph_def)
        .with_level(1)
        .add_pass("algebraic_simplify")
        .with_cleanup(True)
    )

    # Run and get report
    report = pipeline.run()

    # Verify report structure
    assert report.initial_nodes == 3
    assert report.final_nodes < 3
    assert any("AlgebraicSimplify" in name for name in report.pass_stats)

    # Test JSON export
    report.save_json("test_report.json")
    import os

    assert os.path.exists("test_report.json")
    os.remove("test_report.json")

    # Test printing (smoke test)
    report.print_summary()


if __name__ == "__main__":
    test_fluent_api_and_reporting()
    print("Fluent API and Reporting test PASSED")
