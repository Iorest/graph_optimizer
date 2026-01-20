import tensorflow.compat.v1 as tf
import numpy as np
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.transforms.scalar.algebraic_simplify import AlgebraicSimplifyPass
from graph_optimizer.utils.graph_utils import create_node, create_const_node


def test_shape_breakage():
    # Construct a graph:
    # x = Placeholder(shape=[2, 2], dtype=float32)
    # eq = Equal(x, x) -> in TF this is [True, True; True, True]
    # sink = Add(eq, some_other_bool_tensor)

    x = tf.NodeDef(name="x", op="Placeholder")
    x.attr["dtype"].type = tf.float32.as_datatype_enum
    # Mocking shape attr
    shape_proto = tf.TensorShape([2, 128]).as_proto()
    x.attr["shape"].shape.CopyFrom(shape_proto)

    # Equal(x, x)
    eq = create_node("Equal", "eq_node", inputs=["x", "x"])

    # Sink to protect eq
    sink = create_node("Identity", "sink", inputs=["eq_node"])

    graph_def = tf.GraphDef()
    graph_def.node.extend([x, eq, sink])

    optimizer = GraphOptimizer(graph_def)
    pass_instance = AlgebraicSimplifyPass()

    print("Before optimization:")
    print(f"eq_node: {optimizer.nodes['eq_node'].op}")

    # Run optimization
    pass_instance.transform(optimizer, protected_nodes=["sink"])

    print("\nAfter optimization:")
    if "eq_node_bool" in optimizer.nodes:
        new_node = optimizer.nodes["eq_node_bool"]
        print(f"Replacement node op: {new_node.op}")

        # Check shape in top-level or in tensor
        shape = []
        if "shape" in new_node.attr:
            shape = [d.size for d in new_node.attr["shape"].shape.dim]
        elif "value" in new_node.attr:
            ts = new_node.attr["value"].tensor.tensor_shape
            shape = [d.size for d in ts.dim]

        print(f"Replacement node shape: {shape}")

        if shape != [2, 128]:
            print(f"\nBUG DETECTED: Shape mismatch! Expected [2, 128], got {shape}")
        else:
            print("\nSUCCESS: Shape preserved.")
    else:
        print("Replacement not found or not as expected.")


def test_div_zero():
    # c1 = 5, c0 = 0, div = c1 / c0
    c1 = create_const_node("c1", 5.0, dtype="float32", shape=[])
    c0 = create_const_node("c0", 0.0, dtype="float32", shape=[])
    div = create_node("Div", "div_node", inputs=["c1", "c0"])

    graph_def = tf.GraphDef()
    graph_def.node.extend([c1, c0, div])

    optimizer = GraphOptimizer(graph_def)
    from graph_optimizer.transforms.scalar.constant_fold import ConstantFoldPass

    pass_instance = ConstantFoldPass()

    print("\n--- Division by Zero Test ---")
    pass_instance.transform(optimizer)

    if "div_node" in optimizer.nodes and optimizer.nodes["div_node"].op == "Div":
        print("SUCCESS: Division by zero was NOT folded (remained as Div).")
    else:
        print("FAILURE: Division by zero was folded or removed!")


def test_broadcasting_safety():
    # x = Placeholder([2, 128])
    # zero = Const(0, shape=[1]) -> needs broadcasting
    # add = Add(x, zero)
    x = tf.NodeDef(name="x", op="Placeholder")
    x.attr["dtype"].type = tf.float32.as_datatype_enum
    x.attr["shape"].shape.CopyFrom(tf.TensorShape([2, 128]).as_proto())

    zero = create_const_node("zero_tensor", 0.0, dtype="float32", shape=[1])
    add = create_node("Add", "add_node", inputs=["x", "zero_tensor"])
    sink = create_node("Identity", "sink", inputs=["add_node"])

    graph_def = tf.GraphDef()
    graph_def.node.extend([x, zero, add, sink])

    optimizer = GraphOptimizer(graph_def)
    pass_instance = AlgebraicSimplifyPass()

    print("\n--- Broadcasting Safety Test ---")
    pass_instance.transform(optimizer, protected_nodes=["sink"])

    if "add_node" in optimizer.nodes and optimizer.nodes["add_node"].op == "Add":
        print(
            "SUCCESS: Add(x, zero_tensor) was NOT simplified (broadcasting preserved)."
        )
    else:
        print("FAILURE: Add(x, zero_tensor) was simplified (potential shape breakage)!")


if __name__ == "__main__":
    print("Running Edge Case Verifications...")
    test_shape_breakage()
    test_div_zero()
    test_broadcasting_safety()
