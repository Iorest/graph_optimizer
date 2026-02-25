import tensorflow.compat.v1 as tf
import numpy as np
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.transforms.scalar.algebraic_simplify import AlgebraicSimplifyPass
from graph_optimizer.transforms.scalar.cse import CSEPass
from graph_optimizer.transforms.vectorize.pack_vectorize import PackVectorizePass
from graph_optimizer.utils.graph_utils import create_node, create_const_node
from graph_optimizer.utils.logger import set_log_level, DEBUG


def set_shape(node, shape):
    """Sets _output_shapes attribute for a node."""
    shape_proto = tf.TensorShape(shape).as_proto()
    node.attr["_output_shapes"].list.shape.add().CopyFrom(shape_proto)


def test_algebraic_simplify_broadcasting():
    """Verify that AlgebraicSimplify respects broadcasting during identity elimination."""
    # Case 1: Add(x, zero) where x: [4], zero: [3, 4]
    graph_def = tf.GraphDef()
    x = create_node("Placeholder", "x")
    x.attr["shape"].shape.CopyFrom(tf.TensorShape([4]).as_proto())
    set_shape(x, [4])

    zero = create_const_node("zero", value=0.0, dtype="float32", shape=[3, 4])
    set_shape(zero, [3, 4])
    add = create_node("Add", "add", inputs=["x", "zero"])
    set_shape(add, [3, 4])
    graph_def.node.extend([x, zero, add])

    optimizer = GraphOptimizer(graph_def)
    simplify_pass = AlgebraicSimplifyPass()

    optimized = simplify_pass.transform(optimizer, protected_nodes=["add"])
    node_names = {n.name for n in optimized.node}
    assert "add" in node_names

    # Case 2: Add(x, zero) where x: [3, 4], zero: [4]
    graph_def2 = tf.GraphDef()
    x2 = create_node("Placeholder", "x")
    x2.attr["shape"].shape.CopyFrom(tf.TensorShape([3, 4]).as_proto())
    set_shape(x2, [3, 4])
    zero2 = create_const_node("zero", value=0.0, dtype="float32", shape=[4])
    set_shape(zero2, [4])
    add2 = create_node("Add", "add", inputs=["x", "zero"])
    set_shape(add2, [3, 4])
    graph_def2.node.extend([x2, zero2, add2])

    optimizer2 = GraphOptimizer(graph_def2)
    simplify_pass = AlgebraicSimplifyPass()
    optimized2 = simplify_pass.transform(optimizer2)
    node_names2 = {n.name for n in optimized2.node}
    assert "add" not in node_names2


def test_pack_vectorize_deep_hoisting():
    """Verify that PackVectorize can hoist through multiple layers (e.g., Relu -> Add)."""
    set_log_level(DEBUG)
    graph_def = tf.GraphDef()
    x1 = create_node("Placeholder", "x1")
    set_shape(x1, [4])
    x2 = create_node("Placeholder", "x2")
    set_shape(x2, [4])
    b = create_const_node("bias", value=1.0, dtype="float32", shape=[4])
    set_shape(b, [4])

    add1 = create_node("Add", "add1", inputs=["x1", "bias"])
    set_shape(add1, [4])
    add2 = create_node("Add", "add2", inputs=["x2", "bias"])
    set_shape(add2, [4])
    relu1 = create_node("Relu", "relu1", inputs=["add1"])
    set_shape(relu1, [4])
    relu2 = create_node("Relu", "relu2", inputs=["add2"])
    set_shape(relu2, [4])
    pack = create_node("Pack", "pack", inputs=["relu1", "relu2"])
    set_shape(pack, [2, 4])
    pack.attr["axis"].i = 0
    pack.attr["N"].i = 2
    pack.attr["T"].type = tf.float32.as_datatype_enum

    graph_def.node.extend([x1, x2, b, add1, add2, relu1, relu2, pack])

    optimizer = GraphOptimizer(graph_def)
    pack_pass = PackVectorizePass()

    # We don't protect 'pack' if we want it to be fully remapped/rewritten
    optimized = pack_pass.transform(optimizer)

    # Check for batched structure
    ops = [n.op for n in optimized.node]
    assert "Relu" in ops
    assert "Add" in ops

    # In PackVectorize, the original 'pack' node is replaced by the new top node
    # if it was remapped.
    relu_nodes = [n for n in optimized.node if n.op == "Relu"]
    # Should have one batched Relu
    assert len(relu_nodes) == 1
    batched_relu = relu_nodes[0]

    add_nodes = [n for n in optimized.node if n.op == "Add"]
    assert len(add_nodes) == 1
    batched_add = add_nodes[0]
    assert batched_relu.input[0] == batched_add.name


def test_pack_vectorize_attr_mismatch():
    """Verify that PackVectorize correctly detects attribute mismatches and refuses to hoist."""
    graph_def = tf.GraphDef()
    x1 = create_node("Placeholder", "x1")
    set_shape(x1, [4, 4])
    x2 = create_node("Placeholder", "x2")
    set_shape(x2, [4, 4])
    w = create_node("Placeholder", "w")
    set_shape(w, [4, 4])

    mm1 = create_node("MatMul", "mm1", inputs=["x1", "w"])
    mm1.attr["transpose_a"].b = True
    mm1.attr["transpose_b"].b = False
    mm1.attr["T"].type = tf.float32.as_datatype_enum
    set_shape(mm1, [4, 4])

    mm2 = create_node("MatMul", "mm2", inputs=["x2", "w"])
    mm2.attr["transpose_a"].b = False  # MISMATCH
    mm2.attr["transpose_b"].b = False
    mm2.attr["T"].type = tf.float32.as_datatype_enum
    set_shape(mm2, [4, 4])

    pack = create_node("Pack", "pack", inputs=["mm1", "mm2"])
    pack.attr["axis"].i = 0
    pack.attr["N"].i = 2
    pack.attr["T"].type = tf.float32.as_datatype_enum
    set_shape(pack, [2, 4, 4])

    graph_def.node.extend([x1, x2, w, mm1, mm2, pack])

    optimizer = GraphOptimizer(graph_def)
    pack_pass = PackVectorizePass()
    optimized = pack_pass.transform(optimizer)

    mm_ops = [n for n in optimized.node if n.op == "MatMul"]
    assert len(mm_ops) == 2


def test_cse_attribute_safety():
    """Verify that CSE does not merge nodes with different attributes."""
    graph_def = tf.GraphDef()
    x = create_node("Placeholder", "x")
    w = create_node("Placeholder", "w")

    mm1 = create_node("MatMul", "mm1", inputs=["x", "w"])
    mm1.attr["transpose_a"].b = True
    mm2 = create_node("MatMul", "mm2", inputs=["x", "w"])
    mm2.attr["transpose_a"].b = False

    id1 = create_node("Identity", "id1", inputs=["mm1"])
    id2 = create_node("Identity", "id2", inputs=["mm2"])

    graph_def.node.extend([x, w, mm1, mm2, id1, id2])

    optimizer = GraphOptimizer(graph_def)
    cse_pass = CSEPass()
    optimized = cse_pass.transform(optimizer)
    mm_ops = [n for n in optimized.node if n.op == "MatMul"]
    assert len(mm_ops) == 2


def test_inter_pass_convergence():
    """Verify stability and convergence when multiple passes interact."""
    graph_def = tf.GraphDef()
    x = create_node("Placeholder", "x")
    zero1 = create_const_node("zero1", 0.0, "float32", [])
    zero2 = create_const_node("zero2", 0.0, "float32", [])

    add1 = create_node("Add", "add1", inputs=["x", "zero1"])
    add2 = create_node("Add", "add2", inputs=["x", "zero2"])
    final = create_node("Add", "final", inputs=["add1", "add2"])

    graph_def.node.extend([x, zero1, zero2, add1, add2, final])

    optimizer = GraphOptimizer(graph_def)
    cse_pass = CSEPass()
    simplify_pass = AlgebraicSimplifyPass()

    cse_pass.transform(optimizer)
    simplify_pass.transform(optimizer)

    node_names = {n.name for n in optimizer.graph_def.node}
    assert "add1" not in node_names
    assert "add2" not in node_names
    final_node = next(n for n in optimizer.graph_def.node if n.name == "final")
    assert final_node.input == ["x", "x"]
