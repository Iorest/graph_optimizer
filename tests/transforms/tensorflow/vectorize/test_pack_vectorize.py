"""Tests for PackVectorizePass (TensorFlow)."""

import pytest
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import tensor_util
from graph_optimizer.core.tensorflow import TFGraphOptimizer
from graph_optimizer.utils.tf.graph_utils import create_node, make_output_shapes_attr
from graph_optimizer.transforms.tensorflow.vectorize import PackVectorizePass
from tensorflow.core.framework import attr_value_pb2

tf.disable_v2_behavior()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ph(name, shape):
    n = create_node("Placeholder", name)
    n.attr["shape"].shape.CopyFrom(tf.TensorShape(shape).as_proto())
    return n


def _const(name, value, dtype=tf.float32, shape=None):
    return create_node(
        "Const",
        name,
        attr={
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(value, dtype=dtype, shape=shape)
            ),
            "dtype": attr_value_pb2.AttrValue(type=dtype.as_datatype_enum),
        },
    )


def _relu(name, inp, out_shape):
    n = create_node("Relu", name, inputs=[inp])
    n.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([out_shape]))
    return n


def _pack(inputs, axis, out_shape):
    n = create_node("Pack", "pack", inputs=inputs)
    n.attr["axis"].i = axis
    n.attr["N"].i = len(inputs)
    n.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([out_shape]))
    return n


def _run(*nodes):
    gd = tf.GraphDef()
    gd.node.extend(nodes)
    opt = TFGraphOptimizer(gd)
    result = PackVectorizePass().transform(opt)
    return {n.name: n for n in result.node}, result


def _batched_relu(node_map, result):
    return next(
        (n for n in result.node if n.op == "Relu" and "pack" in n.name.lower()), None
    )


# ---------------------------------------------------------------------------
# Axis tests (pack axis is preserved through hoisting)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis,out_shape",
    [
        (0, [2, 4]),
        (-1, [4, 2]),
        (1, [10, 2, 20]),
    ],
)
def test_relu_hoist_preserves_axis(axis, out_shape):
    in_shape = [4] if axis != 1 else [10, 20]
    x1 = _ph("x1", in_shape)
    x2 = _ph("x2", in_shape)
    r1 = _relu("r1", "x1", in_shape)
    r2 = _relu("r2", "x2", in_shape)
    pk = _pack(["r1", "r2"], axis, out_shape)
    node_map, result = _run(x1, x2, r1, r2, pk)

    batched = _batched_relu(node_map, result)
    assert batched is not None, "Expected a batched Relu node"
    p_name = batched.input[0]
    assert node_map[p_name].op == "Pack"
    assert node_map[p_name].attr["axis"].i == axis


# ---------------------------------------------------------------------------
# Dimension-op axis adjustment
# ---------------------------------------------------------------------------


def test_squeeze_axis_adjusted():
    """Squeeze dim 0 hoisted through Pack(axis=0) → squeeze_dims becomes [1]."""
    x1 = _ph("x1", [1, 10])
    x2 = _ph("x2", [1, 10])
    s1 = create_node("Squeeze", "s1", inputs=["x1"])
    s1.attr["squeeze_dims"].list.i.append(0)
    s1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[10]]))
    s2 = create_node("Squeeze", "s2", inputs=["x2"])
    s2.attr["squeeze_dims"].list.i.append(0)
    s2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[10]]))
    pk = _pack(["s1", "s2"], 0, [2, 10])
    node_map, result = _run(x1, x2, s1, s2, pk)
    sq = next(
        (n for n in result.node if n.op == "Squeeze" and "pack" in n.name.lower()), None
    )
    assert sq is not None
    assert list(sq.attr["squeeze_dims"].list.i) == [1]


def test_expand_dims_axis_adjusted():
    """ExpandDims(x, 0) hoisted through Pack(axis=0) → new expand axis = 1."""
    x1 = _ph("x1", [10])
    x2 = _ph("x2", [10])
    ax = _const("expand_axis", 0, dtype=tf.int32)
    e1 = create_node("ExpandDims", "e1", inputs=["x1", "expand_axis"])
    e1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 10]]))
    e2 = create_node("ExpandDims", "e2", inputs=["x2", "expand_axis"])
    e2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 10]]))
    pk = _pack(["e1", "e2"], 0, [2, 1, 10])
    node_map, result = _run(x1, x2, ax, e1, e2, pk)
    exp = next(
        (n for n in result.node if n.op == "ExpandDims" and "pack" in n.name.lower()),
        None,
    )
    assert exp is not None
    ax_name = exp.input[1]
    val = tensor_util.MakeNdarray(node_map[ax_name].attr["value"].tensor)
    assert val == 1


def test_transpose_perm_adjusted():
    """Transpose(x, [1,0]) hoisted through Pack(axis=0) → perm becomes [0,2,1]."""
    x1 = _ph("x1", [10, 20])
    x2 = _ph("x2", [10, 20])
    perm = _const("transpose_perm", [1, 0], dtype=tf.int32)
    t1 = create_node("Transpose", "t1", inputs=["x1", "transpose_perm"])
    t1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[20, 10]]))
    t2 = create_node("Transpose", "t2", inputs=["x2", "transpose_perm"])
    t2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[20, 10]]))
    pk = _pack(["t1", "t2"], 0, [2, 20, 10])
    node_map, result = _run(x1, x2, perm, t1, t2, pk)
    tr = next(
        (n for n in result.node if n.op == "Transpose" and "pack" in n.name.lower()),
        None,
    )
    assert tr is not None
    pn = node_map[tr.input[1]]
    val = tensor_util.MakeNdarray(pn.attr["value"].tensor)
    assert list(val) == [0, 2, 1]


# ---------------------------------------------------------------------------
# MatMul → BatchMatMulV2
# ---------------------------------------------------------------------------


def test_matmul_shared_weights():
    """MatMul(xi, W) × 2 → BatchMatMulV2 with shared W."""
    x1 = _ph("x1", [1, 10])
    x2 = _ph("x2", [1, 10])
    w = _const(
        "W", tf.initializers.glorot_uniform()([10, 20]).eval(session=tf.Session())
    )
    m1 = create_node("MatMul", "m1", inputs=["x1", "W"])
    m1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 20]]))
    m2 = create_node("MatMul", "m2", inputs=["x2", "W"])
    m2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 20]]))
    pk = _pack(["m1", "m2"], 0, [2, 1, 20])
    node_map, result = _run(x1, x2, w, m1, m2, pk)
    bmm = next(
        (
            n
            for n in result.node
            if n.op == "BatchMatMulV2" and "pack" in n.name.lower()
        ),
        None,
    )
    assert bmm is not None
    assert bmm.input[1] == "W"


def test_matmul_different_weights():
    """MatMul(x1,W1) + MatMul(x2,W2) → BatchMatMulV2 with packed weights."""
    x1 = _ph("x1", [1, 10])
    x2 = _ph("x2", [1, 10])
    sess = tf.Session()
    w1 = _const("W1", tf.initializers.glorot_uniform()([10, 20]).eval(session=sess))
    w2 = _const("W2", tf.initializers.glorot_uniform()([10, 20]).eval(session=sess))
    m1 = create_node("MatMul", "m1", inputs=["x1", "W1"])
    m1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 20]]))
    m2 = create_node("MatMul", "m2", inputs=["x2", "W2"])
    m2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 20]]))
    pk = _pack(["m1", "m2"], 0, [2, 1, 20])
    node_map, result = _run(x1, x2, w1, w2, m1, m2, pk)
    bmm = next(
        (
            n
            for n in result.node
            if n.op == "BatchMatMulV2" and "pack" in n.name.lower()
        ),
        None,
    )
    assert bmm is not None
    w_pack = node_map[bmm.input[1]]
    assert w_pack.op == "Pack"
    assert len(w_pack.input) == 2
