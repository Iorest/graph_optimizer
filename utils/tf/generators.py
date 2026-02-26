import tensorflow.compat.v1 as tf
from .graph_utils import create_node
from tensorflow.core.framework import attr_value_pb2, tensor_shape_pb2

tf.disable_v2_behavior()


def create_complex_concat_graph(
    output_path="demos/complex_concat_graph.pbtxt",
):
    """Generates an ultra-complex graph with multi-branch cascading concats for testing."""
    graph_def = tf.GraphDef()

    def make_attr_dtype(dtype):
        return attr_value_pb2.AttrValue(type=dtype)

    def make_attr_shape(dims):
        shape_proto = tensor_shape_pb2.TensorShapeProto()
        for d in dims:
            shape_proto.dim.add(size=d)
        return attr_value_pb2.AttrValue(shape=shape_proto)

    # Inputs
    p1 = create_node(
        "Placeholder",
        "p1",
        attr={
            "dtype": make_attr_dtype(tf.float32.as_datatype_enum),
            "shape": make_attr_shape([1, 10]),
        },
    )
    p2 = create_node(
        "Placeholder",
        "p2",
        attr={
            "dtype": make_attr_dtype(tf.float32.as_datatype_enum),
            "shape": make_attr_shape([1, 10]),
        },
    )
    p3 = create_node(
        "Placeholder",
        "p3",
        attr={
            "dtype": make_attr_dtype(tf.float32.as_datatype_enum),
            "shape": make_attr_shape([1, 10]),
        },
    )
    p4 = create_node(
        "Placeholder",
        "p4",
        attr={
            "dtype": make_attr_dtype(tf.float32.as_datatype_enum),
            "shape": make_attr_shape([1, 10]),
        },
    )
    graph_def.node.extend([p1, p2, p3, p4])

    # Branch 1 (Fusable)
    a1 = create_node(
        "Add",
        "a1",
        inputs=["p1", "p2"],
        attr={"T": make_attr_dtype(tf.float32.as_datatype_enum)},
    )
    c1_axis = create_node(
        "Const",
        "c1/axis",
        attr={
            "dtype": make_attr_dtype(tf.int32.as_datatype_enum),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(1, dtype=tf.int32)
            ),
        },
    )
    c1 = create_node(
        "ConcatV2",
        "c1",
        inputs=["p1", "a1", "c1/axis"],
        attr={
            "T": make_attr_dtype(tf.float32.as_datatype_enum),
            "Tidx": make_attr_dtype(tf.int32.as_datatype_enum),
            "N": attr_value_pb2.AttrValue(i=2),
        },
    )
    graph_def.node.extend([a1, c1_axis, c1])

    # Branch 2 (Fusable)
    a2 = create_node(
        "Add",
        "a2",
        inputs=["p3", "p4"],
        attr={"T": make_attr_dtype(tf.float32.as_datatype_enum)},
    )
    c2_axis = create_node(
        "Const",
        "c2/axis",
        attr={
            "dtype": make_attr_dtype(tf.int32.as_datatype_enum),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(1, dtype=tf.int32)
            ),
        },
    )
    c2 = create_node(
        "ConcatV2",
        "c2",
        inputs=["p3", "a2", "c2/axis"],
        attr={
            "T": make_attr_dtype(tf.float32.as_datatype_enum),
            "Tidx": make_attr_dtype(tf.int32.as_datatype_enum),
            "N": attr_value_pb2.AttrValue(i=2),
        },
    )
    graph_def.node.extend([a2, c2_axis, c2])

    # Branch 3 (Cascade of fusions)
    c3_axis = create_node(
        "Const",
        "c3/axis",
        attr={
            "dtype": make_attr_dtype(tf.int32.as_datatype_enum),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(1, dtype=tf.int32)
            ),
        },
    )
    c3 = create_node(
        "ConcatV2",
        "c3",
        inputs=["c1", "p4", "c3/axis"],
        attr={
            "T": make_attr_dtype(tf.float32.as_datatype_enum),
            "Tidx": make_attr_dtype(tf.int32.as_datatype_enum),
            "N": attr_value_pb2.AttrValue(i=2),
        },
    )
    graph_def.node.extend([c3_axis, c3])

    # Wide Merge (Fuses c1, c2, c3 results)
    cw_axis = create_node(
        "Const",
        "cw/axis",
        attr={
            "dtype": make_attr_dtype(tf.int32.as_datatype_enum),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(1, dtype=tf.int32)
            ),
        },
    )
    c_wide = create_node(
        "ConcatV2",
        "c_wide",
        inputs=["c1", "c2", "c3", "cw/axis"],
        attr={
            "T": make_attr_dtype(tf.float32.as_datatype_enum),
            "Tidx": make_attr_dtype(tf.int32.as_datatype_enum),
            "N": attr_value_pb2.AttrValue(i=3),
        },
    )
    graph_def.node.extend([cw_axis, c_wide])

    # Relu and Bias (Simple Add)
    bias = create_node(
        "Const",
        "bias",
        attr={
            "dtype": make_attr_dtype(tf.float32.as_datatype_enum),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(0.1, dtype=tf.float32)
            ),
        },
    )
    ba1 = create_node(
        "Add",
        "ba1",
        inputs=["c_wide", "bias"],
        attr={"T": make_attr_dtype(tf.float32.as_datatype_enum)},
    )
    r1 = create_node(
        "Relu",
        "r1",
        inputs=["ba1"],
        attr={"T": make_attr_dtype(tf.float32.as_datatype_enum)},
    )
    graph_def.node.extend([bias, ba1, r1])

    # Non-fusable branch (Axis mismatch)
    c4_axis = create_node(
        "Const",
        "c4/axis",
        attr={
            "dtype": make_attr_dtype(tf.int32.as_datatype_enum),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(0, dtype=tf.int32)
            ),
        },
    )
    c4 = create_node(
        "ConcatV2",
        "c4",
        inputs=["r1", "r1", "c4/axis"],
        attr={
            "T": make_attr_dtype(tf.float32.as_datatype_enum),
            "Tidx": make_attr_dtype(tf.int32.as_datatype_enum),
            "N": attr_value_pb2.AttrValue(i=2),
        },
    )
    graph_def.node.extend([c4_axis, c4])

    # Control dependency
    ctrl = create_node("NoOp", "ctrl_trigger")
    output = create_node(
        "Identity",
        "predicts",
        inputs=["c4", "^ctrl_trigger"],
        attr={"T": make_attr_dtype(tf.float32.as_datatype_enum)},
    )
    graph_def.node.extend([ctrl, output])

    # Save to file
    from .graph_utils import save_graph

    save_graph(graph_def, output_path)
