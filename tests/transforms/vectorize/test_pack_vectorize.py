"""
Pack Vectorize Axis Tests - Pack 上浮 axis 测试
==============================================

测试内容：
1. Axis 0 (默认情况)
2. Negative Axis (-1)
3. Middle Axis (axis=1 on 3D)
4. Axis Adjustment (Squeeze, ExpandDims)
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.utils import create_node, make_output_shapes_attr, create_const_node
from graph_optimizer.transforms.vectorize import PackVectorizePass
from tensorflow.core.framework import attr_value_pb2

tf.disable_v2_behavior()

class TestPackVectorizeAxis(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def _create_const(self, name, value, dtype=tf.float32, shape=None):
        attr = {
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(value, dtype=dtype, shape=shape)
            ),
            "dtype": attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
        }
        return create_node("Const", name, attr=attr)

    def test_axis_zero(self):
        """Test hoisting Pack through Relu with axis=0."""
        graph_def = tf.GraphDef()
        x1 = create_node("Placeholder", "x1")
        x1.attr["shape"].shape.CopyFrom(tf.TensorShape([4]).as_proto())
        x2 = create_node("Placeholder", "x2")
        x2.attr["shape"].shape.CopyFrom(tf.TensorShape([4]).as_proto())
        
        r1 = create_node("Relu", "r1", inputs=["x1"])
        r1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[4]]))
        r2 = create_node("Relu", "r2", inputs=["x2"])
        r2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[4]]))
        
        pack = create_node("Pack", "pack", inputs=["r1", "r2"])
        pack.attr["axis"].i = 0
        pack.attr["N"].i = 2
        pack.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[2, 4]]))
        
        graph_def.node.extend([x1, x2, r1, r2, pack])
        
        optimizer = GraphOptimizer(graph_def)
        optimized = PackVectorizePass().transform(optimizer)
        
        node_map = {n.name: n for n in optimized.node}
        relu_node = next((n for n in optimized.node if n.op == "Relu" and "pack" in n.name.lower()), None)
        self.assertIsNotNone(relu_node, "Should find batched Relu node")
        
        # Check that its input is a Pack node
        p_name = relu_node.input[0]
        self.assertIn(p_name, node_map)
        self.assertEqual(node_map[p_name].op, "Pack")
        self.assertEqual(node_map[p_name].attr["axis"].i, 0)

    def test_axis_last(self):
        """Test hoisting Pack through Relu with axis=-1."""
        graph_def = tf.GraphDef()
        x1 = create_node("Placeholder", "x1")
        x1.attr["shape"].shape.CopyFrom(tf.TensorShape([4]).as_proto())
        x2 = create_node("Placeholder", "x2")
        x2.attr["shape"].shape.CopyFrom(tf.TensorShape([4]).as_proto())
        
        r1 = create_node("Relu", "r1", inputs=["x1"])
        r1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[4]]))
        r2 = create_node("Relu", "r2", inputs=["x2"])
        r2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[4]]))
        
        pack = create_node("Pack", "pack", inputs=["r1", "r2"])
        pack.attr["axis"].i = -1
        pack.attr["N"].i = 2
        pack.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[4, 2]]))
        
        graph_def.node.extend([x1, x2, r1, r2, pack])
        
        optimizer = GraphOptimizer(graph_def)
        optimized = PackVectorizePass().transform(optimizer)
        
        node_map = {n.name: n for n in optimized.node}
        relu_node = next((n for n in optimized.node if n.op == "Relu" and "pack" in n.name.lower()), None)
        self.assertIsNotNone(relu_node)
        p_name = relu_node.input[0]
        self.assertEqual(node_map[p_name].attr["axis"].i, -1)

    def test_axis_middle(self):
        """Test hoisting Pack through Relu with axis=1 on 3D."""
        graph_def = tf.GraphDef()
        shape = [10, 20]
        x1 = create_node("Placeholder", "x1")
        x1.attr["shape"].shape.CopyFrom(tf.TensorShape(shape).as_proto())
        x2 = create_node("Placeholder", "x2")
        x2.attr["shape"].shape.CopyFrom(tf.TensorShape(shape).as_proto())
        
        r1 = create_node("Relu", "r1", inputs=["x1"])
        r1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[10, 20]]))
        r2 = create_node("Relu", "r2", inputs=["x2"])
        r2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[10, 20]]))
        
        pack = create_node("Pack", "pack", inputs=["r1", "r2"])
        pack.attr["axis"].i = 1
        pack.attr["N"].i = 2
        pack.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[10, 2, 20]]))
        
        graph_def.node.extend([x1, x2, r1, r2, pack])
        
        optimizer = GraphOptimizer(graph_def)
        optimized = PackVectorizePass().transform(optimizer)
        
        node_map = {n.name: n for n in optimized.node}
        relu_node = next((n for n in optimized.node if n.op == "Relu" and "pack" in n.name.lower()), None)
        self.assertIsNotNone(relu_node)
        p_name = relu_node.input[0]
        self.assertEqual(node_map[p_name].attr["axis"].i, 1)

    def test_axis_adjust_squeeze(self):
        """Test axis adjustment for Squeeze."""
        graph_def = tf.GraphDef()
        x1 = create_node("Placeholder", "x1")
        x1.attr["shape"].shape.CopyFrom(tf.TensorShape([1, 10]).as_proto())
        x2 = create_node("Placeholder", "x2")
        x2.attr["shape"].shape.CopyFrom(tf.TensorShape([1, 10]).as_proto())
        
        s1 = create_node("Squeeze", "s1", inputs=["x1"])
        s1.attr["squeeze_dims"].list.i.append(0)
        s1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[10]]))
        s2 = create_node("Squeeze", "s2", inputs=["x2"])
        s2.attr["squeeze_dims"].list.i.append(0)
        s2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[10]]))
        
        pack = create_node("Pack", "pack", inputs=["s1", "s2"])
        pack.attr["axis"].i = 0
        pack.attr["N"].i = 2
        pack.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[2, 10]]))
        
        graph_def.node.extend([x1, x2, s1, s2, pack])
        
        optimizer = GraphOptimizer(graph_def)
        optimized = PackVectorizePass().transform(optimizer)
        
        sq_node = next((n for n in optimized.node if n.op == "Squeeze" and "pack" in n.name.lower()), None)
        self.assertIsNotNone(sq_node)
        self.assertEqual(list(sq_node.attr["squeeze_dims"].list.i), [1])

    def test_axis_adjust_expand_dims(self):
        """Test axis adjustment for ExpandDims."""
        graph_def = tf.GraphDef()
        x1 = create_node("Placeholder", "x1")
        x1.attr["shape"].shape.CopyFrom(tf.TensorShape([10]).as_proto())
        x2 = create_node("Placeholder", "x2")
        x2.attr["shape"].shape.CopyFrom(tf.TensorShape([10]).as_proto())
        
        axis_const = self._create_const("expand_axis", 0, dtype=tf.int32)
        e1 = create_node("ExpandDims", "e1", inputs=["x1", "expand_axis"])
        e1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 10]]))
        e2 = create_node("ExpandDims", "e2", inputs=["x2", "expand_axis"])
        e2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 10]]))
        
        pack = create_node("Pack", "pack", inputs=["e1", "e2"])
        pack.attr["axis"].i = 0
        pack.attr["N"].i = 2
        pack.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[2, 1, 10]]))
        
        graph_def.node.extend([x1, x2, axis_const, e1, e2, pack])
        
        optimizer = GraphOptimizer(graph_def)
        optimized = PackVectorizePass().transform(optimizer)
        
        exp_node = next((n for n in optimized.node if n.op == "ExpandDims" and "pack" in n.name.lower()), None)
        self.assertIsNotNone(exp_node)
        axis_node = next(n for n in optimized.node if n.name == exp_node.input[1])
        from tensorflow.python.framework import tensor_util
        val = tensor_util.MakeNdarray(axis_node.attr["value"].tensor)
        self.assertEqual(val, 1)

    def test_axis_adjust_transpose(self):
        """Test axis adjustment for Transpose perm."""
        graph_def = tf.GraphDef()
        # x: [10, 20]
        x1 = create_node("Placeholder", "x1")
        x1.attr["shape"].shape.CopyFrom(tf.TensorShape([10, 20]).as_proto())
        x2 = create_node("Placeholder", "x2")
        x2.attr["shape"].shape.CopyFrom(tf.TensorShape([10, 20]).as_proto())
        
        # Transpose(x, perm=[1, 0]) -> [20, 10]
        perm_const = self._create_const("transpose_perm", [1, 0], dtype=tf.int32)
        t1 = create_node("Transpose", "t1", inputs=["x1", "transpose_perm"])
        t1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[20, 10]]))
        t2 = create_node("Transpose", "t2", inputs=["x2", "transpose_perm"])
        t2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[20, 10]]))
        
        # Pack([20, 10], [20, 10], axis=0) -> [2, 20, 10]
        pack = create_node("Pack", "pack", inputs=["t1", "t2"])
        pack.attr["axis"].i = 0
        pack.attr["N"].i = 2
        pack.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[2, 20, 10]]))
        
        graph_def.node.extend([x1, x2, perm_const, t1, t2, pack])
        
        optimizer = GraphOptimizer(graph_def)
        optimized = PackVectorizePass().transform(optimizer)
        
        tr_node = next((n for n in optimized.node if n.op == "Transpose" and "pack" in n.name.lower()), None)
        self.assertIsNotNone(tr_node)
        perm_node = next(n for n in optimized.node if n.name == tr_node.input[1])
        from tensorflow.python.framework import tensor_util
        val = tensor_util.MakeNdarray(perm_node.attr["value"].tensor)
        # Original perm [1, 0] becomes [0, 2, 1] if batch dim is at 0
        self.assertEqual(list(val), [0, 2, 1])

    def test_matmul_shared_weights(self):
        """Test hoisting MatMul with shared weights."""
        graph_def = tf.GraphDef()
        # x1, x2: [1, 10], W: [10, 20]
        x1 = create_node("Placeholder", "x1")
        x1.attr["shape"].shape.CopyFrom(tf.TensorShape([1, 10]).as_proto())
        x2 = create_node("Placeholder", "x2")
        x2.attr["shape"].shape.CopyFrom(tf.TensorShape([1, 10]).as_proto())
        
        w = self._create_const("W", tf.initializers.glorot_uniform()([10, 20]).eval(session=tf.Session()), dtype=tf.float32)
        
        m1 = create_node("MatMul", "m1", inputs=["x1", "W"])
        m1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 20]]))
        m2 = create_node("MatMul", "m2", inputs=["x2", "W"])
        m2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 20]]))
        
        # Pack([1, 20], [1, 20], axis=0) -> [2, 1, 20]
        pack = create_node("Pack", "pack", inputs=["m1", "m2"])
        pack.attr["axis"].i = 0
        pack.attr["N"].i = 2
        pack.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[2, 1, 20]]))
        
        graph_def.node.extend([x1, x2, w, m1, m2, pack])
        
        optimizer = GraphOptimizer(graph_def)
        optimized = PackVectorizePass().transform(optimizer)
        
        node_map = {n.name: n for n in optimized.node}
        bmm = next((n for n in optimized.node if n.op == "BatchMatMulV2" and "pack" in n.name.lower()), None)
        self.assertIsNotNone(bmm)
        # Weight should be shared
        self.assertEqual(bmm.input[1], "W")

    def test_matmul_different_weights(self):
        """Test hoisting MatMul with different weights."""
        graph_def = tf.GraphDef()
        # x1, x2: [1, 10], W1, W2: [10, 20]
        x1 = create_node("Placeholder", "x1")
        x1.attr["shape"].shape.CopyFrom(tf.TensorShape([1, 10]).as_proto())
        x2 = create_node("Placeholder", "x2")
        x2.attr["shape"].shape.CopyFrom(tf.TensorShape([1, 10]).as_proto())
        
        w1 = self._create_const("W1", tf.initializers.glorot_uniform()([10, 20]).eval(session=tf.Session()), dtype=tf.float32)
        w2 = self._create_const("W2", tf.initializers.glorot_uniform()([10, 20]).eval(session=tf.Session()), dtype=tf.float32)
        
        m1 = create_node("MatMul", "m1", inputs=["x1", "W1"])
        m1.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 20]]))
        m2 = create_node("MatMul", "m2", inputs=["x2", "W2"])
        m2.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[1, 20]]))
        
        # Pack([1, 20], [1, 20], axis=0) -> [2, 1, 20]
        pack = create_node("Pack", "pack", inputs=["m1", "m2"])
        pack.attr["axis"].i = 0
        pack.attr["N"].i = 2
        pack.attr["_output_shapes"].CopyFrom(make_output_shapes_attr([[2, 1, 20]]))
        
        graph_def.node.extend([x1, x2, w1, w2, m1, m2, pack])
        
        optimizer = GraphOptimizer(graph_def)
        optimized = PackVectorizePass().transform(optimizer)
        
        node_map = {n.name: n for n in optimized.node}
        bmm = next((n for n in optimized.node if n.op == "BatchMatMulV2" and "pack" in n.name.lower()), None)
        self.assertIsNotNone(bmm)
        # Second input should be a Pack node
        w_pack_name = bmm.input[1]
        self.assertEqual(node_map[w_pack_name].op, "Pack")
        self.assertEqual(len(node_map[w_pack_name].input), 2)

if __name__ == "__main__":
    unittest.main()
