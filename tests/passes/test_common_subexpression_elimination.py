"""
Comprehensive tests for Common Subexpression Elimination (CSE) Pass.

Tests cover:
1. Basic duplicate elimination
2. Const node deduplication  
3. Control dependencies preservation
4. Multi-port outputs
5. Iterative convergence
6. Skip ops (Placeholder, Variable, Identity)
7. Edge cases (empty graph, single node, no duplicates)
"""

import unittest
import tensorflow.compat.v1 as tf
from graph_optimizer.core import GraphOptimizer
from graph_optimizer.optimizers.common_subexpression_elimination import CommonSubexpressionElimination
from graph_optimizer.utils import create_node
from tensorflow.core.framework import attr_value_pb2

tf.disable_v2_behavior()


class TestCSE(unittest.TestCase):
    """Test suite for Common Subexpression Elimination."""
    
    def create_graph(self, nodes):
        """Helper to create a GraphDef from node list."""
        graph_def = tf.GraphDef()
        graph_def.node.extend(nodes)
        return graph_def
    
    def _make_dtype_attr(self, dtype):
        """Helper to create dtype attribute."""
        return attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
    
    def _make_const(self, name, value, dtype):
        """Helper to create Const node."""
        return create_node("Const", name, attr={
            "dtype": self._make_dtype_attr(dtype),
            "value": attr_value_pb2.AttrValue(
                tensor=tf.make_tensor_proto(value, dtype=dtype)
            )
        })
    
    def _make_placeholder(self, name, dtype):
        """Helper to create Placeholder node."""
        return create_node("Placeholder", name, attr={"dtype": self._make_dtype_attr(dtype)})
    
    def test_basic_duplicate_elimination(self):
        """Test 1: Basic duplicate node elimination."""
        nodes = [
            self._make_placeholder("input", tf.int32),
            self._make_const("weights_1", 1.0, tf.float32),
            self._make_const("weights_2", 1.0, tf.float32),  # Duplicate
            create_node("Add", "add_1", inputs=["input", "weights_1"]),
            create_node("Add", "add_2", inputs=["input", "weights_2"]),  # Will become duplicate after weights merge
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should eliminate weights_2 AND add_2 (iterative convergence) = 2 nodes
        self.assertEqual(final_count, initial_count - 2)
        self.assertIn("weights_1", optimizer.nodes)
        self.assertNotIn("weights_2", optimizer.nodes)
        self.assertIn("add_1", optimizer.nodes)
        self.assertNotIn("add_2", optimizer.nodes)
    
    def test_const_different_dtypes(self):
        """Test 2: Const nodes with same value but different dtypes should NOT be merged."""
        nodes = [
            self._make_const("const_int", 1, tf.int32),
            self._make_const("const_float", 1.0, tf.float32),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should NOT eliminate any node (different dtypes)
        self.assertEqual(final_count, initial_count)
        self.assertIn("const_int", optimizer.nodes)
        self.assertIn("const_float", optimizer.nodes)
    
    def test_const_same_dtype_same_value(self):
        """Test 3: Const nodes with same dtype and same value should be merged."""
        nodes = [
            self._make_const("const_1", 3.14, tf.float32),
            self._make_const("const_2", 3.14, tf.float32),  # Duplicate
            create_node("Add", "add", inputs=["const_1", "const_2"]),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should eliminate one Const
        self.assertEqual(final_count, initial_count - 1)
        self.assertTrue("const_1" in optimizer.nodes or "const_2" in optimizer.nodes)
        self.assertFalse("const_1" in optimizer.nodes and "const_2" in optimizer.nodes)
    
    def test_control_dependencies(self):
        """Test 4: CSE should preserve control dependencies."""
        nodes = [
            self._make_placeholder("input", tf.float32),
            self._make_const("w1", 1.0, tf.float32),
            self._make_const("w2", 1.0, tf.float32),  # Duplicate
            create_node("NoOp", "ctrl_op"),
            create_node("Add", "add_1", inputs=["input", "w1", "^ctrl_op"]),  # With control dep
            create_node("Add", "add_2", inputs=["input", "w2"]),  # Without control dep
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        cse_pass.transform(optimizer)
        
        # Verify control dependencies are preserved
        add_1_node = optimizer.nodes.get("add_1")
        self.assertIsNotNone(add_1_node)
        self.assertIn("^ctrl_op", add_1_node.input)
    
    def test_multi_port_outputs(self):
        """Test 5: CSE should handle multi-port outputs correctly."""
        nodes = [
            self._make_placeholder("input", tf.float32),
            create_node("Split", "split_1", inputs=["input"], attr={"num_split": attr_value_pb2.AttrValue(i=2)}),
            create_node("Split", "split_2", inputs=["input"], attr={"num_split": attr_value_pb2.AttrValue(i=2)}),  # Duplicate
            create_node("Add", "add_1", inputs=["split_1:0", "split_1:1"]),
            create_node("Add", "add_2", inputs=["split_2:0", "split_2:1"]),  # Will become duplicate after split merge
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should eliminate split_2 AND add_2 (iterative convergence) = 2 nodes
        self.assertEqual(final_count, initial_count - 2)
        
        # Verify split_2 and add_2 are removed
        self.assertIn("split_1", optimizer.nodes)
        self.assertNotIn("split_2", optimizer.nodes)
        self.assertIn("add_1", optimizer.nodes)
        self.assertNotIn("add_2", optimizer.nodes)
    
    def test_iterative_convergence(self):
        """Test 6: CSE should iterate until convergence."""
        # Create a graph where eliminating duplicates reveals more duplicates
        nodes = [
            self._make_placeholder("x", tf.float32),
            self._make_placeholder("y", tf.float32),
            self._make_const("c1", 2.0, tf.float32),
            self._make_const("c2", 2.0, tf.float32),  # Duplicate
            # First level duplicates
            create_node("Add", "add_1", inputs=["x", "c1"]),
            create_node("Add", "add_2", inputs=["x", "c2"]),  # Will become duplicate after c2->c1
            # Second level (will become duplicates after first level merge)
            create_node("Mul", "mul_1", inputs=["add_1", "y"]),
            create_node("Mul", "mul_2", inputs=["add_2", "y"]),  # Will become duplicate
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should eliminate: c2, add_2, mul_2 (3 nodes)
        expected_removed = 3
        self.assertEqual(final_count, initial_count - expected_removed)
        
        # Verify cascading elimination worked
        self.assertIn("c1", optimizer.nodes)
        self.assertNotIn("c2", optimizer.nodes)
        self.assertIn("add_1", optimizer.nodes)
        self.assertNotIn("add_2", optimizer.nodes)
        self.assertIn("mul_1", optimizer.nodes)
        self.assertNotIn("mul_2", optimizer.nodes)
    
    def test_skip_placeholder(self):
        """Test 7: Placeholders should never be deduplicated."""
        # Create two "identical" placeholders (same dtype)
        nodes = [
            self._make_placeholder("input_1", tf.float32),
            self._make_placeholder("input_2", tf.float32),
            create_node("Add", "add", inputs=["input_1", "input_2"]),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should NOT eliminate any node
        self.assertEqual(final_count, initial_count)
        self.assertIn("input_1", optimizer.nodes)
        self.assertIn("input_2", optimizer.nodes)
    
    def test_skip_variable(self):
        """Test 8: Variables should never be deduplicated."""
        nodes = [
            create_node("VariableV2", "var_1", attr={
                "dtype": self._make_dtype_attr(tf.float32),
                "shape": attr_value_pb2.AttrValue(shape=tf.TensorShape([10]).as_proto())
            }),
            create_node("VariableV2", "var_2", attr={
                "dtype": self._make_dtype_attr(tf.float32),
                "shape": attr_value_pb2.AttrValue(shape=tf.TensorShape([10]).as_proto())
            }),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should NOT eliminate any node
        self.assertEqual(final_count, initial_count)
        self.assertIn("var_1", optimizer.nodes)
        self.assertIn("var_2", optimizer.nodes)
    
    def test_skip_identity(self):
        """Test 9: Identity nodes should be skipped (handled by dedicated pass)."""
        nodes = [
            self._make_placeholder("input", tf.float32),
            create_node("Identity", "id_1", inputs=["input"]),
            create_node("Identity", "id_2", inputs=["input"]),  # "Duplicate" but should be skipped
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should NOT eliminate Identity nodes (dedicated pass handles them)
        self.assertEqual(final_count, initial_count)
        self.assertIn("id_1", optimizer.nodes)
        self.assertIn("id_2", optimizer.nodes)
    
    def test_empty_graph(self):
        """Test 10: Empty graph should not crash."""
        graph_def = self.create_graph([])
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        cse_pass.transform(optimizer)
        
        self.assertEqual(len(optimizer.nodes), 0)
    
    def test_single_node(self):
        """Test 11: Graph with single node should not crash."""
        nodes = [self._make_placeholder("input", tf.float32)]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        self.assertEqual(final_count, initial_count)
        self.assertIn("input", optimizer.nodes)
    
    def test_no_duplicates(self):
        """Test 12: Graph with no duplicates should remain unchanged."""
        nodes = [
            self._make_placeholder("input", tf.float32),
            self._make_const("weights", 1.0, tf.float32),
            create_node("MatMul", "matmul", inputs=["input", "weights"]),
            create_node("Relu", "relu", inputs=["matmul"]),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        self.assertEqual(final_count, initial_count)
    
    def test_different_inputs_same_op(self):
        """Test 13: Same op with different inputs should NOT be merged."""
        nodes = [
            self._make_placeholder("x", tf.float32),
            self._make_placeholder("y", tf.float32),
            self._make_placeholder("z", tf.float32),
            create_node("Add", "add_1", inputs=["x", "y"]),
            create_node("Add", "add_2", inputs=["x", "z"]),  # Different input
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        self.assertEqual(final_count, initial_count)
        self.assertIn("add_1", optimizer.nodes)
        self.assertIn("add_2", optimizer.nodes)
    
    def test_different_attrs_same_inputs(self):
        """Test 14: Same inputs with different attributes should NOT be merged."""
        nodes = [
            self._make_placeholder("input", tf.float32),
            self._make_const("axis_0", 0, tf.int32),
            self._make_const("axis_1", 1, tf.int32),
            create_node("Split", "split_1", inputs=["axis_0", "input"], 
                       attr={"num_split": attr_value_pb2.AttrValue(i=2)}),
            create_node("Split", "split_2", inputs=["axis_1", "input"], 
                       attr={"num_split": attr_value_pb2.AttrValue(i=2)}),  # Different axis
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        cse_pass.transform(optimizer)
        
        # split_1 and split_2 have different axis inputs, so shouldn't merge
        self.assertIn("split_1", optimizer.nodes)
        self.assertIn("split_2", optimizer.nodes)
    
    def test_complex_graph(self):
        """Test 15: Complex graph with multiple types of duplicates."""
        nodes = [
            # Inputs
            self._make_placeholder("x", tf.float32),
            self._make_placeholder("y", tf.float32),
            
            # Duplicate constants
            self._make_const("c1", 1.0, tf.float32),
            self._make_const("c2", 1.0, tf.float32),  # dup
            self._make_const("c3", 2.0, tf.float32),
            self._make_const("c4", 2.0, tf.float32),  # dup
            
            # Duplicate operations
            create_node("Add", "add_1", inputs=["x", "c1"]),
            create_node("Add", "add_2", inputs=["x", "c2"]),  # will be dup after c2->c1
            create_node("Mul", "mul_1", inputs=["y", "c3"]),
            create_node("Mul", "mul_2", inputs=["y", "c4"]),  # will be dup after c4->c3
            
            # More operations
            create_node("Add", "final_1", inputs=["add_1", "mul_1"]),
            create_node("Add", "final_2", inputs=["add_2", "mul_2"]),  # will be dup
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Expected removals: c2, c4, add_2, mul_2, final_2 = 5 nodes
        expected_removed = 5
        self.assertEqual(final_count, initial_count - expected_removed)
        
        # Verify canonical nodes remain
        self.assertIn("c1", optimizer.nodes)
        self.assertIn("c3", optimizer.nodes)
        self.assertIn("add_1", optimizer.nodes)
        self.assertIn("mul_1", optimizer.nodes)
        self.assertIn("final_1", optimizer.nodes)
        
        # Verify duplicates removed
        self.assertNotIn("c2", optimizer.nodes)
        self.assertNotIn("c4", optimizer.nodes)
        self.assertNotIn("add_2", optimizer.nodes)
        self.assertNotIn("mul_2", optimizer.nodes)
        self.assertNotIn("final_2", optimizer.nodes)
    
    def test_canonical_selection(self):
        """Test 16: Verify canonical node selection (shortest name)."""
        nodes = [
            self._make_placeholder("x", tf.float32),
            self._make_const("very_long_name_1", 1.0, tf.float32),
            self._make_const("short", 1.0, tf.float32),
            self._make_const("very_long_name_2", 1.0, tf.float32),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        cse_pass.transform(optimizer)
        
        # "short" should be kept as canonical (shortest name)
        self.assertIn("short", optimizer.nodes)
        self.assertNotIn("very_long_name_1", optimizer.nodes)
        self.assertNotIn("very_long_name_2", optimizer.nodes)
    
    def test_skip_stateful_ops(self):
        """Test 17: Stateful operations should never be deduplicated."""
        nodes = [
            self._make_placeholder("input", tf.float32),
            # Random ops (stateful - each call returns different results)
            create_node("RandomUniform", "rand_1", inputs=["input"], attr={
                "dtype": self._make_dtype_attr(tf.float32),
                "seed": attr_value_pb2.AttrValue(i=42)
            }),
            create_node("RandomUniform", "rand_2", inputs=["input"], attr={
                "dtype": self._make_dtype_attr(tf.float32),
                "seed": attr_value_pb2.AttrValue(i=42)  # Same seed but still shouldn't merge
            }),
            # Print ops (side effects)
            create_node("Print", "print_1", inputs=["input", "input"]),
            create_node("Print", "print_2", inputs=["input", "input"]),  # Same inputs but has side effects
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should NOT eliminate any stateful ops
        self.assertEqual(final_count, initial_count)
        self.assertIn("rand_1", optimizer.nodes)
        self.assertIn("rand_2", optimizer.nodes)
        self.assertIn("print_1", optimizer.nodes)
        self.assertIn("print_2", optimizer.nodes)
    
    def test_skip_variable_read_ops(self):
        """Test 18: Variable read operations should not be deduplicated."""
        nodes = [
            # TF2.x style variables
            create_node("VarHandleOp", "var", attr={
                "dtype": self._make_dtype_attr(tf.float32),
                "shape": attr_value_pb2.AttrValue(shape=tf.TensorShape([10]).as_proto())
            }),
            # Multiple reads from same variable (each read might see different values)
            create_node("ReadVariableOp", "read_1", inputs=["var"], attr={
                "dtype": self._make_dtype_attr(tf.float32)
            }),
            create_node("ReadVariableOp", "read_2", inputs=["var"], attr={
                "dtype": self._make_dtype_attr(tf.float32)
            }),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should NOT eliminate variable read ops (variable may change between reads)
        self.assertEqual(final_count, initial_count)
        self.assertIn("read_1", optimizer.nodes)
        self.assertIn("read_2", optimizer.nodes)
    
    def test_skip_control_flow_ops(self):
        """Test 19: Control flow operations should not be deduplicated."""
        nodes = [
            self._make_placeholder("pred", tf.bool),
            self._make_placeholder("data", tf.float32),
            # Switch nodes with same inputs but in different control flow contexts
            create_node("Switch", "switch_1", inputs=["data", "pred"]),
            create_node("Switch", "switch_2", inputs=["data", "pred"]),
            # Merge nodes
            create_node("Merge", "merge_1", inputs=["switch_1:0", "switch_1:1"]),
            create_node("Merge", "merge_2", inputs=["switch_2:0", "switch_2:1"]),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should NOT eliminate control flow ops
        self.assertEqual(final_count, initial_count)
        self.assertIn("switch_1", optimizer.nodes)
        self.assertIn("switch_2", optimizer.nodes)
        self.assertIn("merge_1", optimizer.nodes)
        self.assertIn("merge_2", optimizer.nodes)
    
    def test_different_control_dependencies(self):
        """Test 20: Nodes with different control dependencies should NOT be merged."""
        nodes = [
            self._make_placeholder("input", tf.float32),
            self._make_const("w", 1.0, tf.float32),
            create_node("NoOp", "ctrl_1"),
            create_node("NoOp", "ctrl_2"),
            # Same op and data inputs, but different control dependencies
            create_node("Add", "add_1", inputs=["input", "w", "^ctrl_1"]),
            create_node("Add", "add_2", inputs=["input", "w", "^ctrl_2"]),
        ]
        
        graph_def = self.create_graph(nodes)
        optimizer = GraphOptimizer(graph_def)
        cse_pass = CommonSubexpressionElimination()
        
        initial_count = len(optimizer.nodes)
        cse_pass.transform(optimizer)
        final_count = len(optimizer.nodes)
        
        # Should NOT merge nodes with different control dependencies
        self.assertEqual(final_count, initial_count)
        self.assertIn("add_1", optimizer.nodes)
        self.assertIn("add_2", optimizer.nodes)


if __name__ == "__main__":
    unittest.main()
