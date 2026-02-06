import time
import tensorflow.compat.v1 as tf
import numpy as np
from graph_optimizer.core import GraphOptimizer, Op, Any
from graph_optimizer.transforms.scalar.algebraic_simplify import AlgebraicSimplifyPass
from graph_optimizer.transforms.scalar.constant_fold import ConstantFoldPass
from graph_optimizer.utils.graph_utils import create_node, create_const_node

def create_large_sparse_graph(num_nodes=5000):
    graph_def = tf.GraphDef()
    # Create many placeholders
    for i in range(num_nodes):
        p = create_node("Placeholder", f"p_{i}", attr={"dtype": tf.AttrValue(type=tf.float32.as_datatype_enum)})
        graph_def.node.extend([p])

    # Create some Add(x, 0) nodes scattered around
    zero = create_const_node("zero", value=0, dtype="float32", shape=[])
    graph_def.node.extend([zero])

    for i in range(100):
        idx = i * (num_nodes // 100)
        a = create_node("Add", f"add_{i}", inputs=[f"p_{idx}", "zero"])
        graph_def.node.extend([a])

    return graph_def

def benchmark():
    num_nodes = 100000
    print(f"Generating graph with {num_nodes} nodes...")
    graph_def = create_large_sparse_graph(num_nodes)

    optimizer = GraphOptimizer(graph_def)
    pass_obj = AlgebraicSimplifyPass()

    print("Running AlgebraicSimplifyPass benchmark...")
    start_time = time.time()
    optimizer.optimize(pass_name="AlgebraicSimplify", max_iterations=1)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    benchmark()
