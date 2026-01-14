import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
from graph_optimizer.utils import load_graph, create_node, create_complex_concat_graph
from graph_optimizer.runner import OptimizationPipeline

tf.disable_v2_behavior()


def evaluate_graph(graph_def, output_node_names):
    """Run the graph and return outputs."""
    with tf.Graph().as_default() as g:
        tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            # Find all placeholders to feed
            feed_dict = {}
            for node in graph_def.node:
                if node.op == "Placeholder":
                    if "shape" in node.attr and node.attr["shape"].HasField("shape"):
                        shape_proto = node.attr["shape"].shape
                        shape = [d.size for d in shape_proto.dim]
                    elif (
                        "_output_shapes" in node.attr
                        and node.attr["_output_shapes"].list.shape
                    ):
                        shape = [
                            d.size
                            for d in node.attr["_output_shapes"].list.shape[0].dim
                        ]
                    else:
                        shape = []

                    shape = [max(1, d) if d >= 0 else 1 for d in shape]
                    data = (
                        np.random.rand(*shape).astype(np.float32)
                        if shape
                        else np.float32(np.random.rand())
                    )
                    tensor = g.get_tensor_by_name(node.name + ":0")
                    feed_dict[tensor] = data

            output_tensors = [
                g.get_tensor_by_name(name + ":0") for name in output_node_names
            ]
            return sess.run(output_tensors, feed_dict=feed_dict), feed_dict


def main():
    # 1. Generate the graph
    input_path = "demos/complex_concat_graph.pbtxt"
    output_path = "demos/complex_concat_graph_optimized.pbtxt"

    # Generate the complex graph
    print(f"Generating graph to {input_path}...")
    create_complex_concat_graph(input_path)

    # 2. Evaluate original graph
    print("Evaluating original graph...")
    original_graph = load_graph(input_path)
    output_nodes = ["output"]
    original_results, feed_dict_values = evaluate_graph(original_graph, output_nodes)

    # 3. Optimize
    print("\nStarting optimization pipeline...")
    pipeline = OptimizationPipeline(
        input_path, output_path, level=2, debug=True, output_nodes=["output"]
    )
    pipeline.run()

    # 4. Evaluate optimized graph
    print("\nEvaluating optimized graph...")
    optimized_graph = load_graph(output_path)

    with tf.Graph().as_default() as g:
        tf.import_graph_def(optimized_graph, name="")
        with tf.Session() as sess:
            optimized_feed_dict = {}
            for placeholder_tensor, value in feed_dict_values.items():
                new_tensor = g.get_tensor_by_name(placeholder_tensor.name)
                optimized_feed_dict[new_tensor] = value

            output_tensors = [
                g.get_tensor_by_name(name + ":0") for name in output_nodes
            ]
            optimized_results = sess.run(output_tensors, feed_dict=optimized_feed_dict)

    # 5. Numerical verification
    print("\n--- Numerical Verification ---")
    all_match = True
    for i, name in enumerate(output_nodes):
        orig = original_results[i]
        opt = optimized_results[i]
        if np.allclose(orig, opt, atol=1e-5):
            print(f"SUCCESS: Output '{name}' matches. Shape: {orig.shape}")
        else:
            max_diff = np.max(np.abs(orig - opt))
            print(f"FAILED: Output '{name}' mismatch! Max diff: {max_diff}")
            all_match = False

    if all_match:
        print("\nOptimization verified: Numerical results are identical.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
