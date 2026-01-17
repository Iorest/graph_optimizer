import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import tensorflow.compat.v1 as tf
import numpy as np
from graph_optimizer.utils import load_graph, create_node, create_complex_concat_graph
from graph_optimizer.runner import OptimizationPipeline
from graph_optimizer.utils.logger import set_log_level, DEBUG, INFO
tf.disable_v2_behavior()
set_log_level(DEBUG)  # Less verbose for multi-test


def get_placeholder_specs(graph_def, graph):
    """Get placeholder specs (name, tensor, shape, dtype) for a graph."""
    specs = []
    for node in graph_def.node:
        if node.op == "Placeholder":
            dtype = node.attr.get("dtype", None)
            dtype_val = dtype.type if dtype else 0
            
            tensor = graph.get_tensor_by_name(node.name + ":0")
            if tensor.shape.ndims == 0:
                shape = ()  # scalar
            elif tensor.shape.dims is not None:
                shape = tuple(max(1, d) if d is not None and d >= 0 else 1 
                             for d in tensor.shape.as_list())
            else:
                shape = (1,)
            specs.append((node.name, tensor, shape, dtype_val))
    return specs


def generate_random_feed(specs, seed=None):
    """Generate random feed dict for given placeholder specs."""
    if seed is not None:
        np.random.seed(seed)
    feed_dict = {}
    random_data = {}
    for name, tensor, shape, dtype_val in specs:
        if dtype_val == 3:  # DT_INT32
            if shape == ():
                data = np.int32(np.random.randint(1, 100))
            else:
                data = np.random.randint(0, 100, size=shape).astype(np.int32)
        else:  # DT_FLOAT (1) or others
            if shape == ():
                data = np.float32(np.random.rand())
            else:
                data = np.random.rand(*shape).astype(np.float32)
        feed_dict[tensor] = data
        random_data[name] = data
    return feed_dict, random_data


def run_consistency_tests(original_graph, optimized_graph, output_nodes, num_tests=10):
    """Run multiple consistency tests with different random inputs."""
    print(f"\n{'='*60}")
    print(f"Running {num_tests} consistency tests with random inputs...")
    print(f"{'='*60}")
    
    all_passed = True
    max_diffs = []
    
    # Build graphs and get placeholder specs
    g_orig = tf.Graph()
    g_opt = tf.Graph()
    
    with g_orig.as_default():
        tf.import_graph_def(original_graph, name="")
        specs_orig = get_placeholder_specs(original_graph, g_orig)
    
    with g_opt.as_default():
        tf.import_graph_def(optimized_graph, name="")
        specs_opt = get_placeholder_specs(optimized_graph, g_opt)
    
    # Verify placeholders match
    print(f"\nOriginal graph placeholders: {len(specs_orig)}")
    print(f"Optimized graph placeholders: {len(specs_opt)}")
    
    # Count data vs resource placeholders
    data_ph = [s for s in specs_orig if 'resource' not in s[0].lower()]
    resource_ph = [s for s in specs_orig if 'resource' in s[0].lower()]
    print(f"  Data placeholders: {len(data_ph)}")
    print(f"  Resource placeholders (frozen variables): {len(resource_ph)}")
    
    names_orig = [s[0] for s in specs_orig]
    names_opt = [s[0] for s in specs_opt]
    if names_orig != names_opt:
        print(f"WARNING: Placeholder names differ!")
        print(f"  Original: {names_orig[:5]}...")
        print(f"  Optimized: {names_opt[:5]}...")
    else:
        print(f"Placeholder names match ✓")
    
    # Create sessions
    sess_orig = tf.Session(graph=g_orig)
    sess_opt = tf.Session(graph=g_opt)
    
    try:
        output_tensors_orig = [g_orig.get_tensor_by_name(n + ":0") for n in output_nodes]
        output_tensors_opt = [g_opt.get_tensor_by_name(n + ":0") for n in output_nodes]
        
        for test_idx in range(num_tests):
            seed = 42 + test_idx * 1000  # Different seed for each test
            
            # Generate random data using original graph specs
            feed_orig, random_data = generate_random_feed(specs_orig, seed)
            
            # Build feed dict for optimized graph using SAME random data
            feed_opt = {}
            for name, tensor, shape, dtype_val in specs_opt:
                feed_opt[tensor] = random_data[name]
            
            # First test: show input summary (only data placeholders, not frozen vars)
            if test_idx == 0:
                print(f"\n  Input data sample (first test, data placeholders only):")
                data_names = [n for n in random_data.keys() if 'resource' not in n.lower()]
                for name in data_names[:5]:
                    data = random_data[name]
                    if isinstance(data, np.ndarray):
                        dtype_str = "int32" if data.dtype == np.int32 else "float32"
                        print(f"    {name}:")
                        print(f"      shape={data.shape}, dtype={dtype_str}")
                        print(f"      values={data.flatten()[:8]}{'...' if data.size > 8 else ''}")
                    else:
                        print(f"    {name}: scalar={data}")
                print(f"    ... ({len(data_names)} data placeholders, {len(random_data) - len(data_names)} frozen vars)")
            
            # Run both graphs
            results_orig = sess_orig.run(output_tensors_orig, feed_dict=feed_orig)
            results_opt = sess_opt.run(output_tensors_opt, feed_dict=feed_opt)
            
            # Compare results
            test_passed = True
            test_max_diff = 0.0
            
            for i, name in enumerate(output_nodes):
                orig = results_orig[i]
                opt = results_opt[i]
                diff = np.max(np.abs(orig - opt))
                test_max_diff = max(test_max_diff, diff)
                
                if not np.allclose(orig, opt, atol=1e-5):
                    test_passed = False
            
            max_diffs.append(test_max_diff)
            status = "PASS" if test_passed else "FAIL"
            print(f"\n  Test {test_idx+1:2d}: {status}  (seed={seed})")
            for i, name in enumerate(output_nodes):
                orig = results_orig[i]
                opt = results_opt[i]
                diff = np.max(np.abs(orig - opt))
                # Show output details
                print(f"    Output '{name}':")
                print(f"      Shape: {orig.shape}")
                print(f"      Original:  {orig.flatten()[:10]}{'...' if orig.size > 10 else ''}")
                print(f"      Optimized: {opt.flatten()[:10]}{'...' if opt.size > 10 else ''}")
                print(f"      Max diff: {diff:.2e}")
            
            if not test_passed:
                all_passed = False
                
    finally:
        sess_orig.close()
        sess_opt.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {sum(1 for d in max_diffs if d < 1e-5)}/{num_tests} tests passed")
    print(f"Max diff across all tests: {max(max_diffs):.2e}")
    print(f"Mean diff across all tests: {np.mean(max_diffs):.2e}")
    print(f"{'='*60}")
    
    return all_passed


def main():
    # Get demo directory path
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(demo_dir, "graph_def_rankmixer_infer.pb")
    output_path = os.path.join(demo_dir, "graph_def_rankmixer_infer_optimized.pb")

    # Load original graph
    print("Loading original graph...")
    original_graph = load_graph(input_path)
    print(f"Original graph: {len(original_graph.node)} nodes")

    # Optimize
    print("\nStarting optimization pipeline...")
    pipeline = OptimizationPipeline(
        input_graph=input_path,
        output_graph=output_path,
        level=3,
        debug=True,  # 启用 debug 模式，生成 run_+时间 目录
        output_nodes=["predicts"],
        protected_nodes=["compile_batch_size_ret"],
    )
    pipeline.run()

    # Load optimized graph
    optimized_graph = load_graph(output_path)
    print(f"Optimized graph: {len(optimized_graph.node)} nodes")
    print(f"Reduction: {len(original_graph.node) - len(optimized_graph.node)} nodes "
          f"({100*(1-len(optimized_graph.node)/len(original_graph.node)):.1f}%)")

    # Run consistency tests
    output_nodes = ["predicts"]
    all_passed = run_consistency_tests(original_graph, optimized_graph, output_nodes, num_tests=20)

    if all_passed:
        print("\n✓ All consistency tests passed!")
        return 0
    else:
        print("\n✗ Some consistency tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
