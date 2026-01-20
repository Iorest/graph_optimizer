import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, tensor_shape_pb2


def infer_shapes_with_dynamic_batch(input_pb_path, output_pb_path, input_shape_map=None):
    """
    推断并回填 _output_shapes，同时严格保留动态维度 (-1)。
    
    Args:
        input_pb_path: 输入模型路径
        output_pb_path: 输出模型路径
        input_shape_map: (可选) 字典 {'input_name': [None, 224, 224, 3]}
                         如果原图 Placeholder 没有任何形状信息，必须通过这个参数指定，
                         否则无法向下推导。这里的 None 会被转为 -1。
    """
    print(f"Loading graph from {input_pb_path}...")
    
    # 1. 读取 GraphDef
    with tf.io.gfile.GFile(input_pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # 2. (可选) 如果 Placeholder 形状完全缺失，先补上带 -1 的形状
    # 这一步是为了让 TF 知道“这是一个 Rank=4 的张量，只是 Batch 未知”，而不是“完全不知道是什么”
    if input_shape_map:
        print("Applying explicit input shapes (preserving dynamic dims)...")
        for node in graph_def.node:
            if node.name in input_shape_map:
                shape_list = input_shape_map[node.name]
                
                # 构造 Shape Proto
                shape_proto = tensor_shape_pb2.TensorShapeProto()
                for dim in shape_list:
                    new_dim = shape_proto.dim.add()
                    # Python 的 None 或 -1 都转为 Proto 的 -1
                    if dim is None or dim == -1:
                        new_dim.size = -1
                    else:
                        new_dim.size = int(dim)
                
                # 更新 shape 属性，让 import_graph_def 能读懂
                if "shape" in node.attr:
                    node.attr["shape"].shape.CopyFrom(shape_proto)

    # 3. 导入到 TF Graph 进行推导
    print("Running shape inference (dynamic)...")
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        
        updated_count = 0
        
        # 4. 遍历所有节点
        for node in graph_def.node:
            try:
                op = graph.get_operation_by_name(node.name)
                
                # 如果没有输出，跳过
                if len(op.outputs) == 0:
                    continue

                output_shapes_list = []
                
                for output_tensor in op.outputs:
                    shape = output_tensor.shape
                    
                    # === 关键点：处理 TensorShape 对象 ===
                    shape_proto = tensor_shape_pb2.TensorShapeProto()
                    
                    if shape.dims is None:
                        # Case A: 完全未知的形状 (Unknown Rank)
                        shape_proto.unknown_rank = True
                    else:
                        # Case B: 已知 Rank，但某些维度可能是 None
                        for dim in shape.dims:
                            dim_proto = shape_proto.dim.add()
                            if dim.value is None:
                                # 核心：将内存中的 None 显式写成 Proto 中的 -1
                                dim_proto.size = -1
                            else:
                                dim_proto.size = dim.value
                                
                    output_shapes_list.append(shape_proto)

                # 5. 回写 _output_shapes
                attr_value = attr_value_pb2.AttrValue()
                attr_value.list.shape.extend(output_shapes_list)
                node.attr["_output_shapes"].CopyFrom(attr_value)
                
                updated_count += 1
                
            except KeyError:
                pass
            except Exception as e:
                print(f"Warning: Could not infer shape for {node.name}: {e}")

    # 6. 保存
    print(f"Inference complete. Updated {updated_count} nodes.")
    print(f"Saving to {output_pb_path}...")
    with tf.io.gfile.GFile(output_pb_path, "wb") as f:
        f.write(graph_def.SerializeToString())
    print("Done.")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 配置路径
    INPUT_MODEL = "./graph_def_rankmixer_infer.pb"
    OUTPUT_MODEL = "./graph_def_rankmixer_infer_with_shapes.pb"
    
    # [进阶技巧]
    # 如果你的输入 Placeholder 是 [None, 224, 224, 3]，
    # 很多中间层可能推断出 [None, 112, 112, 64]。
    # 如果你希望把 None 固定下来（比如变成 1），这通常需要更复杂的图重写。
    # 但对于标准的 _output_shapes 修复，上述代码已经足够。
    
    try:
        infer_shapes_with_dynamic_batch(INPUT_MODEL, OUTPUT_MODEL)
    except Exception as e:
        print(f"Execution failed: {e}")