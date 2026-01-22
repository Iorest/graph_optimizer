import hashlib

def hash_tensor_value(tensor_proto):
    """
    Computes a hash of the tensor's value for efficient comparison.

    Args:
        tensor_proto: The TensorProto object.

    Returns:
        A string hash of the tensor's value.
    """
    # Using MD5 for speed. It's a performance optimization, not a security feature.
    hasher = hashlib.md5()
    hasher.update(tensor_proto.SerializeToString())
    return hasher.hexdigest()
