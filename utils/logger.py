import logging
import functools
import time

# Define Log Levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR


# Singleton logger setup
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


core_logger = get_logger("GraphOptimizer.Core")
tf_logger = get_logger("GraphOptimizer.TF")
torch_logger = get_logger("GraphOptimizer.Torch")


def set_log_level(level):
    """Set log level for all framework loggers."""
    core_logger.setLevel(level)
    tf_logger.setLevel(level)
    torch_logger.setLevel(level)


def _get_optimizer_logger(optimizer):
    """Returns the appropriate logger depending on the optimizer type."""
    if optimizer:
        class_name = optimizer.__class__.__name__
        if "Torch" in class_name:
            return torch_logger
        elif "TF" in class_name:
            return tf_logger
    return core_logger


def trace_transformation(func):
    """Aspect: Log when a transformation/rewriter is executed."""

    @functools.wraps(func)
    def wrapper(match, optimizer, *args, **kwargs):
        log = _get_optimizer_logger(optimizer)
        start_time = time.time()
        result = func(match, optimizer, *args, **kwargs)
        duration = (time.time() - start_time) * 1000

        # Only log when optimization actually happened (result is not None)
        if result:
            # Handle both list format and RewriteResult format
            node_count = (
                len(result.new_nodes) if hasattr(result, "new_nodes") else len(result)
            )
            # Get anchor node name from match context
            anchor_name = (
                next(iter(match.all_matched_nodes), "unknown")
                if match.all_matched_nodes
                else "unknown"
            )
            # Get pass name from optimizer
            pass_name = getattr(optimizer, "current_pass_name", None)
            prefix = f"[{pass_name}] " if pass_name else ""
            log.info(
                f"{prefix}Rewriter {func.__name__} matched at {anchor_name}, generated {node_count} nodes ({duration:.2f}ms)"
            )
        return result

    return wrapper


def log_optimization(func):
    """Aspect: Log the overall optimization process."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        log = _get_optimizer_logger(self)
        pass_name = kwargs.get("pass_name")
        if pass_name is None and len(args) > 0:
            pass_name = args[0]

        prefix = f"[{pass_name}] " if pass_name else ""
        # Use the universal node_count property from BaseOptimizer
        original_node_count = self.node_count
        log.info(
            f"{prefix}Starting graph optimization pass... ({original_node_count} nodes)"
        )
        start_time = time.time()

        result_graph = func(self, *args, **kwargs)

        duration = time.time() - start_time
        final_node_count = self.node_count
        log.info(
            f"{prefix}Optimization finished in {duration:.3f}s. "
            f"Nodes: {original_node_count} -> {final_node_count}"
        )
        return result_graph

    return wrapper


def log_match(func):
    """Aspect: Log matching attempts (DEBUG level)."""

    @functools.wraps(func)
    def wrapper(self, node, optimizer, context=None):
        log = _get_optimizer_logger(optimizer)
        res = func(self, node, optimizer, context)
        if res:
            pass_name = getattr(optimizer, "current_pass_name", None)
            prefix = f"[{pass_name}] " if pass_name else ""
            log.debug(f"{prefix}Matched pattern on node: {node.name} (Op: {node.op})")
        return res

    return wrapper
