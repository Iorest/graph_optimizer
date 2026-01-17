import logging
import functools
import time

# Define Log Levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR


# Singleton logger setup
def get_logger(name="GraphOptimizer"):
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


logger = get_logger()


def set_log_level(level):
    logger.setLevel(level)


def trace_transformation(func):
    """Aspect: Log when a transformation/rewriter is executed."""

    @functools.wraps(func)
    def wrapper(match, optimizer, *args, **kwargs):
        start_time = time.time()
        result = func(match, optimizer, *args, **kwargs)
        duration = (time.time() - start_time) * 1000
        
        # Only log when optimization actually happened (result is not None)
        if result:
            # Handle both list format and RewriteResult format
            node_count = len(result.new_nodes) if hasattr(result, 'new_nodes') else len(result)
            # Get anchor node name from match context
            anchor_name = next(iter(match.all_matched_nodes), "unknown") if match.all_matched_nodes else "unknown"
            # Get pass name from optimizer
            pass_name = getattr(optimizer, 'current_pass_name', None)
            prefix = f"[{pass_name}] " if pass_name else ""
            logger.info(
                f"{prefix}Rewriter {func.__name__} matched at {anchor_name}, generated {node_count} nodes ({duration:.2f}ms)"
            )
        return result

    return wrapper


def log_optimization(func):
    """Aspect: Log the overall optimization process."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        pass_name = kwargs.get("pass_name")
        if pass_name is None and len(args) > 0:
            pass_name = args[0]

        prefix = f"[{pass_name}] " if pass_name else ""
        # 使用 graph_def.node 获取真实节点数，避免 self.nodes 未同步的问题
        original_node_count = len(self.graph_def.node)
        logger.info(f"{prefix}Starting graph optimization pass... ({original_node_count} nodes)")
        start_time = time.time()

        result_graph = func(self, *args, **kwargs)

        duration = time.time() - start_time
        final_node_count = len(result_graph.node)
        logger.info(
            f"{prefix}Optimization finished in {duration:.3f}s. "
            f"Nodes: {original_node_count} -> {final_node_count}"
        )
        return result_graph

    return wrapper


def log_match(func):
    """Aspect: Log matching attempts (DEBUG level)."""

    @functools.wraps(func)
    def wrapper(self, node, optimizer, context=None):
        res = func(self, node, optimizer, context)
        if res:
            pass_name = getattr(optimizer, 'current_pass_name', None)
            prefix = f"[{pass_name}] " if pass_name else ""
            logger.debug(f"{prefix}Matched pattern on node: {node.name} (Op: {node.op})")
        return res

    return wrapper
