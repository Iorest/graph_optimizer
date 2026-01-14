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
        # We assume the first arg is typically the match object
        start_time = time.time()
        result = func(match, optimizer, *args, **kwargs)
        duration = (time.time() - start_time) * 1000
        
        # Only log when optimization actually happened (result is not None)
        if result:
            logger.info(f"Executing rewriter: {func.__name__}")
            logger.info(
                f"Rewriter {func.__name__} generated {len(result)} nodes ({duration:.2f}ms)"
            )
        else:
            logger.debug(f"Rewriter {func.__name__} returned None")
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
        logger.info(f"{prefix}Starting graph optimization pass...")
        original_node_count = len(self.nodes)
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
            logger.debug(f"Matched pattern on node: {node.name} (Op: {node.op})")
        return res

    return wrapper
