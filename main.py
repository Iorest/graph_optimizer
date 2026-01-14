import argparse
import json
import sys
from . import optimizers  # Register all passes
from .utils.logger import logger as custom_logger

# Prevent unused import warning
_ = optimizers


DEFAULT_PASSES = ["constant_folding_add", "identity_removal"]


def main():
    parser = argparse.ArgumentParser(description="Graph Optimizer CLI")
    parser.add_argument(
        "--config", required=True, help="Path to JSON configuration file"
    )
    parser.add_argument("--input", help="Override input graph path")
    parser.add_argument("--output", help="Override output graph path")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (dump intermediate graphs)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Optimization level (default: 1). 1=Basic, 2=Advanced.",
    )
    args = parser.parse_args()

    # Load config if specified
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            custom_logger.error(f"Failed to load config file: {e}")
            sys.exit(1)

    try:
        from .runner import OptimizationPipeline

        pipeline = OptimizationPipeline(
            input_graph=args.input,
            output_graph=args.output,
            level=args.level,
            debug=args.debug,
            config=config,
            # Pass explicit config path just in case, though we loaded dict
            # Pipeline accepts dict 'config'
        )
        pipeline.run()
    except Exception as e:
        custom_logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
