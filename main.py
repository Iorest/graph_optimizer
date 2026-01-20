import argparse
import json
import sys
import os

# Add project root to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from . import transforms  # Register all passes
    from .utils.logger import logger as custom_logger
except ImportError:
    # Direct execution fallback
    import graph_optimizer.transforms as transforms
    from graph_optimizer.utils.logger import logger as custom_logger

# Prevent unused import warning
_ = transforms


def main():
    parser = argparse.ArgumentParser(
        description="Graph Optimizer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python -m graph_optimizer.main --config config.json

  # Override config with command line args
  python -m graph_optimizer.main --config config.json --input input.pb --output output.pb

  # Enable debug mode and set optimization level
  python -m graph_optimizer.main --config config.json --debug --level 2

  # Specify output nodes to protect from pruning
  python -m graph_optimizer.main --config config.json --output-nodes output1,output2

Config file format (JSON):
  {
    "input_graph": "path/to/input.pb",
    "output_graph": "path/to/output.pb",
    "level": 2,
    "debug": true,
    "output_nodes": ["output1", "output2"],
    "protected_nodes": ["important_node"],
    "passes": ["pass1", "pass2"],
    "add_passes": ["extra_pass"],
    "remove_passes": ["unwanted_pass"],
    "log_file": "optimization.log"
  }
        """
    )
    parser.add_argument(
        "--config", help="Path to JSON configuration file"
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
    parser.add_argument(
        "--output-nodes",
        help="Comma-separated list of output node names (protected from pruning)",
    )
    parser.add_argument(
        "--protected-nodes",
        help="Comma-separated list of additional nodes to protect from pruning",
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file",
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

    # Parse command line list arguments
    output_nodes = None
    if args.output_nodes:
        output_nodes = [n.strip() for n in args.output_nodes.split(",") if n.strip()]
    
    protected_nodes = None
    if args.protected_nodes:
        protected_nodes = [n.strip() for n in args.protected_nodes.split(",") if n.strip()]

    try:
        try:
            from .runner import OptimizationPipeline
        except ImportError:
            from graph_optimizer.runner import OptimizationPipeline

        pipeline = OptimizationPipeline(
            input_graph=args.input,          # Override from command line
            output_graph=args.output,        # Override from command line
            level=args.level,                # Optimization level (1 or 2)
            debug=args.debug,                # Enable debug mode
            log_file=args.log_file,          # Log file path
            output_nodes=output_nodes,       # Output nodes to protect
            protected_nodes=protected_nodes, # Additional nodes to protect
            config=config,                   # Config dict from JSON file
            # Command line args take precedence over config file
        )
        pipeline.run()
    except Exception as e:
        custom_logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
