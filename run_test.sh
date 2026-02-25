#!/bin/bash
# Regression test script for Graph Optimizer
set -e # Exit on any error

# Get the script's directory and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:..:$PYTHONPATH"

# Use the local .venv if it exists, otherwise fall back to system python3
if [ -f "$PROJECT_ROOT/.venv/bin/python3" ]; then
    PYTHON_EXE="${PYTHON_EXE:-$PROJECT_ROOT/.venv/bin/python3}"
else
    PYTHON_EXE="${PYTHON_EXE:-python3}"
fi

echo "========================================"
echo "Starting Graph Optimizer Regression Test"
echo "Using: $PYTHON_EXE"
echo "========================================"

# 1. Run All Tests (TF & Torch)
echo -e "\n[1/3] Running Pytest Suite..."
$PYTHON_EXE -m pytest "$SCRIPT_DIR/tests/" -v
if [ $? -ne 0 ]; then
    echo "ERROR: Pytest suite failed!"
    exit 1
fi

# 2. Run TF Demo
echo -e "\n[2/3] Running TensorFlow Demo..."
$PYTHON_EXE "$SCRIPT_DIR/demos/run_demo.py"
if [ $? -ne 0 ]; then
    echo "ERROR: TensorFlow Demo execution failed!"
    exit 1
fi

# 3. Run Torch Demo
echo -e "\n[3/3] Running PyTorch FX Demo..."
$PYTHON_EXE "$SCRIPT_DIR/demos/run_demo_torch.py"
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch FX Demo execution failed!"
    exit 1
fi

echo -e "\n========================================"
echo "SUCCESS: All tests and demo passed!"
echo "========================================"
