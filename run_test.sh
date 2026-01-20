#!/bin/bash
# Regression test script for Graph Optimizer
set -e # Exit on any error

# Get the script's directory and the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:..:$PYTHONPATH"

# Find the best python interpreter
if [ -f "$PROJECT_ROOT/.venv/bin/python3" ]; then
    PYTHON_EXE="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON_EXE="python3"
fi

echo "========================================"
echo "Starting Graph Optimizer Regression Test"
echo "Using: $PYTHON_EXE"
echo "========================================"

# 1. Run Framework Tests
echo -e "\n[1/3] Running Framework Tests..."
$PYTHON_EXE -m unittest discover "$SCRIPT_DIR/tests/framework" -v
if [ $? -ne 0 ]; then
    echo "ERROR: Framework tests failed!"
    exit 1
fi

# 2. Run Pass Tests
echo -e "\n[2/3] Running Pass Tests..."
$PYTHON_EXE -m unittest discover "$SCRIPT_DIR/tests/transforms" -v
if [ $? -ne 0 ]; then
    echo "ERROR: Pass tests failed!"
    exit 1
fi

# 3. Run Demo
echo -e "\n[3/3] Running Demo..."
$PYTHON_EXE "$SCRIPT_DIR/demos/run_demo.py"
if [ $? -ne 0 ]; then
    echo "ERROR: Demo execution failed!"
    exit 1
fi

echo -e "\n========================================"
echo "SUCCESS: All tests and demo passed!"
echo "========================================"
