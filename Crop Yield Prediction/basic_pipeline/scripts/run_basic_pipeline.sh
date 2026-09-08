#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-$ROOT_DIR/archive/yield_df.csv}"

cd "$ROOT_DIR"

echo "Running dataset summary..."
PYTHONPATH=src "$PYTHON_BIN" -m yield_prediction.cli --data "$DATASET_PATH" summary

echo
echo "Checking benchmark dependencies..."
if PYTHONPATH=src "$PYTHON_BIN" -c "import pandas, sklearn" >/dev/null 2>&1; then
  echo "Running benchmark..."
  PYTHONPATH=src "$PYTHON_BIN" -m yield_prediction.cli --data "$DATASET_PATH" benchmark
else
  echo "Skipping benchmark. Install optional dependencies with:"
  echo "  pip install -e '.[ml]'"
fi
