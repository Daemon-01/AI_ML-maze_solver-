#!/bin/bash
# Run all training experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running all experiments..."

python src/training/train_all_conditions.py

echo "All experiments complete!"
