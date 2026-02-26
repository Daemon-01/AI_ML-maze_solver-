#!/bin/bash
# Generate all visualizations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Generating plots..."

python -m src.visualization.plot_training
python -m src.visualization.plot_results

echo "All plots generated!"
