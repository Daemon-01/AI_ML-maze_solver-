#!/bin/bash
# Initial project setup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Setting up PPO Maze Navigation project..."

# Create virtual environment
python -m venv venv
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  source venv/Scripts/activate
fi

# Install dependencies
pip install -r requirements.txt

echo "Setup complete!"
