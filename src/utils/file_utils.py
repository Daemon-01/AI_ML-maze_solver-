"""File I/O helpers."""

import os
import json


def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    """Save data to JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
