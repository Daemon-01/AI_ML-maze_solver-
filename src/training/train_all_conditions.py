"""
Run all training experiments.
"""

import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_static import train_static_baseline
from training.train_dynamic import train_dynamic

SEEDS = [42, 123, 456]
ENV_NAME = 'MiniGrid-FourRooms-v0'
TOTAL_TIMESTEPS = 1_000_000


def run_all_experiments():
    """Run all training conditions with multiple seeds."""
    
    # Static baseline
    print("=" * 50)
    print("Training Static Baseline")
    print("=" * 50)
    for seed in SEEDS:
        print(f"\nTraining with seed {seed}...")
        # Note: train_static_baseline uses hardcoded ENV_NAME from env_factory default
        train_static_baseline(
            seed=seed,
            total_timesteps=TOTAL_TIMESTEPS,
            log_dir='logs/static_baseline',
            model_dir='models/static_baseline'
        )
    
    # Dynamic low (2% change rate)
    print("=" * 50)
    print("Training Dynamic Low (2%)")
    print("=" * 50)
    for seed in SEEDS:
        print(f"\nTraining with seed {seed}...")
        train_dynamic(
            ENV_NAME,
            change_probability=0.02,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=seed,
            log_dir='logs/dynamic_low',
            model_dir='models/dynamic_low'
        )
    
    # Dynamic medium (5% change rate)
    print("=" * 50)
    print("Training Dynamic Medium (5%)")
    print("=" * 50)
    for seed in SEEDS:
        print(f"\nTraining with seed {seed}...")
        train_dynamic(
            ENV_NAME,
            change_probability=0.05,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=seed,
            log_dir='logs/dynamic_medium',
            model_dir='models/dynamic_medium'
        )
    
    print("\nAll experiments completed!")


if __name__ == '__main__':
    run_all_experiments()
