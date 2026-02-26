"""
Training curve plots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(log_dir, tag='rollout/ep_rew_mean'):
    """
    Load training data from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard events
        tag: Tag to extract (default: episode reward mean)
        
    Returns:
        Tuple of (steps, values)
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return np.array(steps), np.array(values)


def plot_training_curve(log_dir, output_path, title='Training Curve'):
    """
    Plot training curve from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard events
        output_path: Path to save the plot
        title: Plot title
    """
    steps, rewards = load_tensorboard_data(log_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, label='Episode Reward')
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Episode Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_all_conditions_comparison(log_dirs, labels, output_path):
    """
    Plot training curves for all conditions on one figure.
    
    Args:
        log_dirs: List of log directories
        labels: List of condition labels
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['blue', 'orange', 'green']
    
    for log_dir, label, color in zip(log_dirs, labels, colors):
        steps, rewards = load_tensorboard_data(log_dir)
        plt.plot(steps, rewards, label=label, color=color, alpha=0.8)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Episode Reward')
    plt.title('Training Comparison: Static vs Dynamic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    pass
