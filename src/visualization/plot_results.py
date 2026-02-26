"""
Results comparison plots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_success_rate_bars(results, output_path):
    """
    Plot success rate comparison as bar chart.
    
    Args:
        results: Dictionary with condition results
        output_path: Path to save the plot
    """
    conditions = list(results.keys())
    success_rates = [results[c]['mean_success_rate'] * 100 for c in conditions]
    errors = [results[c]['std_success_rate'] * 100 for c in conditions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conditions, success_rates, yerr=errors, capsize=5)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f'{rate:.1f}%',
            ha='center',
            va='bottom'
        )
    
    plt.xlabel('Training Condition')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison')
    plt.ylim(0, 110)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_episode_length_boxplot(results, output_path):
    """
    Plot episode length comparison as boxplot.
    
    Args:
        results: Dictionary with condition results
        output_path: Path to save the plot
    """
    conditions = list(results.keys())
    episode_lengths = [
        [r['mean_length'] for r in results[c]['per_seed_results']]
        for c in conditions
    ]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(episode_lengths, labels=conditions)
    plt.xlabel('Training Condition')
    plt.ylabel('Average Episode Length')
    plt.title('Episode Length Distribution Comparison')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_degradation_analysis(static_rate, dynamic_rates, labels, output_path):
    """
    Plot performance degradation analysis.
    
    Args:
        static_rate: Success rate in static environment
        dynamic_rates: List of success rates in dynamic environments
        labels: Labels for dynamic conditions
        output_path: Path to save the plot
    """
    degradations = [
        ((static_rate - dr) / static_rate) * 100
        for dr in dynamic_rates
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, degradations, color=['orange', 'red'])
    
    plt.xlabel('Dynamic Condition')
    plt.ylabel('Performance Degradation (%)')
    plt.title('Performance Degradation from Static Baseline')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    pass
