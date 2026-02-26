"""
Compare static vs dynamic training conditions.
"""

import os
import json
import numpy as np

from .evaluate_model import evaluate_model


def compare_all_conditions(env, model_base_dir='models', seeds=[42, 123, 456]):
    """
    Compare all training conditions.
    
    Args:
        env: Environment to evaluate in
        model_base_dir: Base directory containing models
        seeds: List of random seeds used in training
        
    Returns:
        Dictionary with comparison results
    """
    conditions = ['static_baseline', 'dynamic_low', 'dynamic_medium']
    results = {}
    
    for condition in conditions:
        condition_results = []
        
        for seed in seeds:
            model_path = os.path.join(
                model_base_dir, condition, f'seed_{seed}', 'best_model.zip'
            )
            
            if os.path.exists(model_path):
                seed_results = evaluate_model(model_path, env)
                condition_results.append(seed_results)
        
        if condition_results:
            results[condition] = {
                'mean_success_rate': np.mean([r['success_rate'] for r in condition_results]),
                'std_success_rate': np.std([r['success_rate'] for r in condition_results]),
                'mean_reward': np.mean([r['mean_reward'] for r in condition_results]),
                'std_reward': np.std([r['mean_reward'] for r in condition_results]),
                'mean_length': np.mean([r['mean_length'] for r in condition_results]),
                'std_length': np.std([r['mean_length'] for r in condition_results]),
                'per_seed_results': condition_results
            }
    
    return results


def save_comparison_results(results, output_path='results/experimental_results.json'):
    """Save comparison results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_types(results), f, indent=2)


if __name__ == '__main__':
    pass
