"""
Compare static vs dynamic training conditions with proper reward handling.
"""

import os
import json
import numpy as np
from pathlib import Path
import sys

# Add parent dir to path for imports
SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(SRC_ROOT))

from evaluation.evaluate_model import evaluate_model
from evaluation.metrics import (
    calculate_performance_degradation,
    calculate_confidence_interval,
    get_evaluation_summary
)


def compare_all_conditions(
    model_base_dir='models',
    seeds=[42, 123, 456],
    n_eval_episodes=20,
    use_reward_shaping=True
):
    """
    Compare all training conditions with proper statistical analysis.
    
    Args:
        model_base_dir: Base directory containing models
        seeds: List of random seeds used in training
        n_eval_episodes: Episodes per evaluation
        use_reward_shaping: Whether to use reward shaping in evaluation
        
    Returns:
        Dictionary with comparison results
    """
    conditions = ['static_baseline', 'dynamic_low', 'dynamic_medium']
    results = {}
    
    print("=" * 80)
    print("COMPARING ALL CONDITIONS")
    print("=" * 80)
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {n_eval_episodes}")
    print(f"Reward shaping: {use_reward_shaping}")
    print("=" * 80 + "\n")
    
    for condition in conditions:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {condition}")
        print(f"{'='*80}")
        
        condition_results = []
        
        for seed in seeds:
            model_path = os.path.join(
                model_base_dir, condition, f'seed_{seed}', 'best_model'
            )
            
            # Check if model exists
            if not os.path.exists(model_path + '.zip'):
                print(f"  ⚠ Skipping seed_{seed}: Model not found at {model_path}.zip")
                continue
            
            print(f"\n  Evaluating seed_{seed}...")
            
            try:
                seed_results = evaluate_model(
                    model_path,
                    n_episodes=n_eval_episodes,
                    render=False,  # No rendering for batch evaluation
                    use_reward_shaping=use_reward_shaping
                )
                
                if seed_results:
                    condition_results.append(seed_results)
                    print(f"    ✓ Success rate: {seed_results['success_rate']:.1f}%")
                else:
                    print(f"    ✗ Evaluation failed")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        # Aggregate results across seeds
        if condition_results:
            success_rates = [r['success_rate'] for r in condition_results]
            rewards = [r['mean_reward'] for r in condition_results]
            lengths = [r['mean_length'] for r in condition_results]
            
            # Calculate confidence intervals
            success_ci = calculate_confidence_interval(success_rates)
            reward_ci = calculate_confidence_interval(rewards)
            length_ci = calculate_confidence_interval(lengths)
            
            results[condition] = {
                # Success metrics
                'mean_success_rate': float(np.mean(success_rates)),
                'std_success_rate': float(np.std(success_rates)),
                'success_rate_ci': {
                    'mean': success_ci[0],
                    'lower': success_ci[1],
                    'upper': success_ci[2]
                },
                
                # Reward metrics
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'reward_ci': {
                    'mean': reward_ci[0],
                    'lower': reward_ci[1],
                    'upper': reward_ci[2]
                },
                
                # Length metrics
                'mean_length': float(np.mean(lengths)),
                'std_length': float(np.std(lengths)),
                'length_ci': {
                    'mean': length_ci[0],
                    'lower': length_ci[1],
                    'upper': length_ci[2]
                },
                
                # Per-seed details
                'n_seeds_evaluated': len(condition_results),
                'per_seed_results': condition_results
            }
            
            print(f"\n  {condition} Summary:")
            print(f"    Seeds evaluated: {len(condition_results)}/{len(seeds)}")
            print(f"    Success rate: {results[condition]['mean_success_rate']:.1f}% "
                  f"± {results[condition]['std_success_rate']:.1f}%")
            print(f"    Avg reward: {results[condition]['mean_reward']:.2f} "
                  f"± {results[condition]['std_reward']:.2f}")
        else:
            print(f"\n  ⚠ No results for {condition}")
            results[condition] = None
    
    # Calculate degradation metrics
    if results.get('static_baseline') and results.get('dynamic_low'):
        static_success = results['static_baseline']['mean_success_rate']
        dynamic_low_success = results['dynamic_low']['mean_success_rate']
        
        degradation_low = calculate_performance_degradation(
            static_success, dynamic_low_success
        )
        results['degradation_low'] = degradation_low
    
    if results.get('static_baseline') and results.get('dynamic_medium'):
        static_success = results['static_baseline']['mean_success_rate']
        dynamic_medium_success = results['dynamic_medium']['mean_success_rate']
        
        degradation_medium = calculate_performance_degradation(
            static_success, dynamic_medium_success
        )
        results['degradation_medium'] = degradation_medium
    
    return results


def print_comparison_table(results):
    """Print results in a nice table format."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    
    header = f"{'Condition':<20} {'Success Rate':>15} {'Avg Reward':>15} {'Avg Steps':>15}"
    print(header)
    print("-" * 80)
    
    for condition in ['static_baseline', 'dynamic_low', 'dynamic_medium']:
        if results.get(condition):
            r = results[condition]
            row = (f"{condition:<20} "
                   f"{r['mean_success_rate']:>8.1f}% ± {r['std_success_rate']:>4.1f} "
                   f"{r['mean_reward']:>8.2f} ± {r['std_reward']:>4.2f} "
                   f"{r['mean_length']:>8.1f} ± {r['std_length']:>4.1f}")
            print(row)
        else:
            print(f"{condition:<20} {'NO DATA':>15}")
    
    print("-" * 80)
    
    # Print degradation
    if 'degradation_low' in results:
        print(f"\nPerformance Degradation (2% change rate): {results['degradation_low']:.1f}%")
    if 'degradation_medium' in results:
        print(f"Performance Degradation (5% change rate): {results['degradation_medium']:.1f}%")
    
    print("=" * 80)


def save_comparison_results(results, output_path='results/experimental_results.json'):
    """Save comparison results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    print(f"\n✓ Results saved to: {output_path}")


def save_latex_table(results, output_path='results/results_table.tex'):
    """Save results as LaTeX table for paper."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison: Static vs Dynamic Environments}")
    latex.append("\\label{tab:results}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\hline")
    latex.append("Condition & Success Rate (\\%) & Avg Reward & Avg Steps \\\\")
    latex.append("\\hline")
    
    for condition, label in [
        ('static_baseline', 'Static Baseline'),
        ('dynamic_low', 'Dynamic (2\\% change)'),
        ('dynamic_medium', 'Dynamic (5\\% change)')
    ]:
        if results.get(condition):
            r = results[condition]
            latex.append(
                f"{label} & "
                f"{r['mean_success_rate']:.1f} $\\pm$ {r['std_success_rate']:.1f} & "
                f"{r['mean_reward']:.2f} $\\pm$ {r['std_reward']:.2f} & "
                f"{r['mean_length']:.1f} $\\pm$ {r['std_length']:.1f} \\\\"
            )
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"✓ LaTeX table saved to: {output_path}")


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare all training conditions')
    parser.add_argument(
        '--model-dir', type=str, default='models',
        help='Base directory containing models'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=[42, 123, 456],
        help='Seeds to evaluate'
    )
    parser.add_argument(
        '--episodes', type=int, default=20,
        help='Episodes per evaluation'
    )
    parser.add_argument(
        '--no-reward-shaping', action='store_true',
        help='Disable reward shaping'
    )
    parser.add_argument(
        '--output', type=str, default='results/experimental_results.json',
        help='Output JSON path'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_all_conditions(
        model_base_dir=args.model_dir,
        seeds=args.seeds,
        n_eval_episodes=args.episodes,
        use_reward_shaping=not args.no_reward_shaping
    )
    
    # Print table
    print_comparison_table(results)
    
    # Save results
    save_comparison_results(results, args.output)
    save_latex_table(results, 'results/results_table.tex')
    
    # Warnings
    static_baseline = results.get('static_baseline')
    if static_baseline and static_baseline['mean_success_rate'] < 20:
        print("\n" + "=" * 80)
        print("⚠️ WARNING: BASELINE PERFORMANCE IS VERY LOW")
        print("=" * 80)
        print("\nThis suggests models were trained WITHOUT reward shaping!")
        print("\nTo fix:")
        print("  1. Add reward_shaping_wrapper.py to src/environments/")
        print("  2. Retrain all models: bash scripts/train_all_with_shaping.sh")
        print("  3. Re-run this comparison")
        print("\nExpected improvement: 0-10% → 50-80% success rate")


if __name__ == '__main__':
    main()
