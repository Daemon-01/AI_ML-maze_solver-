"""
Metric calculation functions for RL evaluation.

These metrics are designed to work with both sparse and dense (shaped) rewards.
"""

import numpy as np
from scipy import stats


def calculate_success_rate(successes):
    """
    Calculate success rate from list of success indicators.
    
    Args:
        successes: List of boolean success values
        
    Returns:
        Success rate as a float between 0 and 1
    """
    if not successes or len(successes) == 0:
        return 0.0
    return sum(successes) / len(successes)


def calculate_average_episode_length(episode_lengths):
    """
    Calculate average episode length.
    
    Args:
        episode_lengths: List of episode lengths
        
    Returns:
        Average episode length
    """
    if not episode_lengths or len(episode_lengths) == 0:
        return 0.0
    return float(np.mean(episode_lengths))


def calculate_std_episode_length(episode_lengths):
    """Calculate standard deviation of episode lengths."""
    if not episode_lengths or len(episode_lengths) == 0:
        return 0.0
    return float(np.std(episode_lengths))


def calculate_average_reward(episode_rewards):
    """
    Calculate average episode reward.
    
    NOTE: With reward shaping, expect positive values (5-20).
          With sparse rewards, expect negative values (-50 to 0).
    
    Args:
        episode_rewards: List of total episode rewards
        
    Returns:
        Average reward
    """
    if not episode_rewards or len(episode_rewards) == 0:
        return 0.0
    return float(np.mean(episode_rewards))


def calculate_performance_degradation(static_success_rate, dynamic_success_rate):
    """
    Calculate performance degradation from static to dynamic.
    
    Args:
        static_success_rate: Success rate in static environment (0-100)
        dynamic_success_rate: Success rate in dynamic environment (0-100)
        
    Returns:
        Performance degradation as a percentage
    """
    if static_success_rate == 0:
        return 0.0
    
    degradation = ((static_success_rate - dynamic_success_rate) / static_success_rate) * 100
    return float(degradation)


def calculate_confidence_interval(values, confidence=0.95):
    """
    Calculate confidence interval for a list of values.
    
    Args:
        values: List of numerical values
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        mean = np.mean(values) if values else 0.0
        return float(mean), float(mean), float(mean)
    
    n = len(values)
    mean = np.mean(values)
    sem = stats.sem(values)
    
    # Calculate the critical value
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin = t_critical * sem
    
    return float(mean), float(mean - margin), float(mean + margin)


def calculate_episode_efficiency(episode_lengths, optimal_length=None):
    """
    Calculate how efficient episodes are compared to optimal.
    
    Args:
        episode_lengths: List of episode lengths
        optimal_length: Optimal/shortest possible path length
                       If None, uses minimum observed length
        
    Returns:
        Average efficiency ratio (1.0 = optimal, >1.0 = suboptimal)
    """
    if not episode_lengths or len(episode_lengths) == 0:
        return 0.0
    
    if optimal_length is None:
        optimal_length = min(episode_lengths)
    
    if optimal_length == 0:
        return 0.0
    
    efficiencies = [optimal_length / length for length in episode_lengths]
    return float(np.mean(efficiencies))


def calculate_learning_progress(rewards_over_time, window=100):
    """
    Calculate if agent is improving over time.
    
    Args:
        rewards_over_time: List of rewards in temporal order
        window: Window size for moving average
        
    Returns:
        Dict with early_avg, late_avg, improvement
    """
    if not rewards_over_time or len(rewards_over_time) < window * 2:
        return {
            'early_avg': 0.0,
            'late_avg': 0.0,
            'improvement': 0.0,
            'is_learning': False
        }
    
    early_avg = float(np.mean(rewards_over_time[:window]))
    late_avg = float(np.mean(rewards_over_time[-window:]))
    improvement = late_avg - early_avg
    
    return {
        'early_avg': early_avg,
        'late_avg': late_avg,
        'improvement': improvement,
        'is_learning': improvement > 0
    }


def detect_reward_type(episode_rewards):
    """
    Detect if rewards are sparse or shaped.
    
    Args:
        episode_rewards: List of episode rewards
        
    Returns:
        'sparse', 'shaped', or 'unknown'
    """
    if not episode_rewards or len(episode_rewards) == 0:
        return 'unknown'
    
    mean_reward = np.mean(episode_rewards)
    
    # Shaped rewards are typically positive (5-20 range)
    # Sparse rewards are typically negative (-100 to 0 range)
    if mean_reward > 2.0:
        return 'shaped'
    elif mean_reward < -5.0:
        return 'sparse'
    else:
        return 'unknown'


def get_evaluation_summary(episode_data):
    """
    Generate comprehensive evaluation summary.
    
    Args:
        episode_data: Dict with keys: 'successes', 'lengths', 'rewards'
        
    Returns:
        Dict with all calculated metrics
    """
    successes = episode_data.get('successes', [])
    lengths = episode_data.get('lengths', [])
    rewards = episode_data.get('rewards', [])
    
    # Basic metrics
    success_rate = calculate_success_rate(successes) * 100  # Convert to percentage
    avg_length = calculate_average_episode_length(lengths)
    std_length = calculate_std_episode_length(lengths)
    avg_reward = calculate_average_reward(rewards)
    
    # Confidence intervals
    success_ci = calculate_confidence_interval([1 if s else 0 for s in successes])
    length_ci = calculate_confidence_interval(lengths) if lengths else (0, 0, 0)
    reward_ci = calculate_confidence_interval(rewards) if rewards else (0, 0, 0)
    
    # Additional metrics
    reward_type = detect_reward_type(rewards)
    
    return {
        # Success metrics
        'success_rate': success_rate,
        'success_rate_ci': {
            'mean': success_ci[0] * 100,
            'lower': success_ci[1] * 100,
            'upper': success_ci[2] * 100
        },
        
        # Length metrics
        'avg_length': avg_length,
        'std_length': std_length,
        'length_ci': {
            'mean': length_ci[0],
            'lower': length_ci[1],
            'upper': length_ci[2]
        },
        
        # Reward metrics
        'avg_reward': avg_reward,
        'reward_ci': {
            'mean': reward_ci[0],
            'lower': reward_ci[1],
            'upper': reward_ci[2]
        },
        'reward_type': reward_type,
        
        # Diagnostics
        'n_episodes': len(successes),
        'any_success': any(successes) if successes else False
    }
