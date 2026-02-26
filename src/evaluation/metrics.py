"""
Metric calculation functions.
"""

import numpy as np


def calculate_success_rate(successes):
    """
    Calculate success rate from list of success indicators.
    
    Args:
        successes: List of boolean success values
        
    Returns:
        Success rate as a float between 0 and 1
    """
    if not successes:
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
    if not episode_lengths:
        return 0.0
    return np.mean(episode_lengths)


def calculate_performance_degradation(static_success_rate, dynamic_success_rate):
    """
    Calculate performance degradation from static to dynamic.
    
    Args:
        static_success_rate: Success rate in static environment
        dynamic_success_rate: Success rate in dynamic environment
        
    Returns:
        Performance degradation as a percentage
    """
    if static_success_rate == 0:
        return 0.0
    return ((static_success_rate - dynamic_success_rate) / static_success_rate) * 100


def calculate_confidence_interval(values, confidence=0.95):
    """
    Calculate confidence interval for a list of values.
    
    Args:
        values: List of numerical values
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    from scipy import stats
    
    n = len(values)
    mean = np.mean(values)
    sem = stats.sem(values)
    
    # Calculate the critical value
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin = t_critical * sem
    
    return mean, mean - margin, mean + margin
