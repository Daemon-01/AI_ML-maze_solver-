"""
Environment testing utilities.
"""

from .env_factory import create_environment, create_static_env, create_dynamic_env


def test_environment(env, num_episodes=5, max_steps=100):
    """
    Test an environment with random actions.
    
    Args:
        env: The environment to test
        num_episodes: Number of test episodes
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with test results
    """
    results = {
        'episodes_completed': 0,
        'total_rewards': [],
        'episode_lengths': []
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
                
        results['episodes_completed'] += 1
        results['total_rewards'].append(episode_reward)
        results['episode_lengths'].append(step + 1)
        
    return results


if __name__ == '__main__':
    # Test static environment
    print("Testing static environment...")
    # env = create_static_env('MazeEnv-v0')
    # results = test_environment(env)
    # print(f"Results: {results}")
