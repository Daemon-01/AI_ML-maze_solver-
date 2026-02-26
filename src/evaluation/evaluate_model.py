"""
Evaluate trained PPO model and visualize performance.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(SRC_ROOT))
from environments.env_factory import create_static_env


def _get_agent_position(env):
    """Try to read the agent position from the unwrapped MiniGrid env."""
    try:
        agent_pos = env.unwrapped.agent_pos
        return [int(agent_pos[0]), int(agent_pos[1])]
    except Exception:
        return None


def evaluate_model(model_path, n_episodes=10, render=True, mistake_log_path=None):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to run
        render: Whether to render the environment
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print(f"Evaluating model: {model_path}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)
    
    # Create environment
    env = create_static_env(render_mode='human' if render else None)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    failed_episodes = []
    
    print(f"\nRunning {n_episodes} evaluation episodes...")
    print("-" * 60)
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated) and episode_length < 500:
            # Model predicts action (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Success if reached goal before max steps
        if done and episode_length < 500:
            success_count += 1
            status = "[SUCCESS]"
        else:
            status = "[FAILED]"
            failed_episodes.append(
                {
                    'episode': episode + 1,
                    'reward': float(episode_reward),
                    'steps': int(episode_length),
                    'done': bool(done),
                    'truncated': bool(truncated),
                    'final_agent_pos': _get_agent_position(env)
                }
            )
        
        print(f"Episode {episode+1:2d}: {status} | Reward: {episode_reward:6.1f} | Steps: {episode_length:3d}")
    
    env.close()
    
    # Calculate statistics
    success_rate = (success_count / n_episodes) * 100
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Success Rate:    {success_rate:.1f}% ({success_count}/{n_episodes})")
    print(f"Average Reward:  {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Average Steps:   {mean_length:.1f} +/- {std_length:.1f}")
    print("=" * 60)

    if mistake_log_path:
        mistake_log_path = Path(mistake_log_path)
        mistake_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'model_path': str(model_path),
            'episodes': int(n_episodes),
            'success_rate': float(success_rate),
            'mean_reward': float(mean_reward),
            'mean_length': float(mean_length),
            'failed_episode_count': len(failed_episodes),
            'failed_episodes': failed_episodes
        }
        with mistake_log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(f"[LOG] Mistake log appended to: {mistake_log_path}")
    
    return {
        'success_rate': success_rate,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def main():
    """Main evaluation function."""
    import argparse
    
    default_model = PROJECT_ROOT / 'models' / 'static_baseline' / 'seed_42' / 'best_model'

    parser = argparse.ArgumentParser(description='Evaluate trained PPO model')
    parser.add_argument('--model', type=str, 
                       default=str(default_model),
                       help='Path to model')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument(
        '--mistake-log',
        type=str,
        default=str(PROJECT_ROOT / 'logs' / 'mistakes' / 'static_eval_mistakes.jsonl'),
        help='Where to append failed-episode logs (JSONL)'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    model_zip_path = model_path if model_path.suffix == '.zip' else model_path.with_suffix('.zip')
    model_load_path = model_path.with_suffix('') if model_path.suffix == '.zip' else model_path

    # Check if model exists
    if not model_zip_path.exists():
        print(f"ERROR: Model not found at {model_zip_path}")
        print("\nPlease train a model first:")
        print("  python src/training/train_static.py")
        return
    
    # Evaluate
    results = evaluate_model(
        str(model_load_path),
        n_episodes=args.episodes,
        render=not args.no_render,
        mistake_log_path=args.mistake_log
    )


if __name__ == '__main__':
    main()
