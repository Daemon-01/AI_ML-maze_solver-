"""
Evaluate trained PPO model with support for both sparse and shaped rewards.

CRITICAL: This version expects reward shaping to be used during training!
If your model was trained WITHOUT reward shaping, it will perform poorly.
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

# Import the reward shaping wrapper
try:
    from environments.reward_shaping_wrapper import DenseRewardWrapper
    HAS_REWARD_SHAPING = True
except ImportError:
    print("⚠ WARNING: reward_shaping_wrapper not found!")
    print("   Model will be evaluated WITHOUT reward shaping")
    print("   If model was trained WITH shaping, results will be incorrect")
    HAS_REWARD_SHAPING = False

from environments.env_factory import create_static_env


def _get_agent_position(env):
    """Try to read the agent position from the unwrapped MiniGrid env."""
    try:
        agent_pos = env.unwrapped.agent_pos
        return [int(agent_pos[0]), int(agent_pos[1])]
    except Exception:
        return None


def create_eval_env(render_mode=None, use_reward_shaping=True):
    """
    Create environment for evaluation.
    
    Args:
        render_mode: 'human' for visualization, None for no rendering
        use_reward_shaping: Whether to apply dense reward wrapper
                           MUST match training configuration!
    
    Returns:
        Wrapped environment
    """
    env = gym.make('MiniGrid-DoorKey-8x8-v0', render_mode=render_mode)
    
    # Always need ImgObsWrapper for CNN policy
    from minigrid.wrappers import ImgObsWrapper
    env = ImgObsWrapper(env)
    
    # Apply reward shaping if requested AND available
    if use_reward_shaping and HAS_REWARD_SHAPING:
        env = DenseRewardWrapper(env)
        print("✓ Using dense reward shaping for evaluation")
    elif use_reward_shaping and not HAS_REWARD_SHAPING:
        print("⚠ Reward shaping requested but wrapper not available")
    
    return env


def evaluate_model(
    model_path, 
    n_episodes=10, 
    render=True, 
    mistake_log_path=None,
    use_reward_shaping=True
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to run
        render: Whether to render the environment
        mistake_log_path: Where to log failed episodes
        use_reward_shaping: Use dense rewards (should match training!)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 80)
    print(f"EVALUATING MODEL: {model_path}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Create environment
    print("\nCreating evaluation environment...")
    env = create_eval_env(
        render_mode='human' if render else None,
        use_reward_shaping=use_reward_shaping
    )
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    failed_episodes = []
    
    print(f"\nRunning {n_episodes} evaluation episodes...")
    print("-" * 80)
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        step_rewards = []  # Track rewards per step for debugging
        
        while not (done or truncated) and episode_length < 500:
            # Model predicts action (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            step_rewards.append(float(reward))
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Success if reached goal before max steps
        success = done and episode_length < 500
        if success:
            success_count += 1
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILED"
            failed_episodes.append({
                'episode': episode + 1,
                'reward': float(episode_reward),
                'steps': int(episode_length),
                'done': bool(done),
                'truncated': bool(truncated),
                'final_agent_pos': _get_agent_position(env),
                'reward_breakdown': {
                    'min': float(min(step_rewards)) if step_rewards else 0,
                    'max': float(max(step_rewards)) if step_rewards else 0,
                    'mean': float(np.mean(step_rewards)) if step_rewards else 0
                }
            })
        
        print(f"Episode {episode+1:2d}: {status} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Steps: {episode_length:3d}")
    
    env.close()
    
    # Calculate statistics
    success_rate = (success_count / n_episodes) * 100
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    # Detect reward type
    if mean_reward > 2.0:
        reward_type = "SHAPED (dense)"
    elif mean_reward < -5.0:
        reward_type = "SPARSE (default)"
    else:
        reward_type = "UNKNOWN"
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Success Rate:    {success_rate:.1f}% ({success_count}/{n_episodes})")
    print(f"Average Reward:  {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Steps:   {mean_length:.1f} ± {std_length:.1f}")
    print(f"Reward Type:     {reward_type}")
    print("=" * 80)
    
    # Diagnostic warnings
    if success_rate < 10:
        print("\n⚠️ VERY LOW SUCCESS RATE - POSSIBLE ISSUES:")
        if reward_type == "SPARSE (default)":
            print("  → Model trained with SPARSE rewards (no reward shaping)")
            print("  → This makes learning nearly impossible!")
            print("  → SOLUTION: Retrain with reward shaping wrapper")
        else:
            print("  → Check if reward shaping was used during training")
            print("  → Ensure evaluation uses same wrapper as training")
            print("  → May need more training time")
    elif success_rate < 40:
        print("\n⚠️ MODERATE SUCCESS RATE:")
        print("  → Model is learning but needs more training")
        print("  → Try increasing to 1M timesteps")
    else:
        print("\n✓ GOOD PERFORMANCE!")
    
    # Save mistake log
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
            'reward_type': reward_type,
            'used_reward_shaping': use_reward_shaping and HAS_REWARD_SHAPING,
            'failed_episode_count': len(failed_episodes),
            'failed_episodes': failed_episodes
        }
        with mistake_log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(f"\n[LOG] Evaluation log appended to: {mistake_log_path}")
    
    return {
        'success_rate': success_rate,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'reward_type': reward_type,
        'successes': [r > 0 and l < 500 for r, l in zip(episode_rewards, episode_lengths)]
    }


def main():
    """Main evaluation function."""
    import argparse
    
    default_model = PROJECT_ROOT / 'models' / 'static_baseline' / 'seed_42' / 'best_model'

    parser = argparse.ArgumentParser(description='Evaluate trained PPO model')
    parser.add_argument(
        '--model', type=str, 
        default=str(default_model),
        help='Path to model'
    )
    parser.add_argument(
        '--episodes', type=int, default=10, 
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--no-render', action='store_true',
        help='Disable rendering'
    )
    parser.add_argument(
        '--no-reward-shaping', action='store_true',
        help='Disable reward shaping (use if model trained without it)'
    )
    parser.add_argument(
        '--mistake-log',
        type=str,
        default=str(PROJECT_ROOT / 'logs' / 'mistakes' / 'static_eval_mistakes.jsonl'),
        help='Where to append failed-episode logs (JSONL)'
    )
    
    args = parser.parse_args()
    
    # Resolve model path
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    model_zip_path = model_path if model_path.suffix == '.zip' else model_path.with_suffix('.zip')
    model_load_path = model_path.with_suffix('') if model_path.suffix == '.zip' else model_path

    # Check if model exists
    if not model_zip_path.exists():
        print(f"❌ ERROR: Model not found at {model_zip_path}")
        print("\nAvailable models:")
        models_dir = PROJECT_ROOT / 'models'
        if models_dir.exists():
            for condition_dir in models_dir.iterdir():
                if condition_dir.is_dir():
                    print(f"\n  {condition_dir.name}/")
                    for seed_dir in condition_dir.iterdir():
                        if seed_dir.is_dir():
                            for model_file in seed_dir.glob('*.zip'):
                                print(f"    - {model_file.name}")
        print("\nTrain a model first:")
        print("  python train_static_FIXED.py")
        return
    
    print("\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80)
    print(f"Model: {model_load_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {not args.no_render}")
    print(f"Reward Shaping: {not args.no_reward_shaping}")
    print("=" * 80 + "\n")
    
    # Evaluate
    results = evaluate_model(
        str(model_load_path),
        n_episodes=args.episodes,
        render=not args.no_render,
        mistake_log_path=args.mistake_log,
        use_reward_shaping=not args.no_reward_shaping
    )
    
    if results and results['success_rate'] < 10:
        print("\n" + "=" * 80)
        print("❌ MODEL PERFORMANCE IS POOR")
        print("=" * 80)
        print("\nMost likely cause: Model trained WITHOUT reward shaping")
        print("\nTo fix:")
        print("  1. Copy reward_shaping_wrapper.py to src/environments/")
        print("  2. Run: python train_static_FIXED.py --timesteps 500000")
        print("  3. Wait 30-60 minutes for training")
        print("  4. Re-evaluate with this script")
        print("\nExpected improvement:")
        print("  Current:  ~0% success rate")
        print("  After fix: 50-80% success rate")


if __name__ == '__main__':
    main()
