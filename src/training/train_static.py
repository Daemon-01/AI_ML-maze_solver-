"""
Train PPO agent on static MiniGrid mazes.
This is the baseline for comparison with dynamic environments.
"""

import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import sys
import torch 

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.env_factory import create_static_env
from training.hyperparameters import get_ppo_params, get_ppo_params_tuned
from analysis.failure_analysis import analyze_mistake_log

print(torch.version.cuda)          # should show a CUDA version (e.g., '11.8')
print(torch.cuda.is_available())   # must be True

def train_static_baseline(
    seed=42,
    total_timesteps=500000,
    n_envs=4,
    model_dir='models/static_baseline',
    log_dir='logs/static_baseline',
    hyperparam_profile='tuned',
    env_max_steps=1000,
    mistake_log_path='logs/mistakes/static_eval_mistakes.jsonl'
):
    """
    Train PPO on static mazes.
    
    Args:
        seed: Random seed for reproducibility
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        model_dir: Where to save models
        log_dir: Where to save logs
        hyperparam_profile: "default" or "tuned" PPO hyperparameter preset
        env_max_steps: Maximum steps per episode for MiniGrid env
        mistake_log_path: Historical evaluation mistakes used for training adjustment
    """
    print("=" * 60)
    print(f"Training Static Baseline with seed {seed}")
    print("=" * 60)
    
    # Create directories
    seed_model_dir = f"{model_dir}/seed_{seed}"
    seed_log_dir = f"{log_dir}/seed_{seed}"
    os.makedirs(seed_model_dir, exist_ok=True)
    os.makedirs(seed_log_dir, exist_ok=True)
    
    if hyperparam_profile not in {'default', 'tuned'}:
        raise ValueError("hyperparam_profile must be 'default' or 'tuned'")

    ppo_params = get_ppo_params_tuned() if hyperparam_profile == 'tuned' else get_ppo_params()

    mistake_log = Path(mistake_log_path)
    timeout_rate = None
    if mistake_log.exists():
        summary = analyze_mistake_log(mistake_log)
        failed = summary['total_failed_episodes']
        timeout_failures = summary['timeout_failures']
        if failed > 0:
            timeout_rate = timeout_failures / failed
            # If most failures are timeouts, increase exploration and train longer.
            if timeout_rate >= 0.8:
                ppo_params['ent_coef'] = max(float(ppo_params.get('ent_coef', 0.0)), 0.02)
                ppo_params['gamma'] = max(float(ppo_params.get('gamma', 0.99)), 0.997)
                total_timesteps = max(int(total_timesteps), 500000)
                env_max_steps = max(int(env_max_steps), 1000)

    # Create vectorized training environment (4 parallel environments)
    print(f"\nCreating {n_envs} parallel training environments...")
    vec_env = make_vec_env(
        lambda: create_static_env(max_steps=env_max_steps),
        n_envs=n_envs,
        seed=seed
    )
    
    # Create evaluation environment (single env for testing)
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: create_static_env(max_steps=env_max_steps),
        n_envs=1,
        seed=seed + 1000  # Different seed for eval
    )
    
    # Set up evaluation callback (saves best model automatically)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=seed_model_dir,
        log_path=seed_log_dir,
        eval_freq=5000,  # Evaluate every 5000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    
    # Set up checkpoint callback (saves model periodically)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=seed_model_dir,
        name_prefix='checkpoint'
    )
    
    # Create PPO model
    print("\nInitializing PPO model...")
    policy_kwargs = dict(net_arch=[512, 512, 512])
    model = PPO(
        policy='CnnPolicy',           # CNN for image observations
        env=vec_env,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log=seed_log_dir,
        **ppo_params,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nModel Configuration:")
    print(f"  Policy: CnnPolicy (for image input)")
    print(f"  Hyperparameter Profile: {hyperparam_profile}")
    if timeout_rate is not None:
        print(f"  Mistake Log Timeout Rate: {timeout_rate:.2%}")
    print(f"  Learning Rate: {ppo_params['learning_rate']}")
    print(f"  Batch Size: {ppo_params['batch_size']}")
    print(f"  n_steps: {ppo_params['n_steps']}")
    print(f"  n_epochs: {ppo_params['n_epochs']}")
    print(f"  ent_coef: {ppo_params['ent_coef']}")
    print(f"  gamma: {ppo_params['gamma']}")
    print(f"  Parallel Envs: {n_envs}")
    print(f"  Env Max Steps: {env_max_steps}")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Seed: {seed}")
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{seed_model_dir}/final_model"
    model.save(final_model_path)
    print(f"\n[SUCCESS] Training complete! Final model saved to {final_model_path}")
    
    # Close environments
    vec_env.close()
    eval_env.close()
    
    return model, seed_model_dir


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO on static mazes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--timesteps', type=int, default=300000, help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument(
        '--profile',
        type=str,
        default='tuned',
        choices=['default', 'tuned'],
        help='Hyperparameter profile to use'
    )
    parser.add_argument('--env-max-steps', type=int, default=300, help='Max steps per episode')
    parser.add_argument(
        '--mistake-log',
        type=str,
        default='logs/mistakes/static_eval_mistakes.jsonl',
        help='Mistake log used for timeout-aware training adjustments'
    )
    
    args = parser.parse_args()
    
    # Train model
    model, model_dir = train_static_baseline(
        seed=args.seed,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        hyperparam_profile=args.profile,
        env_max_steps=args.env_max_steps,
        mistake_log_path=args.mistake_log
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel saved in: {model_dir}")
    print("\nNext steps:")
    print("  1. View training progress: tensorboard --logdir logs/static_baseline")
    print("  2. Test the model: python src/evaluation/evaluate_model.py --episodes 20 --no-render")
    print("  3. Analyze failures: python src/analysis/failure_analysis.py")
    print("  4. Visualize results: python src/visualization/visualize_maze.py")


if __name__ == '__main__':
    main()
