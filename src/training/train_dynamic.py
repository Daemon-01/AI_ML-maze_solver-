"""
Training script for dynamic maze environments.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import sys

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.env_factory import create_dynamic_env
from training.hyperparameters import get_ppo_params


def train_dynamic(
    env_name,
    change_probability=0.02,
    total_timesteps=1_000_000,
    seed=42,
    log_dir='logs/dynamic',
    model_dir='models/dynamic'
):
    """
    Train PPO agent on dynamic maze environment.
    
    Args:
        env_name: Name of the maze environment
        change_probability: Probability of door changes per step
        total_timesteps: Total training timesteps
        seed: Random seed
        log_dir: Directory for training logs
        model_dir: Directory to save models
    """
    # Create directories
    seed_log_dir = os.path.join(log_dir, f'seed_{seed}')
    seed_model_dir = os.path.join(model_dir, f'seed_{seed}')
    os.makedirs(seed_log_dir, exist_ok=True)
    os.makedirs(seed_model_dir, exist_ok=True)
    
    # Create environment
    env = create_dynamic_env(env_name, change_probability=change_probability)
    
    # Get hyperparameters
    ppo_params = get_ppo_params()
    
    # Create model
    model = PPO(
        'CnnPolicy',
        env,
        seed=seed,
        tensorboard_log=seed_log_dir,
        **ppo_params
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=seed_model_dir,
        log_path=seed_log_dir,
        eval_freq=10000,
        deterministic=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=seed_model_dir,
        name_prefix='checkpoint'
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save final model
    model.save(os.path.join(seed_model_dir, 'final_model'))
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO on dynamic mazes')
    parser.add_argument('--env', type=str, default='MiniGrid-FourRooms-v0', help='Environment name')
    parser.add_argument('--prob', type=float, default=0.02, help='Dynamic change probability')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total timesteps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_dynamic(
        args.env,
        change_probability=args.prob,
        total_timesteps=args.timesteps,
        seed=args.seed
    )
