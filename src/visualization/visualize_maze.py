"""
Visualize agent behavior in mazes.
"""

import os
import sys
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import numpy as np
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.append(str(SRC_ROOT))


def _run_live_episode(model_path, max_steps=300):
    """Render a live window while the model is running."""
    from environments.env_factory import create_static_env
    env = create_static_env(render_mode='human', max_steps=max_steps)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    env.close()
    return steps, bool(done), bool(truncated)


def visualize_agent_trajectory(model_path, save_path='figures/trajectories/example.png', show_plot=True, live=True, max_steps=300):
    """
    Visualize agent's path through maze.
    
    Args:
        model_path: Path to trained model
        save_path: Where to save visualization
    """
    print(f"Visualizing agent trajectory...")
    
    if live:
        try:
            live_steps, live_done, live_truncated = _run_live_episode(model_path, max_steps=max_steps)
            print(f"[LIVE] Episode complete: steps={live_steps}, done={live_done}, truncated={live_truncated}")
        except Exception as exc:
            print(f"[WARN] Live visualization unavailable: {exc}")
            print("[WARN] Continuing with saved trajectory figure only.")

    # Load model
    model = PPO.load(model_path)
    
    # Create environment with RGB rendering
    # Create environment with RGB rendering
    from environments.env_factory import create_static_env
    env = create_static_env(render_mode='rgb_array', max_steps=max_steps)
    
    # Run one episode and collect frames
    obs, _ = env.reset()
    frames = []
    done = False
    truncated = False
    steps = 0
    
    while not (done or truncated) and steps < max_steps:
        # Render current state
        frame = env.render()
        frames.append(frame)
        
        # Take action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    
    env.close()
    
    # Create figure with key frames
    n_frames = min(6, len(frames))
    indices = np.linspace(0, len(frames)-1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(frames[idx])
        axes[i].set_title(f'Step {idx}')
        axes[i].axis('off')
    
    plt.suptitle(f'Agent Trajectory ({steps} steps total)', fontsize=16)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Saved to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize agent behavior')
    default_model = PROJECT_ROOT / 'models' / 'static_baseline' / 'seed_42' / 'best_model'
    default_output = PROJECT_ROOT / 'figures' / 'trajectories' / 'static_example.png'
    parser.add_argument('--model', type=str,
                       default=str(default_model),
                       help='Path to model')
    parser.add_argument('--output', type=str,
                       default=str(default_output),
                       help='Output path')
    parser.add_argument('--no-show', action='store_true', help='Do not display matplotlib figure window')
    parser.add_argument('--no-live', action='store_true', help='Do not show live environment while running')
    parser.add_argument('--max-steps', type=int, default=300, help='Maximum steps for visualization episode')
    
    args = parser.parse_args()
    
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model_zip_path = model_path if model_path.suffix == '.zip' else model_path.with_suffix('.zip')
    model_load_path = model_path.with_suffix('') if model_path.suffix == '.zip' else model_path

    if not model_zip_path.exists():
        print(f"ERROR: Model not found at {model_zip_path}")
        return
    
    visualize_agent_trajectory(
        str(model_load_path),
        args.output,
        show_plot=not args.no_show,
        live=not args.no_live,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()
