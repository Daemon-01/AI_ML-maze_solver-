"""
Generate demo videos.
"""

import os
import numpy as np
from stable_baselines3 import PPO


def record_episode(model, env, output_path, max_steps=500):
    """
    Record a single episode as video.
    
    Args:
        model: Trained PPO model
        env: Environment (should have render_mode='rgb_array')
        output_path: Path to save the video
        max_steps: Maximum steps per episode
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return
    
    frames = []
    obs, info = env.reset()
    
    for step in range(max_steps):
        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Add final frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break
    
    if frames:
        # Write video
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        
        video.release()
        print(f"Video saved to {output_path}")
    else:
        print("No frames captured")


def create_comparison_video(models, envs, labels, output_path):
    """
    Create side-by-side comparison video.
    
    Args:
        models: List of trained models
        envs: List of environments
        labels: List of condition labels
        output_path: Path to save the video
    """
    # Implementation for side-by-side video
    pass


if __name__ == '__main__':
    pass
