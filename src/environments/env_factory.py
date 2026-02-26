"""
Environment factory functions.
"""

import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from .dynamic_door_wrapper import DynamicDoorWrapper


# Default MiniGrid maze environment (multi-room navigation)
DEFAULT_ENV = 'MiniGrid-FourRooms-v0'
DEFAULT_MAX_STEPS = 300


def create_environment(env_name=None, dynamic=False, change_probability=0.02, **kwargs):
    """
    Create a maze environment with optional dynamic wrapping.
    
    Args:
        env_name: Name of the base environment (defaults to DEFAULT_ENV)
        dynamic: Whether to apply dynamic door wrapper
        change_probability: Probability of door changes per step
        **kwargs: Additional arguments passed to gym.make
        
    Returns:
        The configured environment
    """
    if env_name is None:
        env_name = DEFAULT_ENV

    # Increase episode horizon for sparse-reward mazes unless explicitly overridden.
    kwargs.setdefault('max_steps', DEFAULT_MAX_STEPS)
    env = gym.make(env_name, **kwargs)
    env = RGBImgObsWrapper(env, tile_size=8)  # Render full RGB image
    env = ImgObsWrapper(env)  # Extract image from Dict obs for CnnPolicy
    
    if dynamic:
        env = DynamicDoorWrapper(env, change_probability=change_probability)
        
    return env


def make_static_env():
    """Zero-arg factory for use with make_vec_env."""
    return create_environment(DEFAULT_ENV, dynamic=False)


def create_static_env(env_name=None, **kwargs):
    """Create a static maze environment."""
    return create_environment(env_name, dynamic=False, **kwargs)


def create_dynamic_env(env_name=None, change_probability=0.02, **kwargs):
    """Create a dynamic maze environment."""
    return create_environment(env_name, dynamic=True, change_probability=change_probability, **kwargs)
