"""
CRITICAL FIX: Reward Shaping Wrapper for MiniGrid

This wrapper adds dense rewards to help the agent learn.
Without this, the agent gets almost no feedback!

Place this in: src/environments/reward_shaping_wrapper.py
"""

import gymnasium as gym
import numpy as np


class DenseRewardWrapper(gym.Wrapper):
    """
    Add dense rewards to MiniGrid to make learning possible.
    
    Default MiniGrid reward: -1 per step, +1-N at goal
    This is TOO SPARSE - agent learns nothing!
    
    This wrapper adds:
    1. Distance-based rewards (moving toward goal)
    2. Door interaction rewards (picking up key, opening door)
    3. Exploration bonuses
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance = None
        self.key_picked_up = False
        self.door_opened = False
        self.visited_positions = set()
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset tracking
        self.key_picked_up = False
        self.door_opened = False
        self.visited_positions = set()
        
        # Calculate initial distance to goal
        agent_pos = tuple(self.env.unwrapped.agent_pos)
        goal_pos = tuple(self.env.unwrapped.goal_pos)
        self.prev_distance = np.linalg.norm(
            np.array(agent_pos) - np.array(goal_pos)
        )
        
        self.visited_positions.add(agent_pos)
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Start with base reward
        shaped_reward = reward
        
        # Get current state
        agent_pos = tuple(self.env.unwrapped.agent_pos)
        carrying = self.env.unwrapped.carrying
        
        # ====================================================================
        # 1. DISTANCE-BASED REWARD (most important!)
        # ====================================================================
        goal_pos = tuple(self.env.unwrapped.goal_pos)
        current_distance = np.linalg.norm(
            np.array(agent_pos) - np.array(goal_pos)
        )
        
        # Reward for getting closer, penalty for getting farther
        distance_reward = (self.prev_distance - current_distance) * 1.0
        shaped_reward += distance_reward
        self.prev_distance = current_distance
        
        # ====================================================================
        # 2. KEY PICKUP REWARD
        # ====================================================================
        if carrying is not None and not self.key_picked_up:
            shaped_reward += 5.0  # Big reward for picking up key!
            self.key_picked_up = True
            info['key_picked_up'] = True
        
        # ====================================================================
        # 3. DOOR OPENING REWARD
        # ====================================================================
        # Check if any door was opened this step
        grid = self.env.unwrapped.grid
        door_opened_this_step = False
        
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell and cell.type == 'door' and cell.is_open:
                    if not self.door_opened:
                        door_opened_this_step = True
        
        if door_opened_this_step:
            shaped_reward += 5.0  # Big reward for opening door!
            self.door_opened = True
            info['door_opened'] = True
        
        # ====================================================================
        # 4. EXPLORATION BONUS
        # ====================================================================
        if agent_pos not in self.visited_positions:
            shaped_reward += 0.1  # Small bonus for exploring new tiles
            self.visited_positions.add(agent_pos)
        
        # ====================================================================
        # 5. PENALTY FOR USELESS ACTIONS
        # ====================================================================
        # Penalize spinning in place
        if action in [0, 1]:  # Turn left/right
            shaped_reward -= 0.05
        
        # ====================================================================
        # 6. GOAL REACHED BONUS (keep original big reward!)
        # ====================================================================
        if terminated and not truncated:
            shaped_reward += 10.0  # Extra bonus for success!
        
        return obs, shaped_reward, terminated, truncated, info


class SimpleDistanceReward(gym.Wrapper):
    """
    Even simpler version - just distance to goal.
    Use this if DenseRewardWrapper is too complex.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        agent_pos = self.env.unwrapped.agent_pos
        goal_pos = self.env.unwrapped.goal_pos
        self.prev_distance = np.linalg.norm(
            np.array(agent_pos) - np.array(goal_pos)
        )
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Current distance
        agent_pos = self.env.unwrapped.agent_pos
        goal_pos = self.env.unwrapped.goal_pos
        current_distance = np.linalg.norm(
            np.array(agent_pos) - np.array(goal_pos)
        )
        
        # Shaped reward
        distance_reward = (self.prev_distance - current_distance) * 1.0
        shaped_reward = reward + distance_reward
        
        self.prev_distance = current_distance
        
        return obs, shaped_reward, terminated, truncated, info
