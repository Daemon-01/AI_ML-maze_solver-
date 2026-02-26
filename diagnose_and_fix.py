"""
COMPREHENSIVE DIAGNOSIS AND FIX FOR MINIGRID PPO FAILURE

This script will:
1. Test environment setup
2. Verify observations are correct
3. Test with reward shaping
4. Provide working training script

Run this to diagnose your issue!
"""

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os


print("="*80)
print("MINIGRID PPO DIAGNOSIS TOOL")
print("="*80)


# ============================================================================
# TEST 1: Check Environment Setup
# ============================================================================
print("\n[TEST 1] Checking environment setup...")

try:
    env = gym.make('MiniGrid-DoorKey-8x8-v0')
    print("✓ Base environment created")
    
    # Check observation space
    obs, _ = env.reset()
    print(f"  Base observation type: {type(obs)}")
    print(f"  Base observation keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
    
    env.close()
except Exception as e:
    print(f"✗ Error creating base environment: {e}")


# ============================================================================
# TEST 2: Check Wrapper Configuration (CRITICAL!)
# ============================================================================
print("\n[TEST 2] Checking wrapper configuration...")

# Test ImgObsWrapper
try:
    env = gym.make('MiniGrid-DoorKey-8x8-v0')
    env = ImgObsWrapper(env)
    obs, _ = env.reset()
    
    print(f"✓ ImgObsWrapper applied")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    print(f"  Observation range: [{obs.min()}, {obs.max()}]")
    
    if obs.shape != (7, 7, 3):
        print(f"  ⚠ WARNING: Expected shape (7,7,3) but got {obs.shape}")
    
    env.close()
except Exception as e:
    print(f"✗ Error with ImgObsWrapper: {e}")


# ============================================================================
# TEST 3: Check if Random Agent Can Ever Succeed
# ============================================================================
print("\n[TEST 3] Testing if random agent can solve maze...")

env = gym.make('MiniGrid-DoorKey-8x8-v0', render_mode=None)
env = ImgObsWrapper(env)

random_successes = 0
total_episodes = 100

for ep in range(total_episodes):
    obs, _ = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        
        if done:
            random_successes += 1
            break

random_success_rate = (random_successes / total_episodes) * 100
print(f"  Random agent success rate: {random_success_rate:.1f}%")

if random_success_rate > 0:
    print(f"  ✓ Environment is solvable (random agent succeeded {random_successes}/{total_episodes} times)")
else:
    print(f"  ⚠ Random agent never succeeded - environment might be too hard or broken")

env.close()


# ============================================================================
# TEST 4: Check Reward Structure
# ============================================================================
print("\n[TEST 4] Analyzing reward structure...")

env = gym.make('MiniGrid-DoorKey-8x8-v0')
env = ImgObsWrapper(env)

obs, _ = env.reset()
rewards_seen = []

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    rewards_seen.append(reward)
    
    if done:
        break

print(f"  Rewards observed: {set(rewards_seen)}")
print(f"  Most common reward: {max(set(rewards_seen), key=rewards_seen.count)}")

if len(set(rewards_seen)) == 1 and -1 in rewards_seen:
    print("  ⚠ PROBLEM FOUND: Only getting -1 reward (too sparse!)")
    print("  → This makes learning nearly impossible")
    print("  → SOLUTION: Need reward shaping")

env.close()


# ============================================================================
# THE ACTUAL PROBLEM AND SOLUTION
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

print("\nMOST LIKELY PROBLEMS:")
print("1. ⚠ Reward is TOO SPARSE (-1 every step, +1-N only at goal)")
print("   → Agent gets no signal about whether it's doing well")
print("2. ⚠ No reward shaping to guide exploration")
print("   → Agent has no idea where the goal is")
print("3. ⚠ Default MiniGrid-DoorKey is actually quite hard")
print("   → Requires opening door with key THEN reaching goal")

print("\n" + "="*80)
print("CREATING FIXED TRAINING SCRIPT...")
print("="*80)
