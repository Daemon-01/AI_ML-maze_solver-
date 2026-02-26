"""
Value function heatmaps.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


def extract_value_function(model, env, grid_size):
    """
    Extract value function for each state in the maze.
    
    Args:
        model: Trained PPO model
        env: Maze environment
        grid_size: Tuple of (rows, cols)
        
    Returns:
        2D numpy array of values
    """
    values = np.zeros(grid_size)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Get observation for this position
            obs = env.get_observation_for_position(i, j)
            if obs is not None:
                # Get value from critic network
                value = model.policy.predict_values(obs.reshape(1, -1))
                values[i, j] = value.item()
            else:
                values[i, j] = np.nan
    
    return values


def plot_value_heatmap(values, maze_grid, output_path, title='Value Function'):
    """
    Plot value function as heatmap.
    
    Args:
        values: 2D numpy array of values
        maze_grid: 2D numpy array representing maze
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mask walls
    masked_values = np.ma.masked_where(maze_grid == 1, values)
    
    # Create heatmap
    im = ax.imshow(masked_values, cmap='viridis', origin='upper')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('State Value', rotation=270, labelpad=15)
    
    # Overlay wall cells
    rows, cols = maze_grid.shape
    for i in range(rows):
        for j in range(cols):
            if maze_grid[i, j] == 1:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor='gray',
                    edgecolor='black'
                ))
    
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def compare_value_functions(values_list, titles, maze_grid, output_path):
    """
    Compare value functions side by side.
    
    Args:
        values_list: List of value function arrays
        titles: List of titles for each subplot
        maze_grid: 2D numpy array representing maze
        output_path: Path to save the plot
    """
    n = len(values_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    
    if n == 1:
        axes = [axes]
    
    for ax, values, title in zip(axes, values_list, titles):
        masked_values = np.ma.masked_where(maze_grid == 1, values)
        im = ax.imshow(masked_values, cmap='viridis', origin='upper')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    pass
