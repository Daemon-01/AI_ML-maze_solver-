"""
Agent path visualization.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_trajectory(maze_grid, trajectory, output_path, title='Agent Trajectory'):
    """
    Visualize agent trajectory on maze.
    
    Args:
        maze_grid: 2D numpy array representing maze
        trajectory: List of (row, col) positions
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    rows, cols = maze_grid.shape
    
    # Draw maze
    for i in range(rows):
        for j in range(cols):
            if maze_grid[i, j] == 1:  # Wall
                rect = patches.Rectangle(
                    (j, rows - 1 - i), 1, 1,
                    facecolor='black'
                )
                ax.add_patch(rect)
            else:
                rect = patches.Rectangle(
                    (j, rows - 1 - i), 1, 1,
                    linewidth=0.5,
                    edgecolor='gray',
                    facecolor='white'
                )
                ax.add_patch(rect)
    
    # Draw trajectory
    if trajectory:
        traj_rows = [rows - 1 - pos[0] + 0.5 for pos in trajectory]
        traj_cols = [pos[1] + 0.5 for pos in trajectory]
        
        # Draw path line
        ax.plot(traj_cols, traj_rows, 'b-', linewidth=2, alpha=0.7)
        
        # Draw start and end markers
        ax.plot(traj_cols[0], traj_rows[0], 'go', markersize=15, label='Start')
        ax.plot(traj_cols[-1], traj_rows[-1], 'ro', markersize=15, label='End')
        
        # Add arrows to show direction
        for i in range(len(trajectory) - 1):
            dx = traj_cols[i + 1] - traj_cols[i]
            dy = traj_rows[i + 1] - traj_rows[i]
            ax.arrow(
                traj_cols[i], traj_rows[i], dx * 0.4, dy * 0.4,
                head_width=0.2, head_length=0.1, fc='blue', ec='blue', alpha=0.5
            )
    
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_multiple_trajectories(maze_grid, trajectories, colors, output_path):
    """
    Visualize multiple trajectories on the same maze.
    
    Args:
        maze_grid: 2D numpy array representing maze
        trajectories: List of trajectory lists
        colors: List of colors for each trajectory
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    rows, cols = maze_grid.shape
    
    # Draw maze
    for i in range(rows):
        for j in range(cols):
            if maze_grid[i, j] == 1:
                rect = patches.Rectangle(
                    (j, rows - 1 - i), 1, 1,
                    facecolor='black'
                )
                ax.add_patch(rect)
    
    # Draw trajectories
    for trajectory, color in zip(trajectories, colors):
        traj_rows = [rows - 1 - pos[0] + 0.5 for pos in trajectory]
        traj_cols = [pos[1] + 0.5 for pos in trajectory]
        ax.plot(traj_cols, traj_rows, color=color, linewidth=2, alpha=0.7)
    
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    pass
