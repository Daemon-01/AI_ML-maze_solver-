"""
Generate results tables.
"""

import os
import pandas as pd


def generate_summary_table(results, output_path):
    """Generate summary statistics table."""
    data = []
    for condition, r in results.items():
        data.append({
            'Condition': condition,
            'Success Rate': f"{r['mean_success_rate']*100:.1f}%",
            'Mean Reward': f"{r['mean_reward']:.2f}",
            'Episode Length': f"{r['mean_length']:.1f}"
        })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == '__main__':
    pass
