"""
Analyze failure cases.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_mistake_log(log_path):
    """
    Analyze common failure patterns from evaluation mistake logs.

    Args:
        log_path: Path to a JSONL file produced by evaluate_model.py --mistake-log
    """
    records = []
    with Path(log_path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return {
            'runs': 0,
            'total_failed_episodes': 0,
            'timeout_failures': 0,
            'terminal_failures': 0,
            'common_final_positions': []
        }

    timeout_failures = 0
    terminal_failures = 0
    final_positions = Counter()
    total_failed = 0

    for record in records:
        for episode in record.get('failed_episodes', []):
            total_failed += 1
            if episode.get('truncated', False):
                timeout_failures += 1
            else:
                terminal_failures += 1
            pos = episode.get('final_agent_pos')
            if isinstance(pos, list) and len(pos) == 2:
                final_positions[tuple(pos)] += 1

    return {
        'runs': len(records),
        'total_failed_episodes': total_failed,
        'timeout_failures': timeout_failures,
        'terminal_failures': terminal_failures,
        'common_final_positions': [
            {'position': [x, y], 'count': count}
            for (x, y), count in final_positions.most_common(5)
        ]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze evaluation mistake logs')
    parser.add_argument(
        '--log',
        type=str,
        default='logs/mistakes/static_eval_mistakes.jsonl',
        help='Path to mistake log JSONL'
    )
    args = parser.parse_args()

    summary = analyze_mistake_log(args.log)
    print('=' * 60)
    print('MISTAKE LOG SUMMARY')
    print('=' * 60)
    print(f"Runs analyzed:           {summary['runs']}")
    print(f"Failed episodes:         {summary['total_failed_episodes']}")
    print(f"Timeout failures:        {summary['timeout_failures']}")
    print(f"Terminal failures:       {summary['terminal_failures']}")
    print('Common final positions:')
    for item in summary['common_final_positions']:
        print(f"  Pos {item['position']}: {item['count']} times")
    if not summary['common_final_positions']:
        print('  None')
