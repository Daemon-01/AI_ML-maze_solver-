"""Tests for metrics."""

import unittest
from src.evaluation.metrics import calculate_success_rate


class TestMetrics(unittest.TestCase):
    def test_success_rate(self):
        self.assertEqual(calculate_success_rate([True, True, False]), 2/3)
        self.assertEqual(calculate_success_rate([]), 0.0)


if __name__ == '__main__':
    unittest.main()
