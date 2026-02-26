"""
Statistical analysis functions.
"""

import numpy as np
from scipy import stats


def perform_ttest(group1, group2, paired=False):
    """Perform t-test between two groups."""
    if paired:
        stat, pvalue = stats.ttest_rel(group1, group2)
    else:
        stat, pvalue = stats.ttest_ind(group1, group2)
    return {'statistic': stat, 'pvalue': pvalue}


def perform_anova(groups):
    """Perform one-way ANOVA across groups."""
    stat, pvalue = stats.f_oneway(*groups)
    return {'statistic': stat, 'pvalue': pvalue}


if __name__ == '__main__':
    pass
