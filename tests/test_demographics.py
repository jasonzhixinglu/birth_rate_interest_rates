import numpy as np
import pytest
from src.demographics import bgp_growth_rate, simulate_cohort_sizes


def test_bgp_growth_rate_one_percent():
    """bgp_growth_rate recovers g ≈ 0.01 for b consistent with 1% growth."""
    F_low, F_high = 20, 30
    g_target = 0.01
    b = 1.0 / sum((1 + g_target) ** (-i) for i in range(F_low, F_high))
    g_recovered = bgp_growth_rate(b, F_low=F_low, F_high=F_high)
    assert abs(g_recovered - g_target) < 1e-8


def test_cohort_sizes_converge_to_g_low():
    """Cohort sizes grow at g_low well after the fertility shock."""
    J, F_low, F_high = 75, 20, 30
    b_high = 2.5 / (F_high - F_low)
    b_low  = 1.4 / (F_high - F_low)
    result = simulate_cohort_sizes(
        b_high, b_low,
        shock_year=1970, base_year=1900, end_year=2200,
        J=J, F_low=F_low, F_high=F_high,
    )
    N = result['cohort_sizes']
    years = result['years']
    g_low = result['g_low']

    # By 2150–2180 every parent cohort was born well after the shock;
    # growth rate should match g_low to high precision.
    idx_a = np.searchsorted(years, 2150)
    idx_b = np.searchsorted(years, 2180)
    span = int(years[idx_b] - years[idx_a])
    actual_g = (N[idx_b] / N[idx_a]) ** (1.0 / span) - 1.0
    assert abs(actual_g - g_low) < 1e-3
