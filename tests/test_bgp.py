import numpy as np
import pytest
from src.demographics import bgp_growth_rate
from src.bgp import bgp_interest_rate


def test_bgp_interest_rate_finite():
    """bgp_interest_rate returns a finite scalar in a plausible range."""
    F_low, F_high = 20, 30
    b = 1.4 / (F_high - F_low)
    g = bgp_growth_rate(b, F_low=F_low, F_high=F_high)
    r = bgp_interest_rate(g, b, F_low=F_low, F_high=F_high)
    assert np.isfinite(r)
    assert -0.1 < r < 0.3


def test_higher_g_gives_higher_r():
    """Higher population growth rate implies higher BGP interest rate (Samuelson result)."""
    F_low, F_high = 20, 30
    b_low  = 1.4 / (F_high - F_low)
    b_high = 2.5 / (F_high - F_low)
    g_low  = bgp_growth_rate(b_low,  F_low=F_low, F_high=F_high)
    g_high = bgp_growth_rate(b_high, F_low=F_low, F_high=F_high)
    r_low  = bgp_interest_rate(g_low,  b_low,  F_low=F_low, F_high=F_high)
    r_high = bgp_interest_rate(g_high, b_high, F_low=F_low, F_high=F_high)
    assert r_high > r_low
