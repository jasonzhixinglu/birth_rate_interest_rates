import numpy as np
from scipy.optimize import brentq


def bgp_growth_rate(b, J=75, F_low=20, F_high=30):
    """
    Solves for BGP population growth rate g given birth rate b.
    Equation: b = 1 / sum_{i=F_low}^{F_high-1} (1+g)^{-i}
    Use scipy root-finding. Returns scalar g.
    """
    def equation(g):
        s = sum((1 + g) ** (-i) for i in range(F_low, F_high))
        return b * s - 1.0

    return brentq(equation, -0.05, 0.15)


def simulate_cohort_sizes(b_high, b_low, shock_year, base_year,
                           end_year, J=75, F_low=20, F_high=30):
    """
    Simulates cohort sizes N_t from base_year to end_year.
    Before shock_year: birth rate is b_high.
    From shock_year onward: birth rate drops to b_low.
    Birth equation: N_t = b_t * sum_{i=F_low}^{F_high-1} N_{t-1-i}
    Initialize assuming constant growth at rate g_high.
    Returns dict: years, cohort_sizes, g_high, g_low.
    """
    g_high = bgp_growth_rate(b_high, J, F_low, F_high)
    g_low  = bgp_growth_rate(b_low,  J, F_low, F_high)

    years = np.arange(base_year, end_year + 1)
    T = len(years)
    N = np.zeros(T)

    # Pre-base_year cohorts grow at g_high, normalized so N[base_year] = 1
    def pre_N(year):
        return (1.0 + g_high) ** (year - base_year)

    for i, year in enumerate(years):
        b_t = b_high if year < shock_year else b_low
        parent_sum = 0.0
        for fi in range(F_low, F_high):
            py = year - 1 - fi          # birth year of parent at fertile age fi
            if py < base_year:
                parent_sum += pre_N(py)
            else:
                parent_sum += N[py - base_year]
        N[i] = b_t * parent_sum

    return {
        'years': years,
        'cohort_sizes': N,
        'g_high': g_high,
        'g_low': g_low,
    }


def compute_age_distribution(cohort_sizes, years, eval_year, J=75):
    """
    Returns array of cohort sizes by age at eval_year.
    Index 0 = age 0 (born in eval_year); index J-1 = oldest living cohort.
    """
    years = np.asarray(years)
    cohort_sizes = np.asarray(cohort_sizes)
    dist = np.zeros(J)
    for age in range(J):
        birth_year = eval_year - age
        idx = np.searchsorted(years, birth_year)
        if 0 <= idx < len(years) and years[idx] == birth_year:
            dist[age] = cohort_sizes[idx]
    return dist


if __name__ == '__main__':
    J, F_low, F_high = 75, 20, 30
    b_high = 2.5 / (2 * (F_high - F_low))
    b_low  = 1.4 / (2 * (F_high - F_low))
    result = simulate_cohort_sizes(
        b_high=b_high, b_low=b_low,
        shock_year=1970, base_year=1900, end_year=2200,
        J=J, F_low=F_low, F_high=F_high,
    )
    print(f"g_high = {result['g_high']:.6f}")
    print(f"g_low  = {result['g_low']:.6f}")
