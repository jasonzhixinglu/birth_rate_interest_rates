import numpy as np
from scipy.optimize import brentq

from .firm import k_over_w_from_r
from .household import solve_household


def bgp_interest_rate(g, b, J=75, chi=20, psi=65, F_low=20, F_high=30,
                       alpha=0.5, sigma=0.4, delta=0.1, beta=0.99, theta=2):
    """
    Solves for BGP interest rate r given population growth rate g and birth rate b.

    Equates household asset demand per worker to capital supply per worker.
    Wages normalized to 1; equilibrium condition is in k/w units so the
    wage level cancels.

    Parameters
    ----------
    g : float
        BGP population growth rate (from bgp_growth_rate).
    b : float
        Birth rate (used for completeness; g already encodes demographic steady state).

    Returns
    -------
    float : BGP interest rate r.
    """
    # Cross-sectional cohort-size weights: younger cohorts are larger
    weights = np.array([(1.0 + g) ** (-i) for i in range(J)])

    # Effective labour supply in cross-section
    labor_weights = weights.copy()
    labor_weights[:chi] = 0.0
    labor_weights[psi:] = 0.0
    L = labor_weights.sum()

    def excess_demand(r):
        # Household problem with constant r and w=1 for working ages
        r_const = np.full(J, r)
        w_const = np.zeros(J)
        w_const[chi:psi] = 1.0

        sol = solve_household(r_const, w_const, chi=chi, psi=psi,
                               J=J, beta=beta, theta=theta)

        # Aggregate assets across cohort ages, normalised by labour force
        k_demand_per_w = np.dot(weights, sol['assets']) / L

        # Firm capital supply per unit wage
        k_supply_per_w = k_over_w_from_r(r, alpha, sigma, delta)

        return k_demand_per_w - k_supply_per_w

    return brentq(excess_demand, -0.05, 0.25)
