import numpy as np
from scipy.optimize import brentq

from .firm import k_over_w_from_r
from .household import solve_household


def bgp_interest_rate(g, b, J=80, chi=20, psi=60, F_low=20, F_high=30,
                       alpha=0.33, sigma=0.4, delta=0.05, beta=0.946,
                       theta=2.0, gamma=0.015):
    """
    Solves for BGP interest rate r given population growth rate g and birth rate b.

    TFP growth gamma enters through a modified effective discount factor:
        beta_eff = beta * (1 + gamma)^{1 - theta}
    The nominal interest rate r is passed directly to the household solver;
    wages are normalised to efficiency units (= 1 for working ages).
    This convention is consistent with solve_transition in olg.py.

    Equilibrium condition: household asset demand per worker (k/w) equals
    firm capital supply per worker (k/w).

    Returns
    -------
    float : BGP nominal interest rate r.
    """
    beta_eff = beta * (1.0 + gamma) ** (1.0 - theta) if gamma != 0.0 else beta

    weights = np.array([(1.0 + g) ** (-i) for i in range(J)])
    labor_weights = weights.copy()
    labor_weights[:chi] = 0.0
    labor_weights[psi:] = 0.0
    L = labor_weights.sum()

    def excess_demand(r):
        r_const = np.full(J, r)
        w_const = np.zeros(J)
        w_const[chi:psi] = 1.0

        sol = solve_household(r_const, w_const, chi=chi, psi=psi,
                               J=J, beta=beta_eff, theta=theta)

        k_demand_per_w = np.dot(weights, sol['assets']) / L
        k_supply_per_w = k_over_w_from_r(r, alpha, sigma, delta)

        return k_demand_per_w - k_supply_per_w

    return brentq(excess_demand, -delta + 1e-4, 0.5)


def asset_demand(r, g, b=None, J=80, chi=20, psi=60,
                  alpha=0.33, sigma=0.4, delta=0.05, beta=0.946,
                  theta=2.0, gamma=0.015):
    """
    Aggregate household asset demand per worker (k/w units) at interest rate r.
    Wages normalised to 1 (efficiency units); cohort weights are (1+g)^{-i}.
    """
    beta_eff = beta * (1.0 + gamma) ** (1.0 - theta) if gamma != 0.0 else beta

    weights = np.array([(1.0 + g) ** (-i) for i in range(J)])
    labor_weights = weights.copy()
    labor_weights[:chi] = 0.0
    labor_weights[psi:] = 0.0
    L = labor_weights.sum()

    r_const = np.full(J, r)
    w_const = np.zeros(J)
    w_const[chi:psi] = 1.0
    sol = solve_household(r_const, w_const, chi=chi, psi=psi,
                           J=J, beta=beta_eff, theta=theta)
    return np.dot(weights, sol['assets']) / L


def capital_supply(r, alpha=0.33, sigma=1.0, delta=0.05):
    """Firm capital supply per unit wage (k/w) at interest rate r."""
    return k_over_w_from_r(r, alpha, sigma, delta)
