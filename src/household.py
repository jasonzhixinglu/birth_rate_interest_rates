import numpy as np


def solve_household(r_path, w_path, chi=20, psi=65, J=75, beta=0.99, theta=2):
    """
    Given paths for r and w, solves consumption and asset accumulation
    for a cohort born at t=0.

    Uses standard CRRA Euler equation (no child adjustment term):
        c_{i+1}/c_i = [beta*(1+r_i)]^{1/theta}

    Budget: a_{i+1} = (1+r_i)*a_i + w_i - c_i, with a_0 = a_J = 0.

    Parameters
    ----------
    r_path : array-like, length J
        Interest rate at each age i (rate earned on savings from age i to i+1).
    w_path : array-like, length J
        Wage at each age (zero outside [chi, psi)).
    chi, psi : int
        First working age and first retirement age.
    J : int
        Lifespan (number of periods).
    beta, theta : float
        Discount factor and CRRA coefficient.

    Returns
    -------
    dict with keys 'consumption' and 'assets', both length-J arrays.
    """
    r_path = np.asarray(r_path, dtype=float)
    w_path = np.asarray(w_path, dtype=float)

    # Wage profile restricted to working ages
    wages = np.zeros(J)
    wages[chi:psi] = w_path[chi:psi]

    # Euler growth factors: growth[0]=1, growth[i]=prod_{j=0}^{i-1}[beta*(1+r_j)]^{1/theta}
    log_g = np.zeros(J)
    log_g[1:] = np.cumsum(np.log(beta * (1.0 + r_path[:-1])) / theta)
    growth = np.exp(log_g)

    # Cumulative discount: cum_R[0]=1, cum_R[i]=prod_{j=0}^{i-1}(1+r_j)
    log_R = np.zeros(J)
    log_R[1:] = np.cumsum(np.log(1.0 + r_path[:-1]))
    cum_R = np.exp(log_R)

    # c_0 from lifetime budget: PV(consumption) = PV(income)  [with a_0=a_J=0]
    pv_income = np.sum(wages / cum_R)
    pv_cons_factor = np.sum(growth / cum_R)
    c_0 = pv_income / pv_cons_factor

    consumption = c_0 * growth

    # Forward simulation of assets
    assets = np.zeros(J)
    for i in range(J - 1):
        assets[i + 1] = (1.0 + r_path[i]) * assets[i] + wages[i] - consumption[i]

    return {'consumption': consumption, 'assets': assets}
