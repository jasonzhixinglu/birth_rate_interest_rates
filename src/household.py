import numpy as np


def solve_household(r_path, w_path, chi=20, psi=60, J=80, beta=0.946, theta=2.0):
    """
    Given paths for r and w, solves consumption and asset accumulation
    for a cohort whose economic life begins at age chi.

    "No children in utility": the household does not optimise over ages
    0..chi-1.  Economic life runs from biological age chi to J-1, with
    a_chi = 0 (enters with no assets) and a_J = 0 (no bequest).
    Consumption and assets are zero for ages 0..chi-1.

    Euler equation (CRRA, no child adjustment):
        c_{i+1}/c_i = [beta*(1+r_i)]^{1/theta}   for i >= chi

    Budget: a_{i+1} = (1+r_i)*a_i + w_i - c_i,  a_chi = 0, a_J = 0.

    Parameters
    ----------
    r_path : array-like, length J
        Interest rate at biological age i (rate from age i to i+1).
    w_path : array-like, length J
        Wage at biological age i (zero outside [chi, psi)).
    chi, psi : int
        First working age and first retirement age (biological).
    J : int
        Total lifespan in periods.
    beta, theta : float
        Discount factor and CRRA coefficient.

    Returns
    -------
    dict with keys 'consumption' and 'assets', both length-J arrays.
    Entries 0..chi-1 are zero.
    """
    r_path = np.asarray(r_path, dtype=float)
    w_path = np.asarray(w_path, dtype=float)

    # Economic lifespan: from chi to J-1
    J_econ = J - chi
    r_econ = r_path[chi:]          # length J_econ
    w_econ = np.zeros(J_econ)
    w_econ[:psi - chi] = w_path[chi:psi]   # wages during working years

    # Euler growth factors: growth_econ[0]=1 (at economic age 0 = biological chi)
    #   growth_econ[k] = prod_{j=0}^{k-1} [beta*(1+r_econ[j])]^{1/theta}
    log_g = np.zeros(J_econ)
    log_g[1:] = np.cumsum(np.log(beta * (1.0 + r_econ[:-1])) / theta)
    growth_econ = np.exp(log_g)

    # Cumulative discount from chi: cum_R[0]=1, cum_R[k]=prod_{j=0}^{k-1}(1+r_econ[j])
    log_R = np.zeros(J_econ)
    log_R[1:] = np.cumsum(np.log(1.0 + r_econ[:-1]))
    cum_R_econ = np.exp(log_R)

    # c_chi from lifetime budget: PV(consumption from chi) = PV(income from chi)
    pv_income = np.sum(w_econ / cum_R_econ)
    pv_cons_factor = np.sum(growth_econ / cum_R_econ)
    c_chi = pv_income / pv_cons_factor

    consumption_econ = c_chi * growth_econ

    # Forward simulation of assets starting from a_chi = 0
    assets_econ = np.zeros(J_econ)
    for k in range(J_econ - 1):
        assets_econ[k + 1] = ((1.0 + r_econ[k]) * assets_econ[k]
                               + w_econ[k] - consumption_econ[k])

    # Embed into full length-J arrays (pre-chi entries remain zero)
    consumption = np.zeros(J)
    assets = np.zeros(J)
    consumption[chi:] = consumption_econ
    assets[chi:] = assets_econ

    return {'consumption': consumption, 'assets': assets}
