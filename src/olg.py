import numpy as np

from .household import solve_household
from .firm import k_from_r, wage_from_k, r_from_k


def solve_transition(cohort_sizes, years, r_init, r_terminal,
                      J=75, chi=20, psi=65,
                      alpha=0.5, sigma=0.4, delta=0.1,
                      beta=0.99, theta=2, gamma=0.0,
                      phi=0.05, max_iter=2000, tol=1e-6):
    """
    Iterative algorithm to solve for the transitional interest rate path.

    Algorithm
    ---------
    1. Initialise r_path as linear interpolation from r_init to r_terminal.
    2. Each iteration:
       a. Compute w_path from r_path via firm FOC.
       b. Solve household problem for every birth cohort overlapping [years].
       c. Aggregate assets across cohorts; compute implied capital per worker.
       d. Convert to r_implied via analytical MPK inversion.
       e. Damped update: r_path = (1-phi)*r_path + phi*r_implied.
    3. Converge when max|r - r_implied| < tol.

    Parameters
    ----------
    cohort_sizes : array-like, length T
        Cohort sizes N_t from simulate_cohort_sizes.
    years : array-like, length T
        Calendar years corresponding to cohort_sizes.
    r_init, r_terminal : float
        Initial and terminal (BGP) interest rates for the linear guess.
    gamma : float
        TFP growth rate. When non-zero the Euler equation and wage paths are
        rescaled: households solve in detrended (efficiency-unit) terms using
        beta_eff = beta*(1+gamma)^{1-theta} and r_eff = (1+r)/(1+gamma)-1.
    phi : float
        Damping weight on the implied interest rate update.

    Returns
    -------
    dict with keys: years, r_path, converged (bool), n_iter (int).
    """
    years = np.asarray(years, dtype=int)
    cohort_sizes = np.asarray(cohort_sizes, dtype=float)
    T = len(years)
    y0 = int(years[0])

    year_to_idx = {int(y): i for i, y in enumerate(years)}

    # Estimate pre-simulation growth rate for backward extrapolation of cohort sizes
    g_pre = float(cohort_sizes[1] / cohort_sizes[0]) - 1.0

    def get_cohort_size(by):
        if by in year_to_idx:
            return cohort_sizes[year_to_idx[by]]
        elif by < y0:
            return cohort_sizes[0] * (1.0 + g_pre) ** (by - y0)
        else:
            return cohort_sizes[-1]

    # TFP adjustments applied before passing to solve_household
    if gamma != 0.0:
        beta_eff = beta * (1.0 + gamma) ** (1.0 - theta)
    else:
        beta_eff = beta

    # Initialise r_path
    r_path = np.linspace(r_init, r_terminal, T)

    converged = False
    n_iter = 0

    for _ in range(max_iter):
        # --- Step a: wage path ---
        k_path = np.array([k_from_r(r, alpha, sigma, delta) for r in r_path])
        w_path = np.array([wage_from_k(k, alpha, sigma) for k in k_path])

        # --- Step b: solve all relevant birth cohorts ---
        birth_year_min = y0 - J + 1
        birth_year_max = int(years[-1])
        cohort_assets = {}

        for by in range(birth_year_min, birth_year_max + 1):
            # Calendar years along this cohort's lifetime
            yr_range = np.arange(by, by + J)

            # Index into r_path / w_path, clamped to [0, T-1]
            raw_idx = yr_range - y0
            clipped  = np.clip(raw_idx, 0, T - 1)

            r_cohort = r_path[clipped]
            w_cohort = w_path[clipped]

            if gamma != 0.0:
                # Rescale Euler: detrended interest rate
                r_cohort_eff = (1.0 + r_cohort) / (1.0 + gamma) - 1.0
                # w_cohort stays in efficiency units (already from wage_from_k)
                w_cohort_eff = w_cohort
            else:
                r_cohort_eff = r_cohort
                w_cohort_eff = w_cohort

            sol = solve_household(r_cohort_eff, w_cohort_eff,
                                   chi=chi, psi=psi, J=J,
                                   beta=beta_eff, theta=theta)
            cohort_assets[by] = sol['assets']   # in efficiency units when gamma != 0

        # --- Steps c & d: aggregate and invert to r_implied ---
        r_implied = r_path.copy()

        for t_idx, t in enumerate(years):
            t = int(t)
            total_assets = 0.0
            total_labor  = 0.0

            for age in range(J):
                by = t - age
                if by not in cohort_assets:
                    continue
                n     = get_cohort_size(by)
                a_age = cohort_assets[by][age]

                if gamma != 0.0:
                    # Convert detrended assets to nominal: multiply by A_{by+age}
                    a_age = a_age * (1.0 + gamma) ** (by + age - y0)

                total_assets += n * a_age
                if chi <= age < psi:
                    total_labor += n

            if total_labor > 0.0 and total_assets > 0.0:
                k_implied = total_assets / total_labor
                if gamma != 0.0:
                    # Convert nominal capital per worker to efficiency units
                    k_implied /= (1.0 + gamma) ** (t - y0)
                r_implied[t_idx] = r_from_k(k_implied, alpha, sigma, delta)
            # else: keep r_implied[t_idx] = r_path[t_idx] (no update)

        # --- Step e: damped update and convergence check ---
        diff = np.max(np.abs(r_implied - r_path))
        r_path = (1.0 - phi) * r_path + phi * r_implied
        n_iter += 1

        if diff < tol:
            converged = True
            break

    return {
        'years': years,
        'r_path': r_path,
        'converged': converged,
        'n_iter': n_iter,
    }
