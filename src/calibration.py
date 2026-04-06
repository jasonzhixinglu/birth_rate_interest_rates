"""
Calibration module.

Maps empirical targets to structural model parameters for each country case.
All functions return a parameter dict ready to pass to solve_transition().
"""


def calibrate_germany(
    tfr_high=2.5,
    tfr_low=1.4,
    shock_year=1970,
    base_year=1900,
    end_year=2200,
    J=80,
    chi=20,
    psi=60,
    F_low=20,
    F_high=30,
    alpha=0.33,
    sigma=0.4,
    delta=0.05,
    beta=0.946,
    theta=2.0,
    gamma=0.015,
):
    """
    Return calibrated parameters for the Germany baseline.

    Parameters
    ----------
    tfr_high : float
        Pre-shock total fertility rate.
    tfr_low : float
        Post-shock total fertility rate (post-1970 collapse).
    shock_year : int
        Year of permanent fertility decline.
    J : int
        Individual lifespan in years.
    F_low, F_high : int
        Fertile age window [F_low, F_high).
    alpha : float
        Capital share in CES production.
    sigma : float
        Elasticity of substitution in CES production.
    delta : float
        Annual depreciation rate.
    beta : float
        Annual discount factor.
    theta : float
        CRRA coefficient (inverse of EIS).
    gamma : float
        Annual TFP growth rate.

    Returns
    -------
    dict
        Parameter dict suitable for passing to solve_transition().
    """
    raise NotImplementedError


def calibrate_japan(tfr_high=2.0, tfr_low=1.2, shock_year=1975, **kwargs):
    """Return calibrated parameters for the Japan comparison."""
    raise NotImplementedError


def calibrate_usa(tfr_high=2.5, tfr_low=1.8, shock_year=1972, **kwargs):
    """Return calibrated parameters for the USA comparison."""
    raise NotImplementedError
