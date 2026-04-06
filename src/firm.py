import numpy as np
from scipy.optimize import brentq

_CD_TOL = 1e-10   # threshold for treating sigma as exactly 1 (Cobb-Douglas)


def _rho(sigma):
    return (sigma - 1.0) / sigma


def k_from_r(r, alpha=0.33, sigma=0.4, delta=0.05):
    """
    Capital per worker from interest rate. Invert firm FOC.

    Cobb-Douglas (sigma=1): r + delta = alpha * k^{alpha-1}
    CES (sigma!=1):         r + delta = alpha * k^{rho-1} * y^{1-rho}
    """
    if abs(sigma - 1.0) < _CD_TOL:
        return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha))
    rho = _rho(sigma)
    def residual(k):
        y = (alpha * k ** rho + (1.0 - alpha)) ** (1.0 / rho)
        mpk = alpha * k ** (rho - 1.0) * y ** (1.0 - rho)
        return mpk - (r + delta)
    return brentq(residual, 1e-6, 1e4)


def wage_from_k(k, alpha=0.33, sigma=0.4):
    """
    Wage from capital per worker.

    Cobb-Douglas (sigma=1): w = (1-alpha) * k^alpha
    CES (sigma!=1):         w = (1-alpha) * y^{1-rho}  (at L=1)
    """
    if abs(sigma - 1.0) < _CD_TOL:
        return (1.0 - alpha) * k ** alpha
    rho = _rho(sigma)
    y = (alpha * k ** rho + (1.0 - alpha)) ** (1.0 / rho)
    return (1.0 - alpha) * y ** (1.0 - rho)


def r_from_k(k, alpha=0.33, sigma=0.4, delta=0.05):
    """
    Interest rate from capital per worker (analytical, no root-finding).
    Used in OLG transition solver to convert aggregate capital to r.
    """
    if abs(sigma - 1.0) < _CD_TOL:
        return alpha * k ** (alpha - 1.0) - delta
    rho = _rho(sigma)
    y = (alpha * k ** rho + (1.0 - alpha)) ** (1.0 / rho)
    mpk = alpha * k ** (rho - 1.0) * y ** (1.0 - rho)
    return mpk - delta


def k_over_w_from_r(r, alpha=0.33, sigma=0.4, delta=0.05):
    """k/w as function of r. Used by BGP solver."""
    k = k_from_r(r, alpha, sigma, delta)
    w = wage_from_k(k, alpha, sigma)
    return k / w
