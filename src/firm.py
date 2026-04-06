import numpy as np
from scipy.optimize import brentq


def _rho(sigma):
    return (sigma - 1.0) / sigma


def k_from_r(r, alpha=0.5, sigma=0.4, delta=0.1):
    """
    Capital per worker from interest rate. Invert CES eq 13.
    CES: y = [alpha*k^rho + (1-alpha)]^{1/rho}, MPK = alpha*k^{rho-1}*y^{1-rho}.
    Solves MPK = r + delta numerically.
    """
    rho = _rho(sigma)

    def residual(k):
        y = (alpha * k ** rho + (1.0 - alpha)) ** (1.0 / rho)
        mpk = alpha * k ** (rho - 1.0) * y ** (1.0 - rho)
        return mpk - (r + delta)

    return brentq(residual, 1e-6, 1e4)


def wage_from_k(k, alpha=0.5, sigma=0.4):
    """
    Wage from capital per worker. CES eq 14.
    w = MPL = (1-alpha) * y^{1-rho}  (at L=1).
    """
    rho = _rho(sigma)
    y = (alpha * k ** rho + (1.0 - alpha)) ** (1.0 / rho)
    return (1.0 - alpha) * y ** (1.0 - rho)


def r_from_k(k, alpha=0.5, sigma=0.4, delta=0.1):
    """
    Interest rate from capital per worker (analytical, no root-finding).
    Used in the OLG transition solver to invert aggregate capital to r.
    """
    rho = _rho(sigma)
    y = (alpha * k ** rho + (1.0 - alpha)) ** (1.0 / rho)
    mpk = alpha * k ** (rho - 1.0) * y ** (1.0 - rho)
    return mpk - delta


def k_over_w_from_r(r, alpha=0.5, sigma=0.4, delta=0.1):
    """k/w as function of r. Needed for BGP solver."""
    k = k_from_r(r, alpha, sigma, delta)
    w = wage_from_k(k, alpha, sigma)
    return k / w
