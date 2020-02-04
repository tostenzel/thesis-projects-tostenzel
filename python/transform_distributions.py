"""
Implementation of the inverse Rosenblatt / inverse Nataf transformation in
Ge/Menendez (2017), page  2017 from unit cube to multivariate normal space
with given correlations.

REMARK: All one-dimensional arrays are taken as row vectors by numpy
although they are no transposed correctly but mirrored.

"""
import random

import numpy as np
import scipy.linalg as linalg
from scipy.stats import norm


def covariance_to_correlation(cov):
    """Converts covariance matrix to correlation matrix.

    Parameters
    ----------
    Returns
    -------
    """
    # Standard deviations of each variable.
    sd = np.sqrt(np.diag(cov)).reshape(1, len(cov))
    corr = cov / sd.T / sd

    return corr


def transform_uniform_stnormal_uncorr(row_traj_reordered, numeric_zero=0.01):
    """
    Convert sample from uniform distribution to standard normal space
    without any correlations.

    Parameters
    ----------
    Returns
    -------

    """
    # Need to replace ones, because norm.ppf(1) = inf and zeros because norm.ppf(0) = -inf
    # Numerical Parameters taken from crappy MATLAB code by Ge/Menendez (2017).
    # Highly influential on the EE scale.
    row_approx = np.where(row_traj_reordered == 1, 1 - numeric_zero, row_traj_reordered)
    row_approx = np.where(row_approx == 0, numeric_zero, row_approx)

    # Step 1: Inverse cdf of standard normal distribution (N(0, 1)).
    z = norm.ppf(row_approx)

    return z


def transform_stnormal_normal_corr_lemaire09(z, cov, mu):
    """
    Inverse Rosenblatt/Nataf transformation (from standard normal)
    to multivariate normal space with given correlations.
    Step 2) Compute correlation matrix.
    Step 3) Introduce dependencies to standard normal sample.
    Step 4) De-standardize sample to normal space.

    REMARK: This is equivalent to Gentle (2006), page 197.

    Parameters
    ----------
    Returns
    -------

    """
    # Convert covariance matrix to correlation matrix
    C = covariance_to_correlation(cov)
    Q_prime = linalg.cholesky(C, lower=True)

    z_corr_stnorm = np.dot(Q_prime, z.reshape(len(cov), 1))

    x_norm = z_corr_stnorm * np.sqrt(np.diag(cov)).reshape(len(cov), 1) + mu.reshape(
        len(cov), 1
    )

    return x_norm.T, Q_prime
