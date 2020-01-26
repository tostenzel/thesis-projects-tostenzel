"""
Ge/Menendez (2017)

REMARK: All one-dimensional arrays are taken as row vectors by numpy
except of that they are no transposed correctly but just mirrored.

"""
import random

import numpy as np
import scipy.linalg as linalg
from scipy.stats import norm


def covariance_to_correlation(cov):
    """Converts covariance matrix to correlation matrix."""
    # Standard deviations of each variable.
    sd = np.sqrt(np.diag(cov)).reshape(1, len(cov))
    corr = cov / sd.T / sd
    
    return corr


def sample_stnormal_paramters(n_par, n_draws=100_000, seed=123):
    random.seed(seed)
    sample_stnormal_paramters = np.random.normal(0, 1, n_par * n_draws).reshape(
        n_par, n_draws
    )
    return sample_stnormal_paramters


def transform_uniform_stnormal_uncorr(row_traj_reordered):
    # Need to replace ones, because norm.ppf(1) = inf and zeros because norm.ppf(0) = -inf
    row_approx = np.where(row_traj_reordered == 1, 0.999, row_traj_reordered)
    row_approx = np.where(row_approx == 0, 0.001, row_approx)

    # Step 1: Inverse cdf of standard normal distribution (N(0, 1)).
    z = norm.ppf(row_approx)

    return z


def transform_stnormal_normal_corr_gm17(z, cov, sample_Z_c, mu=None):
    """
    Takes sample from unit cube and transforms it to standard normally
    distributed sample with correlation structure that is implied by the
    provided covariance matrix.
    
    REMARK: The part that involves the upper matrix Q from the Cholesky
    decomposition of the correlation matrix of sample_Z_c seems unnecessary.
    It effectively does nothing because it is approx. an identity matrix.
    """
    if mu is None:
        mu = np.zeros(len(cov))
    else:
        pass

    # Step 2. Skipped transformation of correlation matrix for normally distributed paramters.
    R_z = covariance_to_correlation(cov)

    # Step 3: Perform Cholesky decomposition of (transformed) covariance matrix for
    # upper triangular matrix.
    M = linalg.cholesky(R_z, lower=False)

    # Step 4: Draw random vector from standard normal distribution for each paramter
    # and calculate the corresponding correlation matrix. Then do the same as in
    # Step 3 for the correlation matrix.
    C = np.corrcoef(sample_Z_c)
    Q = linalg.cholesky(C, lower=False)

    # Step 5: Derive the dependent normally distributed vector z_c = z*Q^(-1)*M.
    z_corr = np.ones(len(cov)).reshape(1, len(cov)) * np.nan
    zq = np.dot(z, linalg.inv(Q))
    z_corr = np.dot(zq, M)

    # Step 6: De-standardize.
    x = z_corr * np.sqrt(np.diag(cov)).reshape(1, len(cov)) + mu.reshape(1, len(cov))

    return x


def transform_stnormal_normal_corr_lemaire09(z, cov, mu=None):
    """
    Inverse Rosenblatt/Nataf transformation (from standard normal)
    to multivariate normal space with given correlations.
    Step 2) Compute correlation matrix.
    Step 3) Introduce dependenciec to standard normal sample.
    Step 4) De-standardize sample to normal space.
    
    """
    if mu is None:
        mu = np.zeros(len(cov))
    else:
        pass
    # Convert covariance matrix to correlation matrix
    C = covariance_to_correlation(cov) 
    Q_prime = linalg.cholesky(C, lower=True)
    
    z_corr_stnorm = np.dot(Q_prime, z.reshape(len(cov), 1))
    
    x_norm = z_corr_stnorm * np.sqrt(np.diag(cov)).reshape(len(cov), 1) + mu.reshape(len(cov), 1)

    return x_norm.T
