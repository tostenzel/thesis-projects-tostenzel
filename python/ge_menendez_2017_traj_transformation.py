"""
Ge/Menendez (2017)

REMARK: All one-dimensional arrays are taken as row vectors by numpy
except of that they are no transposed correctly but just mirrored.

"""
import random

import numpy as np
import scipy.linalg as linalg
from sampling_trajectory import morris_trajectory
from scipy.stats import norm


# Transformation 1: Shift the first \omega elements to the back to generate
# an independent vector.
def reorder_trajectory(traj, p_i_plus_one=False):
    traj_trans_one = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        # move FIRST w elements to the BACK
        if p_i_plus_one is False:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i + 1))
        if p_i_plus_one is True:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i))
    return traj_trans_one


# Transformation 3: Undo Transformation 1.
def reverse_reorder_trajectory(traj, p_i_plus_one=False):
    traj_trans_three = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        # move LAST w elements to the FRONT
        if p_i_plus_one is False:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i + 1)))
        if p_i_plus_one is True:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i)))
    return traj_trans_three


n_traj_sample = 20
sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj_list.append(morris_trajectory(n_inputs=5, n_levels=6))




traj = reorder_trajectory(sample_traj_list[0])

row_traj_reordered = traj[0,:]
#traj_trans_one = reorder_trajectory(traj)
#traj_trans_one_compare = reorder_trajectory(traj, p_i_plus_one=True)
#traj_trans_rev = reverse_reorder_trajectory(traj_trans_one)
#traj_trans_rev_compare = reverse_reorder_trajectory(
#    traj_trans_one_compare, p_i_plus_one=True
#)


def sample_stnormal_paramters(n_par, n_draws=100_000, seed=123):
    random.seed(seed)
    sample_stnormal_paramters = np.random.normal(0, 1, n_par * n_draws).reshape(
        n_par, n_draws
    )
    return sample_stnormal_paramters

def covariance_to_correlation(cov):
    """Converts covariance matrix to correlation matrix."""
    # Standard deviations of each variable.
    sd = np.sqrt(np.diag(cov)).reshape(1, len(cov))
    corr = cov / sd.T / sd
    
    return corr

def correlate_normalize_row(row_traj_reordered, cov, sample_Z_c):
    """
    Takes sample from unit cube and transforms it to standard normally
    distributed sample with correlation structure that is implied by the
    provided covariance matrix.
    
    REMARK: The part that involves the upper matrix Q from the Cholesky
    decomposition of the correlation matrix of sample_Z_c seems unnecessary.
    It effectively does nothing because it is approx. an identity matrix.
    """
    # Need to replace ones, because norm.ppf(1) = inf and zeros because norm.ppf(0) = -inf
    row_approx = np.where(row_traj_reordered == 1, 0.999, row_traj_reordered)
    row_approx = np.where(row_approx == 0, 0.001, row_approx)

    # Step 1: Inverse cdf of standard normal distribution (N(0, 5)).
    z = norm.ppf(row_approx)

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
    z_c = np.ones(row_traj_reordered.shape) * np.nan
    zq = np.dot(z, linalg.inv(Q))
    z_c = np.dot(zq, M)

    return z_c


def james_e_gentle_2006(row_traj_reordered, cov, expecation=0):
    """
    Remark 1: If the expectation is 0 and the variances equal 1 (see
    Remark 2) this method can do the same as steps 1 to steps 5 in 
    Ge/Menendez (2017) as implemented in `correlate_normalize_row` but simpler.
    It does it without the computation of a large normally
    distributed sample and the inverse of the upper triangular Cholesky
    matrix of it. Apart from evaluating the normal cdf at the sample from U(0,1),
    this method is equivalent to the inverse Rosenblatt and inverse
    the Nataf Transformation in case of the normal distribution.

    Remark 2: Correlation and Covariance are equal when the variance is
    normalized to one. Therefore, if the main diagonal of the covariace matrix
    is only ones, it does not matter which matrix to decompose
    (compare Rosenblatt and Nataf Transformation). However, this approach does
    also work for non-standard normal distributions with the covariance matrix
    by adding the expecation.

    Transforms a given sample from U(0,1) to multivariate normal space
    with given COVARIANCES as written on page 197 in Gentle (2006).

    """
    # Need to replace ones, because norm.ppf(1) = inf and zeros because norm.ppf(0) = -inf
    row_approx = np.where(row_traj_reordered == 1, 0.999, row_traj_reordered)
    row_approx = np.where(row_approx == 0, 0.001, row_approx)

    # Step 1: Inverse cdf of standard normal distribution (N(0, 1)).
    z = norm.ppf(row_approx) 
    
    # In contrary, Gentle uses the lower matrix from the Choleksy decomposition.
    # (Therefore, he also reverses the order of the matrix multiplication
    # compared to Ge/Menendez(2017). Therefore, its equivalent.)
    M_prime = linalg.cholesky(cov, lower=True)
    
    z_c = np.dot(M_prime,z) + expecation
    
    return z_c


def transform_uniform_stnormal_corr(row_traj_reordered, cov):
    """
    REMARK to understand some connections:
    Apart from evaluating the normal cdf at the sample from U(0,1), this function
    is equivalent to an inverse Rosenblatt and inverse Nataf transformation
    from uniform to dependent STANDARD normal space.
    This is outlined in Lemaire (2009), pp. 77 - 101. 
    It misses the transformation from standard normal to normal that multiplies
    the standard deviation and adds the expectation.
    It does not use the covariance directly as in Gentle (2006) because
    the variances from the input distribution do not necessarily equal 1.

    The aim in Ge/Menendez (2017) is to transform a sample from U(0,1) to
    standard normal space with given
    CORRELATION to directly apply another inverse Rosenblatt/Nataf transformation
    that then accounts for the expecatation and the standar deviation.
    Therefore, this function transforms the covariance matrix to a correlation
    matrix and sets the expectation to 0. This is (like) a standardization.

    """
    # Need to replace ones, because norm.ppf(1) = inf and zeros because norm.ppf(0) = -inf
    row_approx = np.where(row_traj_reordered == 1, 0.999, row_traj_reordered)
    row_approx = np.where(row_approx == 0, 0.001, row_approx)

    # Step 1: Inverse cdf of standard normal distribution (N(0, 1)).
    z = norm.ppf(row_approx) 
    
    # Convert covariance matrix to correlation matrix
    C = covariance_to_correlation(cov)

    # Use commutative ordering as in Gentle(2006).    
    Q_prime = linalg.cholesky(C, lower=True)
    z_c = np.dot(Q_prime,z)
    
    return z_c


def transform_uniform_st_normal_uncorr(row_traj_reordered):
    # Need to replace ones, because norm.ppf(1) = inf and zeros because norm.ppf(0) = -inf
    row_approx = np.where(row_traj_reordered == 1, 0.999, row_traj_reordered)
    row_approx = np.where(row_approx == 0, 0.001, row_approx)

    # Step 1: Inverse cdf of standard normal distribution (N(0, 1)).
    z = norm.ppf(row_approx)
    return z


def transform_stnormal_normal_corr(row_traj_reordered, cov, mu=None):
    """
    Inverse Rosenblatt/Nataf transformation (from standard normal)
    to multivariate normal space with given correlation.
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
    
    x_stnorm = np.dot(Q_prime, z_c.reshape(len(cov), 1))
    
    x_norm = x_stnorm * np.sqrt(np.diag(cov)).reshape(len(cov), 1) + mu.reshape(len(cov), 1)

    return x_norm

    
    
mu = np.array([10, 10, 10, 10, 10])

cov = np.array([
        [10,0,0,2,0.5],
        [0,20,0.4,0.15,0],
        [0,0.4,30,0.05,0],
        [2,0.15,0.05,40,0],
        [0.5,0,0,0,50]])
"""
cov = np.array([
        [1,0,0,0.2,0.5],
        [0,1,0.4,0.15,0],
        [0,0.4,1,0.05,0],
        [0.2,0.15,0.05,1,0],
        [0.5,0,0,0,1]])
"""
row_approx = np.array([0.1, 0.1, 0.2, 0.8, 0.5])

"""check"""
z_c = transform_uniform_stnormal_corr(row_approx, cov)
x = transform_stnormal_normal_corr(z_c, cov, mu)

from distributions import distributions
from nataf_transformation import nataf_transformation

M = list()
M.append(distributions('normal', 'PAR', [mu[0], np.sqrt(cov[0 ,0])]))
M.append(distributions('normal', 'PAR', [mu[1], np.sqrt(cov[1 ,1])]))
M.append(distributions('normal', 'PAR', [mu[2], np.sqrt(cov[2 ,2])]))
M.append(distributions('normal', 'PAR', [mu[3], np.sqrt(cov[3 ,3])]))
M.append(distributions('normal', 'PAR', [mu[4], np.sqrt(cov[4 ,4])]))
# Correlation matrix.
Rho = covariance_to_correlation(cov)
# Applying Nataf transformation
T_Nataf = nataf_transformation(M, Rho)
# Inverse Rosenblatt transformation.
# Transform sample from INDEPENDENT standard normal to DEPENDENT actual/physical space.
X = T_Nataf.U2X(z_c)


