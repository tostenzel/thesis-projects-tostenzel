"""Ge/Menendez (2017)"""
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



def correlate_normalize_row(row_traj_reordered, cov, sample_Z_c):
    """
    Takes sample from unit cube and transforms it to standard normally
    distributed sample with correlation structure that is implied by the
    provided covariance matrix.
    
    REMARK: The part that involves the upper matrix Q from the Cholesky
    decomposition of the correlation matrix of sample_Z_c seems unnecessary.
    It effectively does nothing because it is approx. a identity matrix.
    """
    # Need to replace ones, because norm.ppf(1) = inf and zeros because norm.ppf(0) = -inf
    row_approx = np.where(row_traj_reordered == 1, 0.999, row_traj_reordered)
    row_approx = np.where(row_approx == 0, 0.001, row_approx)

    # Step 1: Inverse cdf of standard normal distribution (N(0, 5)).
    z = norm.ppf(row_approx)

    # Step 2. Skipped transformation of covariance matrix for normally distributed paramters.
    R_z = cov

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


def james_e_gentle_2005(row_traj_reordered, cov):
    """
    Remark 1: This method does the same as steps 1 to steps 5 in Ge/Menendez
    but simpler. It does it without the computation of a large normally
    distributed sample and the inverse of the upper triangular Cholesky
    matrix of it. This method is equivalent to the Rosenblatt and also
    the Nataf Transformation.
    Remark 2: Correlation and Covariance are equal when the variance is
    normalized to one. Therefore it does not matter which matrix to decompose
    (compare Rosenblatt and Nataf Transformation).

    Method to sample from multivariate normal (with mean zero)
    on page 197 in Gentle (1943).
    
    """
    # Need to replace ones, because norm.ppf(1) = inf and zeros because norm.ppf(0) = -inf
    row_approx = np.where(row_traj_reordered == 1, 0.999, row_traj_reordered)
    row_approx = np.where(row_approx == 0, 0.001, row_approx)

    # Step 1: Inverse cdf of standard normal distribution (N(0, 5)).
    z = norm.ppf(row_approx) 
    
    # In contrary, Gentle uses the lower matrix from the Choleksy decomposition.
    M_prime = linalg.cholesky(cov, lower=True)
    
    z_c = np.dot(M_prime,z)
    
    return z_c



cov = np.array([
        [1,0,0,0.2,0.5],
        [0,1,0.4,0.15,0],
        [0,0.4,1,0.05,0],
        [0.2,0.15,0.05,1,0],
        [0.5,0,0,0,1]])

sample_Z_c = sample_stnormal_paramters(5, 100_000)
row_approx = np.array([0.1, 0.1, 0.2, 0.8, 0.5])

"""check"""
check = james_e_gentle_2005(row_approx, cov)

gm17 = correlate_normalize_row(row_approx, cov, sample_Z_c)
