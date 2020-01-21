"""Ge/Menendez (2017)"""
import random

import numpy as np
import scipy.linalg as linalg
from sampling_trajectory import intermediate_ge_menendez_2014
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

    sample_traj_list.append(morris_trajectory(n_inputs=3, n_levels=6))

gm14_array, gm14_list, gm14_pairs_dist = intermediate_ge_menendez_2014(
    sample_traj_list, n_traj=5
)


traj = sample_traj_list[0]

traj_trans_one = reorder_trajectory(traj)
traj_trans_one_compare = reorder_trajectory(traj, p_i_plus_one=True)
traj_trans_rev = reverse_reorder_trajectory(traj_trans_one)
traj_trans_rev_compare = reverse_reorder_trajectory(
    traj_trans_one_compare, p_i_plus_one=True
)


"""Try tranformation 2"""
# Check in mu and cov.
cov = np.identity(3)
cov_1 = np.array([[1, 0.9, 0.4], [0.9, 1, 0.01], [0.4, 0.01, 1]])
mu = np.zeros(np.size(cov, 0))


def sample_stnormal_paramters(n_par, n_draws=100_000, seed=123):
    random.seed(seed)
    sample_stnormal_paramters = np.random.normal(0, 1, n_par * n_draws).reshape(
        n_par, n_draws
    )
    return sample_stnormal_paramters


sample_Z_c = sample_stnormal_paramters(np.size(cov, 0))


row_traj_reordered = traj_trans_one[0, :]


def inverse_nataf_transformation_normal(row_traj_reordered, mu, cov, sample_Z_c):
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

    # Step 6: Apply inverse Nataf transformation with Gaussian copula.
    # Almost equal to z if there are no correlations.
    cdf_z_c = norm.cdf(z_c)

    row_x = np.ones(row_traj_reordered.shape) * np.nan
    row_x = np.dot(mu + norm.ppf(cdf_z_c), np.sqrt(cov))

    return row_x


cov = np.identity(9)
mu = np.zeros(9)

row_traj_reordered = np.array([0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1])
expected = np.array(
    [-3.0902, -2.3263, -1.6449, -0.8416, 0.000, 0.8416, 1.6449, 2.3263, 3.0902]
)

sample_Z_c = sample_stnormal_paramters(np.size(cov, 0), n_draws=10_000_000)

row_x = inverse_nataf_transformation_normal(row_traj_reordered, mu, cov, sample_Z_c)
