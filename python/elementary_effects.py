"""Functions to compute the elementary effects in Ge/Menendez (2017)."""

import numpy as np


from sampling_trajectory import morris_trajectory

from transform_reorder import ee_ind_reorder_trajectory
from transform_reorder import ee_full_reorder_trajectory
from transform_reorder import inverse_ee_ind_reorder_trajectory
from transform_reorder import inverse_ee_full_reorder_trajectory
from transform_reorder import reorder_mu
from transform_reorder import reorder_cov
from transform_reorder import inverse_reorder_mu
from transform_reorder import inverse_reorder_cov

from transform_distributions import transform_stnormal_normal_corr_lemaire09

from numpy.testing import assert_array_equal


def ee_ind_trajectories(sample_traj_list, mu, cov):
    """
    Sample two lists of transformed trajectories for the computation
    of the independent Elementary Effects. As explained in Ge/Menendez (2017),
    pages 33 and 34, the rows equal to T(p_{i}, i) and T(p_{i+1}, i),
    respectively.
    
    """
    assert len(mu) == len(cov) == np.size(sample_traj_list[0], 1)

    n_traj_sample = len(sample_traj_list)
    n_rows = np.size(sample_traj_list[0], 0)
    zero_idx_diff = []
    one_idx_diff = []
    
    # Transformation 1.
    for traj in range(0, n_traj_sample):
        zero_idx_diff.append(ee_ind_reorder_trajectory(sample_traj_list[traj], p_i_plus_one=False))
        one_idx_diff.append(ee_ind_reorder_trajectory(sample_traj_list[traj]))
    
    # Transformation 2 for p_i
    mu_zero = reorder_mu(mu)
    cov_zero = reorder_cov(cov)
    for traj in range(0, n_traj_sample):
        for row in range(0, n_rows):
            zero_idx_diff[traj][row, :] = transform_stnormal_normal_corr_lemaire09(
                    zero_idx_diff[traj][row, :], cov_zero, mu_zero)
            mu_zero = reorder_mu(mu_zero)
            cov_zero = reorder_cov(cov_zero)
    
    # Transformation 2 for p_{i+1}
    # No re-arrangement need as the first transformation for p_{i+1}
    # is using the original order of mu and cov.
    mu_one = mu
    cov_one = cov
    for traj in range(0, n_traj_sample):
        for row in range(0, n_rows):
            one_idx_diff[traj][row, :] = transform_stnormal_normal_corr_lemaire09(
                    one_idx_diff[traj][row, :], cov_one, mu_one)
            mu_one = reorder_mu(mu_one)
            cov_one = reorder_cov(cov_one)
    
    # Transformation 3.
    trans_zero = []
    trans_one = []
    for traj in range(0, n_traj_sample):
        trans_zero.append(inverse_ee_ind_reorder_trajectory(zero_idx_diff[traj], p_i_plus_one=False))
        trans_one.append(inverse_ee_ind_reorder_trajectory(one_idx_diff[traj]))
    
    return trans_zero, trans_one


"""As written in Ge/Menendez (2017), page 34: The elements in vectors T(p_{i}, i) and
T(p_{i+1}, i) are the same except of the ith element."""

mu = np.array([10, 11, 12, 13, 14])

cov = np.array(
    [
        [10, 0, 0, 2, 0.5],
        [0, 20, 0.4, 0.15, 0],
        [0, 0.4, 30, 0.05, 0],
        [2, 0.15, 0.05, 40, 0],
        [0.5, 0, 0, 0, 50],
    ]
)

n_traj_sample = 100
sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj_list.append(morris_trajectory(n_inputs=5, n_levels=6, seed=seed))

trans_zero, trans_one = ee_ind_trajectories(
        sample_traj_list, mu, cov)

for traj in range(0, len(trans_zero)):
    for row in range(0, np.size(trans_zero[0], 0) - 1):
        zero = np.delete(trans_zero[traj][row, :], row)
        one = np.delete(trans_one[traj][row + 1, :], row)
        assert_array_equal(zero, one)

