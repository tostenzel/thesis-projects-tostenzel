"""Sandbox."""

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

mu = np.array([10, 11, 12, 13, 14])

# Covariance matrix.
cov = np.array(
    [
        [10, 0, 0, 2, 0.5],
        [0, 20, 0.4, 0.15, 0],
        [0, 0.4, 30, 0.05, 0],
        [2, 0.15, 0.05, 40, 0],
        [0.5, 0, 0, 0, 50],
    ]
)



n_inputs=5
n_rows = n_inputs + 1
n_levels=6
n_traj_sample = 1
sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj_list.append(morris_trajectory(n_inputs=5, n_levels=6))

zero_idx_diff = []
one_idx_diff = []

for traj in range(0, n_traj_sample):
    zero_idx_diff.append(ee_ind_reorder_trajectory(sample_traj_list[traj], p_i_plus_one=False))
    one_idx_diff.append(ee_ind_reorder_trajectory(sample_traj_list[traj]))


mu_zero = reorder_mu(mu)
cov_zero = reorder_cov(cov)
for traj in range(0, n_traj_sample):
    for row in range(0, n_rows):
        zero_idx_diff[traj][row, :] = transform_stnormal_normal_corr_lemaire09(
                zero_idx_diff[traj][row, :], cov_zero, mu_zero)
        mu_zero = reorder_mu(mu_zero)
        cov_zero = reorder_cov(cov_zero)


mu_one = mu
cov_one = cov
for traj in range(0, n_traj_sample):
    for row in range(0, n_rows):
        one_idx_diff[traj][row, :] = transform_stnormal_normal_corr_lemaire09(
                one_idx_diff[traj][row, :], cov_one, mu_one)
        mu_one = reorder_mu(mu_one)
        cov_one = reorder_cov(cov_one)

trans_zero = []
trans_one = []
for traj in range(0, n_traj_sample):
    trans_zero.append(inverse_ee_ind_reorder_trajectory(zero_idx_diff[traj], p_i_plus_one=False))
    trans_one.append(inverse_ee_ind_reorder_trajectory(one_idx_diff[traj]))




        





    
"""
traj = sample_traj_list[0]

traj_trans_one = ee_ind_reorder_trajectory(traj)
traj_trans_one_subtract = ee_ind_reorder_trajectory(traj, p_i_plus_one=False)
traj_trans_rev = reverse_ee_ind_reorder_trajectory(traj_trans_one)
traj_trans_rev_subtract = reverse_ee_ind_reorder_trajectory(traj_trans_one, p_i_plus_one=False)


full = ee_full_reorder_trajectory(traj)
full_subtract = ee_full_reorder_trajectory(traj, p_i_plus_one=False)

rev_full = reverse_ee_full_reorder_trajectory(full)
rev_full_subtract = reverse_ee_full_reorder_trajectory(full, p_i_plus_one=True)
"""
