"""Tests elementary_effects.py"""
import sys

# Define parent folder as relative path.
sys.path.append("python")

import numpy as np

from numpy.testing import assert_array_equal

from sampling_trajectory import morris_trajectory
from transform_traj_elementary_effects import trans_ee_ind_trajectories


def test_trans_ee_ind_trajectories():
    """
    As written in Ge/Menendez (2017), page 34: The elements in vectors T(p_{i}, i) and
    T(p_{i+1}, i) are the same except of the ith element.

    """
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
    
    n_inputs = 5
    n_levels = 10
    n_traj_sample = 10
    sample_traj_list = list()
    for traj in range(0, n_traj_sample):
        seed = 123 + traj

        m_traj, _ = morris_trajectory(n_inputs, n_levels, seed=seed)
        sample_traj_list.append(m_traj)

    trans_zero, trans_one = trans_ee_ind_trajectories(sample_traj_list, cov, mu)

    for traj in range(0, len(trans_zero)):
        for row in range(0, np.size(trans_zero[0], 0) - 1):
            zero = np.delete(trans_zero[traj][row, :], row)
            one = np.delete(trans_one[traj][row + 1, :], row)
            assert_array_equal(zero, one)
