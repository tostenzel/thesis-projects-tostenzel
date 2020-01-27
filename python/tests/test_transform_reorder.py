"""Test transform_reorder.py"""
import sys

# Define parent folder as relative path.
sys.path.append("python")

import numpy as np

from numpy.testing import assert_array_equal

from transform_reorder import ee_full_reorder_trajectory
from transform_reorder import inverse_ee_full_reorder_trajectory
from transform_reorder import ee_ind_reorder_trajectory
from transform_reorder import inverse_ee_ind_reorder_trajectory
from transform_reorder import reorder_mu
from transform_reorder import reorder_cov


def test_ee_ind_reorder_trajectory():
    traj = np.array([[0, 0, 0], [1, 0, 0], [2, 3, 0], [4, 5, 6]])

    assert_array_equal(
        ee_ind_reorder_trajectory(traj),
        np.array([[0, 0, 0], [0, 0, 1], [0, 2, 3], [4, 5, 6]]),
    )

    assert_array_equal(
        traj, inverse_ee_ind_reorder_trajectory(ee_ind_reorder_trajectory(traj))
    )

    assert_array_equal(
        ee_ind_reorder_trajectory(traj, p_i_plus_one=False),
        np.array([[0, 0, 0], [0, 1, 0], [2, 3, 0], [5, 6, 4]]),
    )

    assert_array_equal(
        traj,
        inverse_ee_ind_reorder_trajectory(
            ee_ind_reorder_trajectory(traj, p_i_plus_one=False), p_i_plus_one=False
        ),
    )


def test_ee_full_reorder_trajectory():
    traj = np.array([[0, 0, 0], [1, 0, 0], [2, 3, 0], [4, 5, 6]])

    assert_array_equal(
        ee_full_reorder_trajectory(traj),
        np.array([[0, 0, 0], [1, 0, 0], [3, 0, 2], [6, 4, 5]]),
    )

    assert_array_equal(
        traj, inverse_ee_full_reorder_trajectory(ee_full_reorder_trajectory(traj))
    )

    assert_array_equal(
        ee_full_reorder_trajectory(traj, p_i_plus_one=False),
        np.array([[0, 0, 0], [0, 0, 1], [0, 2, 3], [4, 5, 6]]),
    )

    assert_array_equal(
        traj,
        inverse_ee_full_reorder_trajectory(
            ee_full_reorder_trajectory(traj, p_i_plus_one=False), p_i_plus_one=False
        ),
    )


def test_reorder_mu():
    mu = np.arange(10)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    assert_array_equal(expected, reorder_mu(mu))


def test_reorder_cov():
    cov = np.array(
        [
            [10, 2, 3, 4, 5],
            [2, 20, 6, 7, 8],
            [3, 6, 30, 9, 10],
            [4, 7, 9, 40, 11],
            [5, 8, 10, 11, 50],
        ]
    )
    expected = np.array(
        [
            [20, 6, 7, 8, 2],
            [6, 30, 9, 10, 3],
            [7, 9, 40, 11, 4],
            [8, 10, 11, 50, 5],
            [2, 3, 4, 5, 10],
        ]
    )
    assert_array_equal(expected, reorder_cov(cov))
