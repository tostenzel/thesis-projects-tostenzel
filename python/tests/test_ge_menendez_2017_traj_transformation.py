"""Test ge_menendez_2017.py"""
import sys

# Define parent folder as relative path.
sys.path.append("python")

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from ge_menendez_2017_traj_transformation import reorder_trajectory
from ge_menendez_2017_traj_transformation import reverse_reorder_trajectory
from ge_menendez_2017_traj_transformation import sample_stnormal_paramters
from ge_menendez_2017_traj_transformation import inverse_nataf_transformation_normal
from ge_menendez_2017_traj_transformation import sample_stnormal_paramters
from ge_menendez_2017_traj_transformation import correlate_normalize_row
from ge_menendez_2017_traj_transformation import james_e_gentle_2005

def test_transformations():
    traj = np.array([[0, 0, 0], [1, 0, 0], [2, 3, 0], [4, 5, 6]])

    assert_array_equal(
        reorder_trajectory(traj, p_i_plus_one=True),
        np.array([[0, 0, 0], [0, 0, 1], [0, 2, 3], [4, 5, 6]]),
    )

    assert_array_equal(
        traj,
        reverse_reorder_trajectory(
            reorder_trajectory(traj, p_i_plus_one=True), p_i_plus_one=True
        ),
    )

    assert_array_equal(
        reorder_trajectory(traj, p_i_plus_one=False),
        np.array([[0, 0, 0], [0, 1, 0], [2, 3, 0], [5, 6, 4]]),
    )

    assert_array_equal(
        traj,
        reverse_reorder_trajectory(
            reorder_trajectory(traj, p_i_plus_one=False), p_i_plus_one=False
        ),
    )


def test_inverse_nataf_transformation_normal_uncorrelated():
    """Tests only a vector of uncorrelated paramters."""
    cov = np.identity(9)
    mu = np.zeros(9)

    row_traj_reordered = np.array([0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1])
    expected = np.array(
        [-3.0902, -2.3263, -1.6449, -0.8416, 0.000, 0.8416, 1.6449, 2.3263, 3.0902]
    )

    sample_Z_c = sample_stnormal_paramters(np.size(cov, 0), n_draws=10_000_000)

    row_x = inverse_nataf_transformation_normal(row_traj_reordered, mu, cov, sample_Z_c)

    assert_array_almost_equal(expected, row_x, 0.01)

def test_correlate_normalize_row():
    row_approx = np.array([0.1, 0.1, 0.2, 0.8, 0.5])
    cov = np.array([
        [1,0,0,0.2,0.5],
        [0,1,0.4,0.15,0],
        [0,0.4,1,0.05,0],
        [0.2,0.15,0.05,1,0],
        [0.5,0,0,0,1]])

    sample_Z_c = sample_stnormal_paramters(5, 100_000)

    expected = james_e_gentle_2005(row_approx, cov)
    gm17 = correlate_normalize_row(row_approx, cov, sample_Z_c)
    assert_array_almost_equal(gm17, expected, 0.01)
