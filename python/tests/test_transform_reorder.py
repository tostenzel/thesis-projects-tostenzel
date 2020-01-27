"""Test transform_reorder.py"""
import sys

# Define parent folder as relative path.
sys.path.append("python")

import numpy as np

from numpy.testing import assert_array_equal

from transform_reorder import reorder_trajectory
from transform_reorder import reverse_reorder_trajectory


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
