"""Test ge_menendez_2017.py"""
# Define parent folder as relative path.
import sys

sys.path.append("..")

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


from ge_menendez_2017 import transformation_one
from ge_menendez_2017 import transformation_three


def test_transformations():
    traj = np.array([[0, 0, 0], [1, 0, 0], [2, 3, 0], [4, 5, 6]])

    assert_array_equal(
        transformation_one(traj, p_i_plus_one=True),
        np.array([[0, 0, 0], [0, 0, 1], [0, 2, 3], [4, 5, 6]]),
    )

    assert_array_equal(
        traj,
        transformation_three(
            transformation_one(traj, p_i_plus_one=True), p_i_plus_one=True
        ),
    )

    assert_array_equal(
        transformation_one(traj, p_i_plus_one=False),
        np.array([[0, 0, 0], [0, 1, 0], [2, 3, 0], [5, 6, 4]]),
    )

    assert_array_equal(
        traj,
        transformation_three(
            transformation_one(traj, p_i_plus_one=False), p_i_plus_one=False
        ),
    )