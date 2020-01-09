"""Test morris_screening.py"""
# Define parent folder as relative path.
import sys

sys.path.append("..")
import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from morris_screening import morris_trajectories
from morris_screening import elementary_effect_i
from morris_screening import compute_trajectory_distance
from morris_screening import distance_matrix
from morris_screening import combinations
from morris_screening import select_trajectories


def test_morris_trajectories():
    init_input_pars = np.array([1 / 3, 1 / 3])
    stepsize = 2 / 3
    expected = np.array([[1, 1 / 3], [1, 1], [1 / 3, 1]])
    assert_array_equal(
        expected,
        morris_trajectories(
            init_input_pars, stepsize, seed=123, test_D_star_rand_2dim=True
        ),
    )


def lin_portfolio(q1, q2, c1=2, c2=1, *args):
    return c1 * q1 + c2 * q2


def test_elemtary_effect_i():
    assert 2 == round(
        elementary_effect_i(lin_portfolio, 0, [0.5, 1], stepsize=2 / 3), 10
    )

    assert 1 == round(
        elementary_effect_i(lin_portfolio, 1, [0.5, 1], stepsize=2 / 3), 10
    )


def test_compute_trajectory_distance():
    traj_0 = np.ones((3, 2))
    traj_1 = np.zeros((3, 2))
    assert 4 * np.sqrt(3) == compute_trajectory_distance(traj_0, traj_1)


def test_distance_matrix():
    traj_list = [np.ones((3, 2)), np.zeros((3, 2))]
    expected = np.array([[0, 4 * np.sqrt(3)], [4 * np.sqrt(3), 0]])
    assert_array_equal(expected, distance_matrix(traj_list))


def test_combinations():
    expected_0 = [[0]]
    expected_1 = [[0], [1]]
    expected_2 = [[0, 1]]
    expected_3 = [[0, 1], [0, 2], [1, 2]]
    expected_4 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert expected_0 == list(combinations([0], 1))
    assert expected_1 == list(combinations([0, 1], 1))
    assert expected_2 == list(combinations([0, 1], 2))
    assert expected_3 == list(combinations([0, 1, 2], 2))
    assert expected_4 == list(combinations([0, 1, 2, 3], 2))


def test_select_trajectories():
    test_traj_dist_matrix = np.array(
        [[0, 1, 2, 4], [1, 0, 3, 100], [2, 3, 0, 200], [4, 100, 200, 0]]
    )
    test_indices, test_select = select_trajectories(test_traj_dist_matrix, 3)

    expected_dist_indices = [1, 2, 3]
    expected_fourth_row = [1, 2, 3, np.sqrt(3 ** 2 + 100 ** 2 + 200 ** 2)]

    assert test_indices == expected_dist_indices
    assert_array_almost_equal(test_select[3, :], expected_fourth_row, 0.00000001)
