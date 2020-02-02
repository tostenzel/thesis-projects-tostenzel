"""Tests select_sample_sets.py."""

import sys

# Define parent folder as relative path.
sys.path.append("..")

import numpy as np
import pytest

from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from sampling_trajectory import morris_trajectory
from select_sample_sets import compute_trajectory_distance
from select_sample_sets import distance_matrix
from select_sample_sets import combi_wrapper
from select_sample_sets import select_trajectories
from select_sample_sets import campolongo_2007
from select_sample_sets import intermediate_ge_menendez_2014
from select_sample_sets import select_trajectories_wrapper_iteration
from select_sample_sets import total_distance
from select_sample_sets import final_ge_menendez_2014

def test_compute_trajectory_distance():
    traj_0 = np.ones((3, 2))
    traj_1 = np.zeros((3, 2))
    assert 4 * np.sqrt(3) == compute_trajectory_distance(traj_0, traj_1)


def test_distance_matrix():
    traj_list = [np.ones((3, 2)), np.zeros((3, 2))]
    expected = np.array([[0, 4 * np.sqrt(3)], [4 * np.sqrt(3), 0]])
    assert_array_equal(expected, distance_matrix(traj_list))


def test_combi_wrapper():
    expected_0 = [[0]]
    expected_1 = [[0], [1]]
    expected_2 = [[0, 1]]
    expected_3 = [[0, 1], [0, 2], [1, 2]]
    expected_4 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert expected_0 == combi_wrapper([0], 1)
    assert expected_1 == combi_wrapper([0, 1], 1)
    assert expected_2 == combi_wrapper([0, 1], 2)
    assert expected_3 == combi_wrapper([0, 1, 2], 2)
    assert expected_4 == combi_wrapper([0, 1, 2, 3], 2)


def test_select_trajectories_1():
    """The difference between sample and selection size is not large enough for high trust."""
    test_traj_dist_matrix = np.array(
        [[0, 1, 2, 4], [1, 0, 3, 100], [2, 3, 0, 200], [4, 100, 200, 0]]
    )

    test_indices, test_select = select_trajectories(test_traj_dist_matrix, 3)

    expected_dist_indices = [1, 2, 3]
    expected_fourth_row = [1, 2, 3, np.sqrt(3 ** 2 + 100 ** 2 + 200 ** 2)]

    assert test_indices == expected_dist_indices
    assert_array_equal(test_select[3, :], expected_fourth_row)


def test_select_trajectories_2():
    """The difference between sample and selection size is not large enough for high trust."""
    dist_matrix = np.array([[0, 4, 5, 6], [4, 0, 7, 8], [5, 7, 0, 9], [6, 8, 9, 0]])

    exp_max_dist_indices = [1, 2, 3]

    exp_combi_distance = np.array(
        [
            [0, 1, 2, np.sqrt(4 ** 2 + 5 ** 2 + 7 ** 2)],
            [0, 1, 3, np.sqrt(4 ** 2 + 6 ** 2 + 8 ** 2)],
            [0, 2, 3, np.sqrt(5 ** 2 + 6 ** 2 + 9 ** 2)],
            [1, 2, 3, np.sqrt(7 ** 2 + 8 ** 2 + 9 ** 2)],
        ]
    )

    max_dist_indices, combi_distance = select_trajectories(dist_matrix, 3)

    assert_array_equal(exp_max_dist_indices, max_dist_indices)
    assert_array_equal(exp_combi_distance, combi_distance)


def test_select_trajectories_3():
    """The difference between sample and selection size is not large enough for high trust."""
    dist_matrix = np.array([[0, 4, 5, 6], [4, 0, 7, 8], [5, 7, 0, 9], [6, 8, 9, 0]])

    exp_max_dist_indices = [2, 3]

    exp_combi_distance = np.array(
        [
            [0, 1, np.sqrt(4 ** 2)],
            [0, 2, np.sqrt(5 ** 2)],
            [0, 3, np.sqrt(6 ** 2)],
            [1, 2, np.sqrt(7 ** 2)],
            [1, 3, np.sqrt(8 ** 2)],
            [2, 3, np.sqrt(9 ** 2)],
        ]
    )

    max_dist_indices, combi_distance = select_trajectories(dist_matrix, 2)

    assert_array_equal(exp_max_dist_indices, max_dist_indices)
    assert_array_equal(exp_combi_distance, combi_distance)


def test_select_trajectories_iteration_1():
    """The difference between sample and selection size is not large enough for high trust."""
    dist_matrix = np.array([[0, 4, 5, 6], [4, 0, 7, 8], [5, 7, 0, 9], [6, 8, 9, 0]])

    exp_max_dist_indices = [2, 3]

    # indices in the array below do not match the original dist_matrix.
    exp_combi_distance = np.array(
        [[0, 1, np.sqrt(7 ** 2)], [0, 2, np.sqrt(8 ** 2)], [1, 2, np.sqrt(9 ** 2)]]
    )

    max_dist_indices, combi_distance = select_trajectories_wrapper_iteration(
        dist_matrix, 2
    )

    assert_array_equal(exp_max_dist_indices, max_dist_indices)
    assert_array_equal(exp_combi_distance, combi_distance)


def test_select_trajectories_iteration_2():
    """The difference between sample and selection size is not large enough for high trust."""
    test_traj_dist_matrix = np.array(
        [[0, 1, 2, 4], [1, 0, 3, 100], [2, 3, 0, 200], [4, 100, 200, 0]]
    )

    max_dist_indices, _ = select_trajectories(test_traj_dist_matrix, 2)
    max_dist_indices_iter, _ = select_trajectories_wrapper_iteration(
        test_traj_dist_matrix, 2
    )

    assert_array_equal(max_dist_indices, max_dist_indices_iter)


@pytest.mark.skip(
    reason="The following behavior is expected by Ge/Menendez (2014). \
    Oftentimes the test works. \
    However, due to numerical reasons, sometimes intermediate_ge_menendez_2014 \
    selects a different, slightly worse trajectory set\
    compared to campolongo_2007."
)
def test_compare_camp_07_int_ge_men_14_1():
    """
    A share of times, the test failes because the path of combinations
    in the iteration in intermediate_ge_menendez_2014 slightly deviates from
    the optimal one. Yet, the total distance of the given combinations
    are relative close.

    """
    n_inputs = 4
    n_levels = 1
    n_traj_sample = 50
    n_traj = 5

    sample_traj_list = list()
    for traj in range(0, n_traj_sample):
        seed = 123 + traj

        m_traj = morris_trajectory(n_inputs, n_levels, seed=seed)
        sample_traj_list.append(m_traj)

    _, select_list, select_distance_matrix = campolongo_2007(sample_traj_list, n_traj)
    _, select_list_2, select_distance_matrix_2 = intermediate_ge_menendez_2014(
        sample_traj_list, n_traj
    )

    assert_array_equal(np.array(select_list), np.array(select_list_2))
    assert_array_equal(select_distance_matrix, select_distance_matrix_2)


def test_compare_camp_07_int_ge_men_14_2():
    """
    Tests wether the trajectory set computed by compolongo_2007
    and intermediate_ge_menendez_2014 are reasonably close in terms
    of their total distance.

    """
    n_inputs = 4
    n_levels = 10
    n_traj_sample = 30
    n_traj = 5

    sample_traj_list = list()
    for traj in range(0, n_traj_sample):
        seed = 123 + traj

        m_traj, _ = morris_trajectory(n_inputs, n_levels, seed=seed)
        sample_traj_list.append(m_traj)

    _, select_list, select_distance_matrix = campolongo_2007(sample_traj_list, n_traj)
    _, select_list_2, select_distance_matrix_2 = intermediate_ge_menendez_2014(
        sample_traj_list, n_traj
    )

    dist_camp = total_distance(select_distance_matrix)
    dist_gm = total_distance(select_distance_matrix_2)

    assert dist_camp - dist_gm < 0.03 * dist_camp


@pytest.mark.skip(
    reason="The following behavior is expected by Ge/Menendez (2014). \
    Oftentimes the test works. \
    However, due to numerical reasons, sometimes intermediate_ge_menendez_2014 \
    selects a different, slightly worse trajectory set\
    compared to campolongo_2007."
)
def test_compare_camp_07_final_ge_men_14_1():
    n_inputs = 4
    n_levels = 10
    n_traj_sample = 30
    n_traj = 5

    sample_traj_list = list()
    for traj in range(0, n_traj_sample):
        seed = 123 + traj

        m_traj, _ = morris_trajectory(n_inputs, n_levels, seed=seed)
        sample_traj_list.append(m_traj)

    traj_array, traj_list, diagonal_dist_matrix = final_ge_menendez_2014(
        sample_traj_list, n_traj
    )
    test_array, test_list, test_diagonal_dist_matrix = intermediate_ge_menendez_2014(
        sample_traj_list, n_traj
    )

    assert_array_equal(traj_array, test_array)
    assert_array_equal(traj_list, test_list)
    assert_array_equal(diagonal_dist_matrix, test_diagonal_dist_matrix)


def test_compare_camp_07_final_ge_men_14_2():
    """
    Tests wether the trajectory set computed by compolongo_2007
    and final_ge_menendez_2014 are reasonably close in terms
    of their total distance.

    """
    n_inputs = 4
    n_levels = 10
    n_traj_sample = 30
    n_traj = 5

    sample_traj_list = list()
    for traj in range(0, n_traj_sample):
        seed = 123 + traj

        m_traj, _ = morris_trajectory(n_inputs, n_levels, seed=seed)
        sample_traj_list.append(m_traj)

    _, select_list, select_distance_matrix = campolongo_2007(sample_traj_list, n_traj)
    _, select_list_2, select_distance_matrix_2 = final_ge_menendez_2014(
        sample_traj_list, n_traj
    )

    dist_camp = total_distance(select_distance_matrix)
    dist_gm = total_distance(select_distance_matrix_2)

    assert dist_camp - dist_gm < 0.4 * dist_camp
