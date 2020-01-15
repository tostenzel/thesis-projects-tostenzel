"""Test sampling_trajectory.py"""
# Define parent folder as relative path.
import sys

sys.path.append("..")

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from sampling_trajectory import stepsize
from sampling_trajectory import morris_trajectory
from not_sampling_functions import elementary_effect_i
from sampling_trajectory import compute_trajectory_distance
from sampling_trajectory import distance_matrix
from sampling_trajectory import combi_wrapper
from sampling_trajectory import select_trajectories
from sampling_trajectory import campolongo_2007
from sampling_trajectory import intermediate_ge_menendez_2014


def test_morris_trajectories():
    """
    Can not account for proplems with the fixed random matrices/vectors/scalers.

    """
    expected = np.array([[1 / 3, 1], [1, 1], [1, 1 / 3]])
    assert_array_equal(
        expected,
        morris_trajectory(
            n_inputs=2, n_levels=4, step_function=stepsize, seed=123, test=True
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


def test_select_trajectories():
    test_traj_dist_matrix = np.array(
        [[0, 1, 2, 4], [1, 0, 3, 100], [2, 3, 0, 200], [4, 100, 200, 0]]
    )
    test_indices, test_select = select_trajectories(test_traj_dist_matrix, 3)

    expected_dist_indices = [1, 2, 3]
    expected_fourth_row = [1, 2, 3, np.sqrt(3 ** 2 + 100 ** 2 + 200 ** 2)]

    assert test_indices == expected_dist_indices
    assert_array_almost_equal(test_select[3, :], expected_fourth_row, 0.00000001)

def test_select_trajectories_2():
	dist_matrix = np.array([
		[0, 4, 5, 6],
		[4, 0, 7, 8],
		[5, 7, 0, 9],
		[6, 8, 9, 0]])

	exp_max_dist_indices = [1, 2, 3]

	exp_combi_distance = np.array([
		[0, 1, 2, np.sqrt(4**2 + 5**2 + 7**2)],
		[0, 1, 3, np.sqrt(4**2 + 6**2 + 8**2)],
		[0, 2, 3, np.sqrt(5**2 + 6**2 + 9**2)],
		[1, 2, 3, np.sqrt(7**2 + 8**2 + 9**2)]])

	max_dist_indices, combi_distance = select_trajectories(dist_matrix, 3)

	assert_array_equal(exp_max_dist_indices, max_dist_indices)
	assert_array_equal(exp_combi_distance, combi_distance)

def test_select_trajectories_3():
	dist_matrix = np.array([
		[0, 4, 5, 6],
		[4, 0, 7, 8],
		[5, 7, 0, 9],
		[6, 8, 9, 0]])

	exp_max_dist_indices = [2, 3]

	exp_combi_distance = np.array([
		[0, 1, np.sqrt(4**2)],
		[0, 2, np.sqrt(5**2)],
		[0, 3, np.sqrt(6**2)],
		[1, 2, np.sqrt(7**2)],
		[1, 3, np.sqrt(8**2)],
		[2, 3, np.sqrt(9**2)]])

	max_dist_indices, combi_distance = select_trajectories(dist_matrix, 2)

	assert_array_equal(exp_max_dist_indices, max_dist_indices)
	assert_array_equal(exp_combi_distance, combi_distance)
"""
def test_compare_camp_07_int_ge_men_14():
	n_inputs = 4
	n_levels = 11
	n_traj_sample = 50
	n_traj = 3


	sample_traj_list = list()
	for traj in range(0, n_traj_sample):
	    seed = 123 + traj

	    sample_traj_list.append(
	        morris_trajectory(n_inputs, n_levels, step_function=stepsize, seed=seed)
	    )
	    
	_, select_list, select_distance_matrix = campolongo_2007(sample_traj_list, n_traj)

	_, select_list_2, select_distance_matrix_2 = intermediate_ge_menendez_2014(sample_traj_list, n_traj)

	assert_array_equal(np.array(select_list), np.array(select_list_2))
	assert_array_equal(select_distance_matrix, select_distance_matrix_2)
"""