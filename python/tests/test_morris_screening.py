"""Test morris_screening.py"""
# Define parent folder as relative path.
import sys

sys.path.append("..")

import numpy as np


# Delete later
from scipy.special import binom

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from morris_screening import morris_trajectories
from morris_screening import elementary_effect_i
from morris_screening import compute_trajectory_distance
from morris_screening import distance_matrix
from morris_screening import combinations
from morris_screening import select_trajectories


def test_morris_trajectories():
    """Can not account for proplems with the fixed random matrices/vectors/scalers.
    """
    expected = np.array([[1 / 3, 1], [1, 1], [1, 1 / 3]])
    assert_array_equal(
        expected, morris_trajectories(n_inputs=2, stepsize=2 / 3, seed=123, test=True)
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


"""test"""
test_traj_dist_matrix = np.array(
    [[0, 1, 2, 4], [1, 0, 3, 100], [2, 3, 0, 200], [4, 100, 200, 0]]
)
test_indices, test_select = select_trajectories(test_traj_dist_matrix, 3)

traj_dist_matrix = test_traj_dist_matrix
n_traj = 3


combi = list(combinations(np.arange(0, np.size(traj_dist_matrix, 1)), n_traj))
# Convert list of tuples to list of lists.
assert np.all(np.abs(traj_dist_matrix - traj_dist_matrix.T) < 1e-8)
assert len(combi) == binom(np.size(traj_dist_matrix, 1), n_traj)
combi_distance = np.ones([len(combi), n_traj + 1]) * np.nan
combi_distance[:, 0:n_traj] = np.array(combi)

for row in range(0, len(combi)):
    combi_distance[row, n_traj] = 0
    pair_combi = list(combinations(combi[row], 2))
    for pair in pair_combi:

        combi_distance[row, n_traj] += traj_dist_matrix[int(pair[0])][int(pair[1])] ** 2
combi_distance[:, n_traj] = np.sqrt(combi_distance[:, n_traj])
# indices of combination that yields highest distance figure
#
max_dist_indices_row = combi_distance[:, n_traj].argsort()[-1:][::-1].tolist()
max_dist_indices = combi_distance[max_dist_indices_row, 0:n_traj]
# Convert list of float indices to list of ints.
max_dist_indices = [int(i) for i in max_dist_indices.tolist()[0]]
