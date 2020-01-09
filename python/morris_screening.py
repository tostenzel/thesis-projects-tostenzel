"""Morris Screening"""
import random

import numpy as np
from scipy.special import binom


def stepsize(n_levels):
    """Leads to relatively equiprobable sampling from the input distribution"""
    return n_levels / (2 * (n_levels - 1))


def morris_trajectories(n_inputs, n_levels, seed=123, test=False):
    """Returns n parameter vectors, Dim n x Theta.
    n is also Theta+1."""
    np.random.seed(seed)
    step = stepsize(n_levels)
    #  B is (p+1)*p strictily lower triangular matrix of ones.
    B = np.tril(np.ones([n_inputs + 1, n_inputs]), -1)
    # J is (p+1)*p matrix of ones.
    J = np.ones([n_inputs + 1, n_inputs])
    # Matrix of zeros with values radom choices between -1 and 1 on main
    # diagonal.
    if test is True:
        base_value_vector_rand = [1 / 3] * 2
        P_star_rand = np.identity(n_inputs)
        D_star_rand = np.array([[1, 0], [0, -1]])
    else:
        # Choose a random value from the differenent Elementary Effects.
        # Must be lower than 1 - step otherwise one could sample values > 1.
        # base_value rand \in [i/(1-step)]
        value_grid = [0, 1 - step]
        idx = 1
        while idx / (n_levels - 1) < 1 - step:
            value_grid.append(idx / idx / (n_levels - 1))
            idx = idx + 1
        base_value_vector_rand = np.array(random.choices(value_grid, k=n_inputs))
        # Influenced by seed. Seed 123 generates P_star in book.
        P_star_rand = np.identity(n_inputs)
        # Shuffle columns
        np.random.shuffle(P_star_rand.T)
        D_star_rand = np.zeros([n_inputs, n_inputs])
        np.fill_diagonal(D_star_rand, random.choices([-1, 1], k=n_inputs))
    # Be careful with np.dot vs. np.matmul vs. *.
    B_star_rand = np.dot(
        J * base_value_vector_rand
        + (step / 2) * (np.dot((2 * B - J), D_star_rand) + J),
        P_star_rand,
    )
    return B_star_rand


def elementary_effect_i(model, i_python, init_input_pars, stepsize):
    vector_e = np.zeros(len(init_input_pars))
    vector_e[i_python] = 1
    step_input_pars = init_input_pars + (vector_e * stepsize)

    return (model(*step_input_pars.tolist()) - model(*init_input_pars)) / stepsize


def scaled_elementary_effect_i(
    model, i_python, init_input_pars, stepsize, sd_i, sd_model
):
    """Scales EE by (SD_i / SD_M)"""
    ee_i = elementary_effect_i(model, i_python, init_input_pars, stepsize)

    return ee_i * (sd_i / sd_model)


def compute_trajectory_distance(traj_0, traj_1):
    """
    Computes the sum of the root of the square distance between each
    parameter vector of one trajectory to each vector of the other trajectory.
    Trajectories are np.Arrays with iterations as rows and parameters as vectors.
    
    """
    distance = 0

    assert np.size(traj_0, 0) == np.size(traj_0, 1) + 1
    assert traj_0.shape == traj_1.shape
    if np.any(np.not_equal(traj_0, traj_1)):
        for col_0 in range(0, np.size(traj_0, 1)):
            for col_1 in range(0, np.size(traj_1, 1)):
                distance += np.sqrt(sum((traj_0[:, col_0] - traj_1[:, col_1]) ** 2))
    else:
        pass

    return distance


def distance_matrix(trajectory_list):
    """Compute distance between each pair of trajectories.
    Return symmetric matrix.
    """
    distance_matrix = np.nan * np.ones(
        shape=(len(trajectory_list), len(trajectory_list))
    )
    for i in range(0, len(trajectory_list)):
        for j in range(0, len(trajectory_list)):
            distance_matrix[i, j] = compute_trajectory_distance(
                trajectory_list[i], trajectory_list[j]
            )
    return distance_matrix


def combinations(iterable, r):
    """
    Takes list of elements and returns list of list
    of all possible r combinations regardless the order. 
    E.g. combinations(range(4), 3) returns
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]].
    The order matters!!!
    Taken from
    https://docs.python.org/2/library/itertools.html#itertools.combinations
    with slight modifications to update syntax and return lists.


    """
    pool = list(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield list(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield list(pool[i] for i in indices)


def select_trajectories(traj_dist_matrix, n_traj):
    """
    Convert symmetric matrix of distances for each trajectory pair
    to matrix with rows for each combination of n_traj combination of pairs.
    The distance_matrix are in [row,:n_traj] and the root-squared sum is in
    [row,n_traj+1]
    Also returns list of indices for traj_dist list to select the optimal trajs.
    """
    # Get all possible combinations of input parameters by their indices.
    # Unfortunatley returned as tuples.
    combi = list(combinations(np.arange(0, np.size(traj_dist_matrix, 1)), n_traj))
    # Convert list of tuples to list of lists.
    assert np.all(np.abs(traj_dist_matrix - traj_dist_matrix.T) < 1e-8)
    assert len(combi) == binom(np.size(traj_dist_matrix, 1), n_traj)
    combi_distance = np.ones([len(combi), n_traj + 1]) * np.nan
    combi_distance[:, 0:n_traj] = np.array(combi)

    # This loop needs to be parallelized.
    for row in range(0, len(combi)):
        combi_distance[row, n_traj] = 0
        pair_combi = list(combinations(combi[row], 2))
        for pair in pair_combi:

            combi_distance[row, n_traj] += (
                traj_dist_matrix[int(pair[0])][int(pair[1])] ** 2
            )
    combi_distance[:, n_traj] = np.sqrt(combi_distance[:, n_traj])
    # indices of combination that yields highest distance figure
    #
    max_dist_indices_row = combi_distance[:, n_traj].argsort()[-1:][::-1].tolist()
    max_dist_indices = combi_distance[max_dist_indices_row, 0:n_traj]
    # Convert list of float indices to list of ints.
    max_dist_indices = [int(i) for i in max_dist_indices.tolist()[0]]

    return max_dist_indices, combi_distance


def sobol_model(input_pars, coeffs_a):
    """
     - Tested by comparing graphs for 3 specifications to book.
    Arguments are lists. Strongly nonlinear, nonmonotonic, and nonzero interactions.
    Analytic results for Sobol Indices.

    """
    assert len(input_pars) == len(coeffs_a)

    def g_i(input_par_i, coeffs_a_i):
        return (abs(4 * input_par_i - 2) + coeffs_a_i) / (1 + coeffs_a_i)

    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs_a[i])
    return y


# Campolongo_2007
def campolongo_2007(n_inputs, n_levels, n_traj_sample, n_traj):
    """Takes number of input parametes, samplesize of trajectories,
    and selected number of trajectories as arguments.
    Returns an array with n_inputs at the verical and n_traj at the
    horizontal axis.
    """
    sample_traj = list()
    for traj in range(0, n_traj_sample):
        seed = 123 + traj

        sample_traj.append(morris_trajectories(n_inputs, n_levels, seed=seed))
    pair_matrix = distance_matrix(sample_traj)
    select_indices, dist_matrix = select_trajectories(pair_matrix, n_traj)

    select_trajs = [sample_traj[idx] for idx in select_indices]
    # rows are parameters, cols is number of drawn parameter vectors.
    input_par_array = np.vstack(select_trajs)

    return input_par_array.T


input_par_array = campolongo_2007(n_inputs=27, n_levels=6, n_traj_sample=20, n_traj=4)

import itertools


new_list = input_par_array.reshape(-1, 1).tolist()
merged = list(itertools.chain.from_iterable(new_list))

import matplotlib.pyplot as plt

plt.hist(merged, range=[-0.3, 1.3])
# a = [78, 12, 0.5, 2, 97, 33]

# function_evals = list()

# for i in range(0, np.size(input_par_array, 1)):
# function_evals.append(sobol_model(input_par_array[:,i], a))

# test = np.ones([5, 6])

# np.tril(test, -1)
