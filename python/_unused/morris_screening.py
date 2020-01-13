"""Winding stairs sampling + Morris(1991) improvement  + Campolongo(2007) improvement"""
import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom


def stepsize(n_levels):
    """
    Book recommendation:
    Leads not to equiprobable sampling from the input distribution.
    and is HALFWAY EQUISPACED.

    """
    return n_levels / (2 * (n_levels - 1))


def stepsize_equidistant(n_levels):
    """
    Leads to concentration at middle-sized values in the sample.
    The reason is the following:
    Imagine the levels are linspace(0,1,5).
    Also imagine the first parameter starts as zero. Then stepsize 0.2 is added.
    This yields to an additional number of 0.2s.
    The number of additional 0.2 in this case equals the number of paramters.
    This can not happen to 0.0s. If the stepsize does not yield repetitive
    values, there is no equidistance.

    """
    return 1 / (n_levels - 1)


def morris_trajectories(
    n_inputs, n_levels, step_function, stairs=True, seed=123, test=False
):
    """
    Returns n parameter vectors, Dim n x Theta.
    n is also Theta+1.
    Uses stepsize function.

    IMPORTANT:
    - Shuffling of identity matrix is turned off to obtain stairs
    for the Qiao We / Menendez (2016) paper for correlated paramters.
    - The levels have not to be equidistant because the first level after zero
    is the step. Thereafter, the levels are equispaced. Therefore, one may want
    to adjust the stepsize to the number of levels by hand instead of following
    the above function.
    - If the stepsize equals the distance between the other levels,
    then level 0 is less frequent.

    """
    np.random.seed(seed)
    step = step_function(n_levels)
    #  B is (p+1)*p strictily lower triangular matrix of ones.
    B = np.tril(np.ones([n_inputs + 1, n_inputs]), -1)
    # J is (p+1)*p matrix of ones.
    J = np.ones([n_inputs + 1, n_inputs])
    # Matrix of zeros with random choices between -1 and 1 on main
    # diagonal.
    if test is False:
        # Choose a random value from the differenent Elementary Effects.
        # Must be lower than 1 - step otherwise one could sample values > 1.
        # base_value rand \in [i/(1-step)]
        # !!!The upper bound must be 1 - step, because step is added on top of the
        # values!!!
        value_grid = [0, 1 - step]
        idx = 1
        while idx / (n_levels - 1) < 1 - step:
            value_grid.append(idx / idx / (n_levels - 1))
            idx = idx + 1
        base_value_vector_rand = np.array(random.choices(value_grid, k=n_inputs))
        # Influenced by seed.
        P_star_rand = np.identity(n_inputs)
        D_star_rand = np.zeros([n_inputs, n_inputs])
        np.fill_diagonal(D_star_rand, random.choices([-1, 1], k=n_inputs))
        """Shuffle columns: Commented Out to get simple stairs form!!!"""
        if stairs is False:
            np.random.shuffle(P_star_rand.T)
        else:
            pass
    else:
        base_value_vector_rand = [1 / 3] * 2
        P_star_rand = np.identity(n_inputs)
        D_star_rand = np.array([[1, 0], [0, -1]])
    # Be careful with np.dot vs. np.matmul vs. *.
    B_star_rand = np.dot(
        J * base_value_vector_rand
        + (step / 2) * (np.dot((2 * B - J), D_star_rand) + J),
        P_star_rand,
    )
    return B_star_rand


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
    """
    Computes distance between each pair of trajectories.
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
    yield [pool[i] for i in indices]
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield [pool[i] for i in indices]


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
    # Indices of combination that yields highest distance figure.
    max_dist_indices_row = combi_distance[:, n_traj].argsort()[-1:][::-1].tolist()
    max_dist_indices = combi_distance[max_dist_indices_row, 0:n_traj]
    # Convert list of float indices to list of ints.
    max_dist_indices = [int(i) for i in max_dist_indices.tolist()[0]]

    return max_dist_indices, combi_distance


def campolongo_2007(n_inputs, n_levels, n_traj_sample, n_traj):
    """
    Takes number of input parametes, samplesize of trajectories,
    and selected number of trajectories as arguments.
    Returns an array with n_inputs at the verical and n_traj at the
    horizontal axis.

    """
    sample_traj = list()
    for traj in range(0, n_traj_sample):
        seed = 123 + traj

        sample_traj.append(
            morris_trajectories(n_inputs, n_levels, step_function=stepsize, seed=seed)
        )
    pair_matrix = distance_matrix(sample_traj)
    select_indices, dist_matrix = select_trajectories(pair_matrix, n_traj)

    select_trajs = [sample_traj[idx] for idx in select_indices]
    # Rows are parameters, cols is number of drawn parameter vectors.
    input_par_array = np.vstack(select_trajs)

    return input_par_array.T, select_trajs


def simple_stairs(n_inputs, n_levels, n_traj_sample, n_traj, step_function):
    """Creates list of Morris trajectories in winding stairs format."""
    sample_traj = list()
    for traj in range(0, n_traj_sample):
        seed = 123 + traj

        sample_traj.append(
            morris_trajectories(
                n_inputs, n_levels, step_function=step_function, seed=seed
            )
        )

    # Rows are parameters, cols is number of drawn parameter vectors.
    input_par_array = np.vstack(sample_traj)

    return input_par_array.T, sample_traj


"""Experiment stepsize equidistant"""
input_par_array, trajs_list = simple_stairs(
    n_inputs=5,
    n_levels=6,
    n_traj_sample=1000,
    n_traj=1000,
    step_function=stepsize_equidistant,
)

new_list = input_par_array.reshape(-1, 1).tolist()
merged = list(itertools.chain.from_iterable(new_list))

plt.figure(2)
plt.hist(merged, range=[-0.3, 1.3])

"""Experiment stepsize"""
input_par_array, trajs_list = simple_stairs(
    n_inputs=5, n_levels=6, n_traj_sample=1000, n_traj=1000, step_function=stepsize
)

new_list = input_par_array.reshape(-1, 1).tolist()
merged = list(itertools.chain.from_iterable(new_list))

plt.figure(1)
plt.hist(merged, range=[-0.3, 1.3])
