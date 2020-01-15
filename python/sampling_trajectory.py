"""
Winding stairs sampling
+ Morris (1991) improvement + Campolongo (2007) improvement

"""
import random
from itertools import combinations as combis

import numpy as np
from scipy.special import binom


def stepsize(n_levels):
    """
    Book recommendation:
    Leads not to equiprobable sampling from the input distribution.
    and is HALFWAY EQUISPACED.

    """
    return n_levels / (2 * (n_levels - 1))


def morris_trajectory(
    n_inputs, n_levels, step_function=stepsize, stairs=True, seed=123, test=False
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


def combi_wrapper(iterable, r):
    tup_tup = combis(iterable, r)
    list_list = [list(x) for x in tup_tup]
    return list_list


def select_trajectories(traj_dist_matrix, n_traj):
    """
    This function can be very slow because it grows by
    n_traj sample binomial n_traj by computing each possible distance between
    pair of trajectories.

    Convert symmetric matrix of distances for each trajectory pair
    to matrix with rows for each combination of n_traj combination of pairs.
    The distance_matrix are in [row,:n_traj] and the root-squared sum is in
    [row,n_traj+1]
    Also returns list of indices for traj_dist list to select the optimal trajs.

    """
    assert np.all(np.abs(traj_dist_matrix - traj_dist_matrix.T) < 1e-8)
    # Get all possible combinations of input parameters by their indices.
    # Unfortunatley returned as tuples.
    combi = combi_wrapper(list(np.arange(0, np.size(traj_dist_matrix, 1))), n_traj)
    assert len(combi) == binom(np.size(traj_dist_matrix, 1), n_traj)
    # leave last column open for aggregate distance
    combi_distance = np.ones([len(combi), n_traj + 1]) * np.nan
    combi_distance[:, 0:n_traj] = np.array(combi)

    # This loop needs to be parallelized.
    for row in range(0, len(combi)):
        # Assign last column
        combi_distance[row, n_traj] = 0
        pair_combi = combi_wrapper(combi[row], 2)
        for pair in pair_combi:

            combi_distance[row, n_traj] += (
                traj_dist_matrix[int(pair[0])][int(pair[1])] ** 2
            )
    # Below, there is no * 0.5 in contrary to Ge/Menendez (2014).
    combi_distance[:, n_traj] = np.sqrt(combi_distance[:, n_traj])
    # Indices of combination that yields highest distance figure.
    max_dist_indices_row = combi_distance[:, n_traj].argsort()[-1:][::-1].tolist()
    max_dist_indices = combi_distance[max_dist_indices_row, 0:n_traj]
    # Convert list of float indices to list of ints.
    max_dist_indices = [int(i) for i in max_dist_indices.tolist()[0]]

    return max_dist_indices, combi_distance









traj_dist_matrix = np.array([
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

n_traj = 2


def select_trajectories_iteration(traj_dist_matrix, n_traj):
    """Drop one trajectory in each iteration.
    Save lost indices. Then return all original indices minus the lost ones?
    
    """
    n_traj_sample = np.size(traj_dist_matrix, 0)
    lost_indices = []
    original_indices = np.arange(0, np.size(traj_dist_matrix, 0)).tolist()
    for i in range(0,n_traj_sample - n_traj):
    
        indices = np.arange(0, np.size(traj_dist_matrix, 0)).tolist()
        # get list of all indices
        # get list of surviving indices
        max_dist_indices, combi_distance =  select_trajectories(traj_dist_matrix, np.size(traj_dist_matrix, 0) - 1)
        # lost index
        lost_index = [item for item in indices if item not in max_dist_indices][0]
        # need to account for indices that have been deleted before
        count = sum(lost_index >= idx for idx in lost_indices)
        lost_indices.append(lost_index + count)
        # delete pairs with dropped trajectory from distance matrix
        traj_dist_matrix = np.delete(traj_dist_matrix, lost_index, axis=0)
        traj_dist_matrix = np.delete(traj_dist_matrix, lost_index, axis=1)
    
    left_max_dist_indices = [item for item in original_indices if item not in lost_indices]
    
    return left_max_dist_indices, combi_distance



        

max_dist_indices, combi_distance = select_trajectories(traj_dist_matrix, 2)

max_dist_indices_iter, combi_distance_iter = select_trajectories_iteration(traj_dist_matrix, 2)    
    







    

def campolongo_2007(sample_traj_list, n_traj):
    """
    Takes number of input parametes, samplesize of trajectories,
    and selected number of trajectories as arguments.
    Returns an array with n_inputs at the verical and n_traj at the
    horizontal axis.

    """

    pair_matrix = distance_matrix(sample_traj_list)
    select_indices, dist_matrix = select_trajectories(pair_matrix, n_traj)

    select_trajs = [sample_traj_list[idx] for idx in select_indices]
    # Rows are parameters, cols is number of drawn parameter vectors.
    input_par_array = np.vstack(select_trajs)
    select_dist_matrix = distance_matrix(select_trajs)

    return input_par_array.T, select_trajs, select_dist_matrix


def simple_stairs(n_inputs, n_levels, n_traj):
    """Creates list of Morris trajectories in winding stairs format."""
    sample_traj_list = list()
    for traj in range(0, n_traj):
        seed = 123 + traj

        sample_traj_list.append(
            morris_trajectory(
                n_inputs, n_levels, step_function=stepsize, seed=seed
            )
        )

    # Rows are parameters, cols is number of drawn parameter vectors.
    input_par_array = np.vstack(sample_traj_list)

    return input_par_array.T, sample_traj_list


def intermediate_ge_menendez_2014(sample_traj_list, n_traj):
    """
    Intermediate because it still computes the the distance matrix anew
    for each decreased sample.
    
    Selects n_traj trajectories out of n_traj_sample by shrinking the latter by one trajectory in
    each of the n_traj_sample - n_traj iteration.
    This saves the generation of n_traj_sample binomial n_traj combinations.
    Thus, this approach is much more efficient than the one in Campolongo(2007).
    Another differece is also the multiplication wiuth 0.5 in the aggregate distance.
    
    """
    n_traj_sample = len(sample_traj_list)
    for i in range(0, n_traj_sample - n_traj + 1):
        traj_dist_matrix = distance_matrix(sample_traj_list)
        select_indices, _ = select_trajectories(traj_dist_matrix, n_traj_sample - i)
        sample_traj_list = [sample_traj_list[idx] for idx in select_indices]

    # iteration of distance_matrix may introduce rounding errors?
    traj_dist_matrix = distance_matrix(sample_traj_list)
    # Rows are parameters, cols is number of drawn parameter vectors.
    input_par_array = np.vstack(sample_traj_list)

    return input_par_array, sample_traj_list, traj_dist_matrix




# Rows are parameters, cols is number of drawn parameter vectors.
# input_par_array = np.vstack(sample_traj)
"""compare campolongo with ge/menendez"""
"""
n_inputs = 4
n_levels = 11
n_traj_sample = 50
n_traj = 5


sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj_list.append(
        morris_trajectory(n_inputs, n_levels, step_function=stepsize, seed=seed)
    )
    
_, select_list, select_distance_matrix = campolongo_2007(sample_traj_list, n_traj)

_, select_list_2, select_distance_matrix_2 = intermediate_ge_menendez_2014(sample_traj_list, n_traj)

(sum(sum(select_distance_matrix)))
(sum(sum(select_distance_matrix_2)))
"""
