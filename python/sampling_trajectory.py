"""
Winding stairs sampling
+ Morris (1991) improvement + Campolongo (2007) improvement
+ Ge/Menendez (2014) improvement.

"""
import random
from itertools import combinations

import numpy as np
from scipy.special import binom


def stepsize(n_levels):
    """
    Book recommendation:
    W/o post-selection the trajectory sample, this leads to equiprobable
    sampling from the input grid. This is, because the stepsize is not the
    distance between the grid points. Therefore, a step from one grid point to
    another one can not "land" on another grid point.
    The grid points are HALFWAY EQUISPACED. The steps may create grid points
    that are closer to a selection of original grid points.

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
    - Shuffling of identity matrix is turned off by default to obtain stairs
    for the Qiao We / Menendez (2016) paper for correlated input paramters.
    The respective function argument is `stairs`.
    - Containts a test option to fix the initial random matrices to check one
    specific testcase.

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
        # base_value rand \in [i/(1-step)]
        # !!!The upper bound must be 1 - step, because step is added on top of the
        # values. Otherwise one could sample values > 1.!!!
        value_grid = [0, 1 - step]
        idx = 1
        while idx / (n_levels - 1) < 1 - step:
            value_grid.append(idx / idx / (n_levels - 1))
            idx = idx + 1
        # The below arrays are random and therefore influenced by the seed.
        # Choose a random vector from the parameter grid to as first level.
        base_value_vector_rand = np.array(random.choices(value_grid, k=n_inputs))
        # P_star defines the element in the above vector where that
        # takes the first step in the second trajectory column.
        P_star_rand = np.identity(n_inputs)
        # Unsure: Take step up or down?
        D_star_rand = np.zeros([n_inputs, n_inputs])
        np.fill_diagonal(D_star_rand, random.choices([-1, 1], k=n_inputs))
        if stairs is False:
            np.random.shuffle(P_star_rand.T)
        else:
            pass
    else:
        base_value_vector_rand = [1 / 3] * 2
        P_star_rand = np.identity(n_inputs)
        D_star_rand = np.array([[1, 0], [0, -1]])
    # Computes complete trajectory.
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
    Trajectories are np.Arrays with step iterations as rows
    and parameters as columns.

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
    """
    Wrapper around `itertools.combinations (written in C; see
    https://docs.python.org/2/library/itertools.html#itertools.combinations).
    Needs a hashable container (e.g. a list) and returns
    np.binomial(len(iterable), r) combinations of the iterable.
    These combinations are sorted.
    - Example: combi_wrapper([0, 1, 2, 3], 2) returns
    [[0, 1], [0,2], [0,3], [1,2], [1,3], [2,3]].
    - This wrapper returns a list of lists instead of a tuple of tuples.

    """
    tup_tup = combinations(iterable, r)
    list_list = [list(x) for x in tup_tup]
    return list_list


def select_trajectories(traj_dist_matrix, n_traj):
    """
    WARNING 2: This function can be very slow because it computes distances
    between np.binomial(len(traj_dist_matrix, n_traj) pairs of trajectories.
    - Example: np.biomial(30,15) = 155117520.

    IMPORTANT REMARK 1: The aggregate distance is the same as
    in Ge/Menendez (2014). In equation (10) they effectively loop over
    each line and column in the distance_matrix. Since the matrix is symmetric.
    they multiply it by 0.5. This function, however, effectively, only
    loops over a lower triangular distance matrix because it inserts each
    combination only one time:
    ```
    combi_distance[row, n_traj] += (
        traj_dist_matrix[int(pair[0])][int(pair[1])] ** 2
            )
    Perhaps, following Ge/Menendez(2014), the speed can be improved,
    by using their formula, thereby applying the distance matrix on each
    non-pair combination.
    ```

    IMPORTANT REMARK 2: This selection function yields precise results
    because each aggregate distance for each possible combination of
    trajectories is computed directly. The faster, iterative methods
    can yield different results that are, however, close in the aggregate
    distance. The aggregate distances tend to differentiate clearly.
    Therefore, the optimal combination is precisely determined.
    
    Converts symmetric matrix of distances with 0 diagonal for each
    trajectory pair to matrix with rows for each combination of
    n_traj combination of pairs. The returned matrix is `combi_distance`.
    The index-based combinations given traj_dist_matrix are in [row,:n_traj]
    and the aggregate distance of the pair combinations, their
    rooted square sum is in [row,n_traj+1]
    Also returns list of indices for traj_dist list to select the optimal
    trajsectories.
    The indices in the matrix of the optimal trajectory (pairs) are returned
    in the list `max_dist_indices`.

    """
    assert np.all(np.abs(traj_dist_matrix - traj_dist_matrix.T) < 1e-8)
    # Get all possible combinations of input parameters by their indices.
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
            # Aggreate the pair distance to the aggregate distance of the
            # trajectory combination.
            # There is no * 0.5 in contrary to Ge/Menendez (2014).
            combi_distance[row, n_traj] += (
                traj_dist_matrix[int(pair[0])][int(pair[1])] ** 2
            )
    combi_distance[:, n_traj] = np.sqrt(combi_distance[:, n_traj])
    # Select indices of combination that yields highest aggregate distance.
    max_dist_indices_row = combi_distance[:, n_traj].argsort()[-1:][::-1].tolist()
    max_dist_indices = combi_distance[max_dist_indices_row, 0:n_traj]
    # Convert list of float indices to list of ints.
    max_dist_indices = [int(i) for i in max_dist_indices.tolist()[0]]

    return max_dist_indices, combi_distance


def select_trajectories_wrapper_iteration(traj_dist_matrix, n_traj):
    """
    WARNING: Oftentimes this function leads to diffent combinations than
    `select_trajectories`. The reason seems to be that this function
    deviates from the optimal path due to numerical reasons as different
    combinations may be very close.
    However, the aggregate sum of the returned combinations are close.
    Therefore, the quality loss is negligible compared to the speed gain
    for large numbers of trajectory combinations.

    Wraps `select_trajectories`.
    To reduce the computational burden of computing a binomial coefficent
    of distances, The function is applied iteratively as follows:
    ```
    for i in range(1,n_traj_sample - n_traj):
        intermediate_result = select_trajectories(, n_traj_sample - i)
    ```
    Therefore, `combi_distance` differs from the one in `select_trajectories`
    because it only contains the combination indices from the last iteration.
    
    """
    n_traj_sample = np.size(traj_dist_matrix, 0)
    tracker_original_indices = np.arange(0, np.size(traj_dist_matrix, 0))
    for i in range(0,n_traj_sample - n_traj):

        indices = np.arange(0, np.size(traj_dist_matrix, 0)).tolist()
        # get list of all indices
        # get list of surviving indices
        max_dist_indices, combi_distance =  select_trajectories(traj_dist_matrix, np.size(traj_dist_matrix, 0) - 1)
        # lost index
        lost_index = [item for item in indices if item not in max_dist_indices][0]
        
        # delete pairs with dropped trajectory from distance matrix
        traj_dist_matrix = np.delete(traj_dist_matrix, lost_index, axis=0)
        traj_dist_matrix = np.delete(traj_dist_matrix, lost_index, axis=1)
        tracker_original_indices = np.delete(tracker_original_indices, lost_index, axis=0)
    
    return tracker_original_indices.tolist(), combi_distance


def simple_stairs(n_inputs, n_levels, n_traj):
    """
    Returns array and list of Morris trajectories without further
    post-selection.
    
    """
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


def campolongo_2007(sample_traj_list, n_traj):
    """
    WARNING: Slow for large (len(sample_traj_list) - n_traj),
    see `select_trajectories`.
    Takes a list of Morris trajectories and selects the n_traj trajectories
    with the largest distance between them.
    Returns the selection as array with n_inputs at the verical and n_traj at the
    horizontal axis and as a list.
    It also returns the diagonal matrix that contains the pair distance
    between each trajectory pair.

    """
    pair_matrix = distance_matrix(sample_traj_list)
    select_indices, combi_distance = select_trajectories(pair_matrix, n_traj)

    select_trajs = [sample_traj_list[idx] for idx in select_indices]
    # Rows are parameters, cols is number of drawn parameter vectors.
    input_par_array = np.vstack(select_trajs)
    select_dist_matrix = distance_matrix(select_trajs)

    return input_par_array.T, select_trajs, select_dist_matrix


def intermediate_ge_menendez_2014(sample_traj_list, n_traj):
    """
    WARNING: Oftentimes this function leads to diffent combinations than
    `select_trajectories`. However, their aggregate distance if very close
    to the optimal solution, see `select_trajectories_wrapper_iteration`.

    This function implements the first part of the sampling improvement in
    terms of computation time by Ge/Menendez(2014). It is the iterative
    selection of the optimal trajectories given an aggregate distance
    in `select_trajectories_wrapper_iteration`.
    The next step is to compute the distances by using those from the last
    instead of computing them anew in each iteration.

    
    """
    pair_matrix = distance_matrix(sample_traj_list)
    # this function is the difference to campolongo
    select_indices, combi_distance = select_trajectories_wrapper_iteration(pair_matrix, n_traj)

    select_trajs = [sample_traj_list[idx] for idx in select_indices]
    # Rows are parameters, cols is number of drawn parameter vectors.
    input_par_array = np.vstack(select_trajs)
    select_dist_matrix = distance_matrix(select_trajs)

    return input_par_array.T, select_trajs, select_dist_matrix 





# Rows are parameters, cols is number of drawn parameter vectors.
# input_par_array = np.vstack(sample_traj)
"""compare campolongo with ge/menendez"""

n_inputs = 4
n_levels = 5
n_traj_sample = 30
n_traj = 5


sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj_list.append(
        morris_trajectory(n_inputs, n_levels, step_function=stepsize, seed=seed)
    )
    
_, select_list, select_distance_matrix = campolongo_2007(sample_traj_list, n_traj)

_, select_list_2, select_distance_matrix_2 = intermediate_ge_menendez_2014(sample_traj_list, n_traj)

np.sqrt(sum(sum(np.tril(select_distance_matrix**2))))
np.sqrt(sum(sum(np.tril(select_distance_matrix_2**2))))

"""Compute aggregate distance and compare the result for a test."""
traj_dist_matrix = distance_matrix(sample_traj_list)
indices, combi_distance = select_trajectories(traj_dist_matrix, n_traj)
combi = combi_wrapper(list(np.arange(0, np.size(traj_dist_matrix, 1))), n_traj)
"""Compute aggregate distance of a trajectory by using the trajectoy pairs"""
# This loop needs to be parallelized.
for row in range(0, len(combi)):
    # Assign last column
    combi_distance[row, n_traj] = 0
    pair_combi = combi_wrapper(combi[row], 2)
    for pair in pair_combi:
        # Aggreate the pair distance to the aggregate distance of the
        # trajectory combination.
        # There is no * 0.5 in contrary to Ge/Menendez (2014).
        combi_distance[row, n_traj] += (
            traj_dist_matrix[int(pair[0])][int(pair[1])] ** 2
        )
combi_distance[:, n_traj] = np.sqrt(combi_distance[:, n_traj])





