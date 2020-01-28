"""Fix Morris trajectory."""

import numpy as np
import random

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

n_inputs = 3
n_levels = 10
step_function = stepsize
stairs=True
seed=123
test = False

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
        value_grid.append(idx / (n_levels - 1))
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
# B_star_rand = J * base_value_vector_rand + step * B would be only
# upwards steps.



