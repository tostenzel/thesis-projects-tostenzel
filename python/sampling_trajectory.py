"""
Functions that create trajectories. These are samples that start with a row of random values
and add a fixed value each row to one additional cell which column was not chosen before.
Therefore, the number of rows is the number of columns plus 1.

"""
import random

import numpy as np
from transform_distributions import transform_uniform_stnormal_uncorr


def stepsize(n_levels):
    """
    Book recommendation:
    W/o post-selection the trajectory sample, this leads to equiprobable
    sampling from the input grid if n_levels is even.
    The reason is the following: Without the random mixing of elements,
    the first row in a trajectory produced by `morris_trajectory` are the
    smaller half of the points. Then stepsize is added on each point.
    This creates the upper half of the desired grid points.
    This does only work if the number of grid points `n_levels` is even.

    """
    assert float(
        n_levels / 2
    ).is_integer(), "n_levels must be an even number see function docstring."
    return n_levels / (2 * (n_levels - 1))


def morris_trajectory(
    n_inputs,
    n_levels,
    seed=123,
    normal=False,
    numeric_zero=0.01,
    step_function=stepsize,
    stairs=True,
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
    # Here, I eed to take care of the dimensions in case of normal is True.
    base_value_vector_rand = np.array(random.choices(value_grid, k=n_inputs)).reshape(
        1, n_inputs
    )
    # P_star defines the element in the above vector where that
    # takes the first step in the second trajectory column.
    P_star_rand = np.identity(n_inputs)
    # Unsure: Take step up or down? --> We want to have both! See last #.
    D_star_rand = np.zeros([n_inputs, n_inputs])
    np.fill_diagonal(D_star_rand, random.choices([-1, 1], k=n_inputs))
    if stairs is False:
        np.random.shuffle(P_star_rand.T)
    else:
        pass
    # Computes complete trajectory.
    # Be careful with np.dot vs. np.matmul vs. *.
    # Remove row dimension equals 1 from base_value_vector_rand to use element-wise `*`.
    B_star_rand = np.dot(
        J * np.squeeze(base_value_vector_rand)
        + (step / 2) * (np.dot((2 * B - J), D_star_rand) + J),
        P_star_rand,
    )
    # For standard normally distributed draws.
    if normal is True:
        # Be aware that the numeric_zero drastically influences the stepsize due to shape of ppt function.
        B_star_rand = np.apply_along_axis(
            transform_uniform_stnormal_uncorr, 1, B_star_rand, numeric_zero
        )
    else:
        pass
    # Need delta because it can be positive or negative.
    # If normal is true, delta is scaled non-linearily by ppt*sigma
    trans_steps = np.array([1, n_inputs])
    trans_steps = B_star_rand[-1, :] - B_star_rand[0, :]

    return B_star_rand, trans_steps
