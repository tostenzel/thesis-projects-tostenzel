"""Functions that create random samples of the trajectory and the radial design."""

import sys

# Define parent folder as relative path.
sys.path.append("../scrypy")

import random

import numpy as np
from transform_distributions import transform_uniform_stnormal_uncorr


def stepsize(n_levels):
    """
    Computes stepsize to create equiprobable sample points for the trajectory design.

    Parameters
    ----------
    n_levels : int
        Number of points in a trajectory sample.

    Returns
    -------
    step : float
        Step added to each lower half point of the point grid.

    Raises
    ------
    AssertionError
        If the number of levels is not an even integer.

    Notes
    -----
    This function, published in [1], assumes that the number of sample points called
    "levels" is an even integer. The first row in the trajectory is initialized with
    the lower half of the desired equispaced points between 0 and 1. Given the below
    formula, the step added to the lowest, second lowest, ..., highest point in the
    lower half creates the lowest, second lowest, ..., highest point in the upper half
    of the point grid.

    References
    ----------
    [1] Morris, M. D. (1991). Factorial sampling plans for preliminary computational experiments.
    Technometrics 33 (2), 161–174.

    """

    assert float(
        n_levels / 2
    ).is_integer(), "n_levels must be an even number, see function docstring."

    step = n_levels / (2 * (n_levels - 1))

    return step


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
    Creates random sample in trajectory design.

    This function creates a random sample for a number of function parameters
    (columns). The sample itself consists of the number plus one vectors of
    parameter draws (rows).

    Parameters
    ----------
    n_inputs : int
        Number of input paramters / columns / rows - 1.
    n_levels : int
        Number of distict grid points.
    seed : int
        Random seed.
    normal : bool
        Indicates whether to transform points by `scipy.normal.ppt`
    numeric_zero : float
        `if normal is True`: Prevents `scipy.normal.ppt` to return `-Inf`
        and `Inf` for 0 and 1.
    step_function : function
        Constant step as function of `n_levels` added to lower half of point grid.
    stairs : bool
        if False: Randomly shuffle columns, dissolves stairs shape.

    Returns
    -------
    B_random : ndarray
        Random sample in trajectory design.
        Dimension `n_inputs` x `n_inputs + 1`.
    trans_steps : ndarray
        Column vector of steps added to base value point. Sorted by
        parameter/column. Dimension `n_inputs` x `1`.

    See Also
    --------
    stepsize : See parameter `step_function`.
    transform_uniform_stnormal_uncorr : See parameter `numeric_zero`.

    Notes
    -----
    The method is described in [1]. This function follows the notation therein.
    The idea is tailored to compute a random sample of function arguments to
    compute local derivates. First, a random row of paramters is drawn. Then, one
    parameter is changed by a fixed step in each row. The local derivatives can be
    computed by subtracting the function evaluations of each row from its upper row,
    thereby obtaining one local derivative for each parameter. The order of rows and
    columns may be shuffled. Shuffling rows creates a negative stepsize. By default,
    the shuffling of columns is turned off to facilitate post-processing.

    Importantly, an additional option is to evaluate the points by the inverse
    normal cdf to account for normally distributed input paramters vice versa
    uniformly distributed ones. For this purpose, zeros and ones are slighly shifted
    towards the centre of [0,1], so that no infinite values arise. Given the shape
    of the inverse cdf, the specific transformation choice has large influences
    on the stepsize and therefore the Elementary Effects.
    To account for transformations, the step is recomputed for each parameter by
    subtracting the last first row from the last row.

    References
    ----------
    [1] Morris, M. D. (1991). Factorial sampling plans for preliminary computational
    experiments. Technometrics 33 (2), 161–174.

    """

    np.random.seed(seed)

    step = stepsize(n_levels)

    #  Assisting matrices to induce stairs shape; Lower triangular matrix of ones.
    B = np.tril(np.ones([n_inputs + 1, n_inputs]), -1)
    J = np.ones([n_inputs + 1, n_inputs])

    # Lower half values of point grid.
    value_grid = np.linspace(0, ((n_levels // 2) - 1) / (n_levels - 1), n_levels // 2)

    # Shuffle the lower half of the point grid to obtain the first row.
    base_value_vector_rand = np.array(random.choices(value_grid, k=n_inputs)).reshape(
        1, n_inputs
    )

    # P_random implies the order in which the step is added to the lower half value
    # of a column. Random shuffling may dissolve the stairs shape.
    P_random = np.identity(n_inputs)
    if stairs is False:
        np.random.shuffle(P_random.T)
    else:
        pass

    # Randomly flips columns to induce negative steps.
    D_random = np.zeros([n_inputs, n_inputs])
    np.fill_diagonal(D_random, random.choices([-1, 1], k=n_inputs))

    B_random = np.dot(
        J * np.squeeze(base_value_vector_rand)
        + (step / 2) * (np.dot((2 * B - J), D_random) + J),
        P_random,
    )

    # For standard normally distributed draws.
    if normal is True:
        B_random = np.apply_along_axis(
            transform_uniform_stnormal_uncorr, 1, B_random, numeric_zero
        )
    else:
        pass

    # Recompute step for each point because it may have been transformed by `(-1)` or
    # or by `transform_uniform_stnormal_uncorr`.
    trans_steps = np.array([1, n_inputs])
    trans_steps = B_random[-1, :] - B_random[0, :]

    return B_random, trans_steps
