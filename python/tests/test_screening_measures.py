"""Tests for `screening_measures`."""
import sys

# Define parent folder as relative path.
sys.path.append("..")

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from sampling_schemes import morris_trajectory
from screening_measures import screening_measures


def sobol_model(a, b, c, d, e, f, coeffs, *args):
    """
    Test function used in `test_screening_measures_uncorrelated_g_function`.

    Notes
    -----
    Strongly nonlinear, nonmonotonic, and nonzero interactions.
    Analytic results for Sobol Indices.

    """
    input_pars = np.array([a, b, c, d, e, f])

    def g_i(input_pars, coeffs):
        return (abs(4 * input_pars - 2) + coeffs) / (1 + coeffs)

    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs[i])

    return y


def test_screening_measures_uncorrelated_g_function():
    """Tests the screening measures for six uncorrelated parameters.

    Data and results taken from pages 123 - 127 in [1]. The data is
    four trajectories and the results are the Elementary Effects, the absolute
    Elementary Effects and the SD of the Elementary Effects for six paramters.

    Notes
    -----
    -Many intermediate results are given as well. `screening_measures` is able
    to compute all of them precisely.
    -The function uses a lot of reorderings. The reason is that `screening_measures`
    assumes that the first columns has the first step addition etc. This facilitates
    the necessary transformations to account for correlations. In this example
    the order of the paramters to which the step is added is different for each
    trajectory. To account for this discrepancy in trajectory format, the trajectories
    and `sobol_model` have to be changed accordingly. Additionally, the effects have to
    be recomputed for each trajectory because the the reordered trajectories with columns
    in order of the step addition are still composed of columns that represent different
    paramters.

    References
    ----------
    [1] Saltelli, A., M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli, M. Saisana,
    and S. Tarantola (2008). Global Sensitivity Analysis: The Primer. John Wiley & Sons.

    """
    # Covariance matrix
    cov = np.zeros(36).reshape(6, 6)
    np.fill_diagonal(cov, np.ones(5))

    # This is not the expectation for x \in U[0,1]. Yet, prevents transformation.
    mu = np.array([0, 0, 0, 0, 0, 0])

    # Data: Four trajectories.
    # The columns are randomly shuffled in contrary to what this program assumes
    traj_one = np.array(
        [
            [0, 2 / 3, 1, 0, 0, 1 / 3],
            [0, 2 / 3, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [2 / 3, 0, 1, 0, 0, 1],
            [2 / 3, 0, 1, 2 / 3, 0, 1],
            [2 / 3, 0, 1 / 3, 2 / 3, 0, 1],
            [2 / 3, 0, 1 / 3, 2 / 3, 2 / 3, 1],
        ]
    )
    traj_two = np.array(
        [
            [0, 1 / 3, 1 / 3, 1, 1, 2 / 3],
            [0, 1, 1 / 3, 1, 1, 2 / 3],
            [0, 1, 1, 1, 1, 2 / 3],
            [2 / 3, 1, 1, 1, 1, 2 / 3],
            [2 / 3, 1, 1, 1, 1, 0],
            [2 / 3, 1, 1, 1, 1 / 3, 0],
            [2 / 3, 1, 1, 1 / 3, 1 / 3, 0],
        ]
    )
    traj_three = np.array(
        [
            [1, 2 / 3, 0, 2 / 3, 1, 0],
            [1, 2 / 3, 0, 0, 1, 0],
            [1 / 3, 2 / 3, 0, 0, 1, 0],
            [1 / 3, 2 / 3, 0, 0, 1 / 3, 0],
            [1 / 3, 0, 0, 0, 1 / 3, 0],
            [1 / 3, 0, 2 / 3, 0, 1 / 3, 0],
            [1 / 3, 0, 2 / 3, 0, 1 / 3, 2 / 3],
        ]
    )
    traj_four = np.array(
        [
            [1, 1 / 3, 2 / 3, 1, 0, 1 / 3],
            [1, 1 / 3, 2 / 3, 1, 0, 1],
            [1, 1 / 3, 0, 1, 0, 1],
            [1, 1 / 3, 0, 1 / 3, 0, 1],
            [1, 1 / 3, 0, 1 / 3, 2 / 3, 1],
            [1, 1, 0, 1 / 3, 2 / 3, 1],
            [1 / 3, 1, 0, 1 / 3, 2 / 3, 1],
        ]
    )
    # The uncorrices show the order of columns to which the step is added.
    idx_one = [5, 1, 0, 3, 2, 4]
    idx_two = [1, 2, 0, 5, 4, 3]
    idx_three = [3, 0, 4, 1, 2, 5]
    idx_four = [5, 2, 3, 4, 1, 0]

    # Create stairs shape:
    # Transform trajectories so that the the step is first added to the frist columns etc.
    traj_one = traj_one[:, idx_one]
    traj_two = traj_two[:, idx_two]
    traj_three = traj_three[:, idx_three]
    traj_four = traj_four[:, idx_four]

    coeffs = np.array([78, 12, 0.5, 2, 97, 33])

    # Define wrappers around `sobol_model` to account for different coeffient order
    # due to the column shuffling. Argument order changes.
    def wrapper_one(a, b, c, d, e, f, coeffs=coeffs[idx_one]):
        return sobol_model(f, b, a, d, c, e, coeffs[idx_one])

    def wrapper_two(a, b, c, d, e, f, coeffs=coeffs[idx_two]):
        return sobol_model(b, c, a, f, e, d, coeffs[idx_two])

    def wrapper_three(a, b, c, d, e, f, coeffs=coeffs[idx_three]):
        return sobol_model(d, a, e, b, c, f, coeffs[idx_three])

    def wrapper_four(a, b, c, d, e, f, coeffs=coeffs[idx_four]):
        return sobol_model(f, c, d, e, b, a, coeffs[idx_four])

    # Compute step sizes because rows are also randomly shuffeled.
    # The uncorrices account for the column order for stairs.
    positive_steps = np.array([2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3])
    steps_one = positive_steps * np.array([1, -1, -1, 1, 1, 1])[idx_one]
    steps_two = positive_steps * np.array([1, 1, 1, -1, -1, -1])[idx_two]
    steps_three = positive_steps * np.array([-1, -1, 1, -1, -1, +1])[idx_three]
    steps_four = positive_steps * np.array([-1, +1, -1, -1, 1, 1])[idx_four]

    # Compute the uncorrependent Elementary Effects.
    # Since there is no correlation, they equal their abolute versions.
    one_ee_uncorr, _, _, _, _, _ = screening_measures(
        wrapper_one, [traj_one], [steps_one], cov, mu
    )

    two_ee_uncorr, _, _, _, _, _ = screening_measures(
        wrapper_two, [traj_two], [steps_two], cov, mu
    )

    three_ee_uncorr, _, _, _, _, _ = screening_measures(
        wrapper_three, [traj_three], [steps_three], cov, mu
    )

    four_ee_uncorr, _, _, _, _, _ = screening_measures(
        wrapper_four, [traj_four], [steps_four], cov, mu
    )

    # `argsort` inverses the transformation that uncorruced the stairs shape to the trajectories.
    ee_one = np.array(one_ee_uncorr).reshape(6, 1)[np.argsort(idx_one)]
    ee_two = np.array(two_ee_uncorr).reshape(6, 1)[np.argsort(idx_two)]
    ee_three = np.array(three_ee_uncorr).reshape(6, 1)[np.argsort(idx_three)]
    ee_four = np.array(four_ee_uncorr).reshape(6, 1)[np.argsort(idx_four)]

    ee_i = np.concatenate((ee_one, ee_two, ee_three, ee_four), axis=1)

    # Compute summary measures "by hand" because `screening_measures`
    # takes only a list of one trajectory because the argument order is different.
    ee = np.mean(ee_i, axis=1).reshape(6, 1)
    abs_ee = np.mean(abs(ee_i), axis=1).reshape(6, 1)
    # `np.var` does not work because it scales by 1/n instead of 1/(n - 1).
    sd_ee = np.sqrt((1 / (4 - 1)) * (np.sum((ee_i - ee) ** 2, axis=1).reshape(6, 1)))

    expected_ee = np.array([-0.006, -0.078, -0.130, -0.004, 0.012, -0.004]).reshape(
        6, 1
    )
    expected_abs_ee = np.array([0.056, 0.277, 1.760, 1.185, 0.034, 0.099]).reshape(6, 1)
    expected_sd_ee = np.array([0.064, 0.321, 2.049, 1.370, 0.041, 0.122]).reshape(6, 1)

    assert_array_equal(np.round(ee, 3), expected_ee, 3)
    assert_allclose(np.round(abs_ee, 3), expected_abs_ee, 3, atol=0.01)
    assert_array_equal(np.round(sd_ee, 3), expected_sd_ee, 3)


def lin_portfolio(q1, q2, c1=2, c2=1, *args):
    """Simple linear function with analytic EE solution for the next test."""
    return c1 * q1 + c2 * q2


def test_screening_measures_uncorrelated_linear_function():
    """
    Test for a linear function with two paramters. Non-unit variance and EEs are coefficients.

    Results data taken from [1], page 335.

    Notes
    -----
    This test contains intuition for reasable results (including correlations) for the first
    two testcases in [2] that also use a linear function. The corresponding EE
    should be the coefficients plus the correlation times the coefficients of the correlated
    parameters.

    References
    ----------
    [1] Smith, R. C. (2014). Uncertainty Quantification: Theory, Implementation, and Applications.
    Philadelphia: SIAM-Society for Industrial and Applied Mathematics.
    [2] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
    sensitivityanalysis of models with dependent inputs. Reliability Engineering &
    System Safety 100 (162), 28–39.

    """
    cov = np.array([[1, 0], [0, 9]])

    # mu does not matter because the function is linear. You subtract what you add.
    mu = np.array([0, 0])

    numeric_zero = 0.01
    seed = 2020
    n_levels = 10
    n_inputs = 2
    n_traj_sample = 10_000

    traj_list = list()
    step_list = list()
    for traj in range(0, n_traj_sample):
        seed = seed + traj
        m_traj, step = morris_trajectory(n_inputs, n_levels, seed, True, numeric_zero)
        traj_list.append(m_traj)
        step_list.append(step)

    (
        ee_uncorr,
        ee_corr,
        abs_ee_uncorr,
        abs_ee_corr,
        sd_ee_uncorr,
        sd_ee_corr,
    ) = screening_measures(lin_portfolio, traj_list, step_list, cov, mu)

    exp_ee = np.array([2, 1]).reshape(n_inputs, 1)
    exp_sd = np.array([0, 0]).reshape(n_inputs, 1)

    assert_array_equal(exp_ee, ee_uncorr)
    assert_array_equal(exp_ee, abs_ee_uncorr)
    assert_array_equal(exp_ee, ee_corr)
    assert_array_equal(exp_ee, abs_ee_corr)
    assert_allclose(exp_sd, sd_ee_corr, atol=1.0e-15)
    assert_allclose(exp_sd, sd_ee_corr, atol=1.0e-15)
