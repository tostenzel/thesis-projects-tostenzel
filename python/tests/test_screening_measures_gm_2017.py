"""Tests for `screening_measures_gm_2017`."""
import sys

# Define parent folder as relative path.
sys.path.append("..")

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from sampling_schemes import morris_trajectory
from screening_measures_gm_2017 import screening_measures_gm_2017


def sobol_model(a, b, c, d, e, f, coeffs, *args):
    """
    Tested by comparing graphs for 3 specifications to book.
    Arguments are lists. Strongly nonlinear, nonmonotonic, and nonzero interactions.
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

    Data and results taken from pages 123 - 127 in [1]
    
    References
    ----------
    Saltelli, A., M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli, M. Saisana,
    and S. Tarantola (2008). Global Sensitivity Analysis: The Primer. John Wiley & Sons.

    """
    # Covariance matrix
    cov = np.zeros(36).reshape(6, 6)
    np.fill_diagonal(cov, np.ones(5))
    
    # This is not the expectation for x \in U[0,1]. Yet, prevents transformation.
    mu = np.array([0, 0, 0, 0, 0, 0])
    
    n_levels = 4
    
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
    # The indices show the order of columns to which the step is added.
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
    # The indices account for the column order for stairs.
    positive_steps = np.array([2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3])
    steps_one = positive_steps * np.array([1, -1, -1, 1, 1, 1])[idx_one]
    steps_two = positive_steps * np.array([1, 1, 1, -1, -1, -1])[idx_two]
    steps_three = positive_steps * np.array([-1, -1, 1, -1, -1, +1])[idx_three]
    steps_four = positive_steps * np.array([-1, +1, -1, -1, 1, 1])[idx_four]
    
    # Compute the independent Elementary Effects.
    # Since there is no correlation, they equal their abolute versions.
    one_ee_ind, _ , _, _, _, _ = screening_measures_gm_2017(
        wrapper_one, [traj_one], [steps_one], n_levels, cov, mu, numeric_zero=0.0,
        )
    
    two_ee_ind, _, _, _, _, _= screening_measures_gm_2017(
        wrapper_two, [traj_two], [steps_two], n_levels, cov, mu, numeric_zero=0.0,
        )
    
    three_ee_ind, _, _, _, _, _ = screening_measures_gm_2017(
        wrapper_three, [traj_three], [steps_three],n_levels, cov, mu, numeric_zero=0.0,
        )
    
    four_ee_ind, _, _, _, _, _ = screening_measures_gm_2017(
        wrapper_four, [traj_four], [steps_four], n_levels, cov, mu, numeric_zero=0.00,
        )
    
    # `argsort` inverses the transformation that induced the stairs shape to the trajectories.
    ee_one = np.array(one_ee_ind).reshape(6, 1)[np.argsort(idx_one)]
    ee_two = np.array(two_ee_ind).reshape(6, 1)[np.argsort(idx_two)]
    ee_three = np.array(three_ee_ind).reshape(6, 1)[np.argsort(idx_three)]
    ee_four = np.array(four_ee_ind).reshape(6, 1)[np.argsort(idx_four)]
    
    ee_i = np.concatenate((ee_one, ee_two, ee_three, ee_four), axis=1)
    
    # Compute summary measures "by hand" because `screening_measures_gm_2017`
    # takes only a list of one trajectory because the argument order is different.
    ee = np.mean(ee_i, axis=1).reshape(6, 1)
    abs_ee = np.mean(abs(ee_i), axis=1).reshape(6, 1)
    # `np.var` does not work because it scales by 1/n instead of 1/(n - 1).
    sd_ee = np.sqrt((1 / (4 - 1)) * (np.sum((ee_i - ee) ** 2, axis=1).reshape(6, 1)))
    
    expected_ee = np.array([-0.006, -0.078, -0.130, -0.004, 0.012, -0.004]).reshape(6, 1)
    expected_abs_ee = np.array([0.056, 0.277, 1.760, 1.185, 0.034, 0.099]).reshape(6, 1)
    expected_sd_ee = np.array([0.064, 0.321, 2.049, 1.370, 0.041, 0.122]).reshape(6, 1)
    
    assert_array_equal(np.round(ee, 3), np.round(expected_ee, 3))
    assert_allclose(np.round(abs_ee, 3), np.round(expected_abs_ee, 3), atol=0.01)
    assert_array_equal(np.round(sd_ee, 3), np.round(expected_sd_ee, 3))




def lin_portfolio(q1, q2, c1=2, c2=1, *args):
    """Simple function with analytic EE solution to support testing."""
    return c1 * q1 + c2 * q2


def test_screening_measures_uncorrelated_linear_function():
    """Taken from Ralph C. Smith (2014): Uncertainty Quantification, page 335."""
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
        ee_ind,
        ee_full,
        abs_ee_ind,
        abs_ee_full,
        sd_ee_ind,
        sd_ee_full,
    ) = screening_measures_gm_2017(
        lin_portfolio,
        traj_list,
        step_list,
        n_levels,
        cov,
        mu,
        numeric_zero=0.01,
    )

    exp_ee = np.array([2, 1]).reshape(n_inputs, 1)
    exp_sd = np.array([0, 0]).reshape(n_inputs, 1)

    assert_array_equal(exp_ee, ee_ind)
    assert_array_equal(exp_ee, abs_ee_ind)
    assert_array_equal(exp_ee, ee_full)
    assert_array_equal(exp_ee, abs_ee_full)
    assert_allclose(exp_sd, sd_ee_full, atol=1.0e-15)
    assert_allclose(exp_sd, sd_ee_full, atol=1.0e-15)
