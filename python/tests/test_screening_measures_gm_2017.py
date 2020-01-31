"""Tests for `screening_measures_gm_2017`."""
import sys

# Define parent folder as relative path.
sys.path.append("..")

import numpy as np

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_allclose

from sampling_trajectory import morris_trajectory
from screening_measures_gm_2017 import screening_measures_gm_2017


def test_screening_measures_gm_2017_uncorrelated_parameters():
    seed = 2020
    n_levels = 10
    n_inputs = 10
    n_traj_sample = 10_000

    sample_traj_list = list()

    for traj in range(0, n_traj_sample):
        seed = seed + traj
        sample_traj_list.append(morris_trajectory(n_inputs, n_levels, seed=seed))

    expectation = np.zeros([10])
    cov = np.zeros(100).reshape(10, 10)
    np.fill_diagonal(cov, 1)

    # coefficients
    co = np.linspace(1, 10, 10)

    def linear_function(a, b, c, d, e, f, g, h, i, j, *args):
        return (
            co[0] * a
            + co[1] * b
            + co[2] * c
            + co[3] * d
            + co[4] * e
            + co[5] * f
            + co[6] * g
            + co[7] * h
            + co[8] * i
            + co[9] * j
        )

    abs_ee_ind, abs_ee_full, sd_ee_ind, sd_ee_full = screening_measures_gm_2017(
        linear_function,
        sample_traj_list,
        cov,
        n_levels,
        mu=expectation,
        numeric_zero=0.001,
    )

    expected_abs_ee_ind = np.ones(len(co)).reshape(abs_ee_ind.shape) * np.nan
    expected_abs_ee_full = np.ones(len(co)).reshape(abs_ee_full.shape) * np.nan
    expected_sd_ee_ind = np.ones(len(co)).reshape(abs_ee_ind.shape) * np.nan
    expected_sd_ee_full = np.ones(len(co)).reshape(abs_ee_full.shape) * np.nan

    for i in range(1, len(co) + 1):
        expected_abs_ee_ind[i - 1] = abs_ee_ind[0][0] * i
        expected_abs_ee_full[i - 1] = abs_ee_full[0][0] * i
        expected_sd_ee_ind[i - 1] = sd_ee_ind[0][0] * i
        expected_sd_ee_full[i - 1] = sd_ee_full[0][0] * i

    # Relative tolerance for differences is 1%.
    assert_allclose(abs_ee_ind, expected_abs_ee_ind, rtol=0.01)
    assert_allclose(abs_ee_full, expected_abs_ee_full, rtol=0.01)
    assert_allclose(sd_ee_ind, expected_sd_ee_ind, rtol=0.01)
    assert_allclose(sd_ee_full, expected_sd_ee_full, rtol=0.01)
