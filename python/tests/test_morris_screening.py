"""Test morris_screening.py"""
# Define parent folder as relative path.
import sys

sys.path.append("..")
import numpy as np

from numpy.testing import assert_array_equal
from morris_screening import morris_trajectories
from morris_screening import elementary_effect_i


def test_morris_trajectories():
    n_inputs = 2
    init_input_pars = np.array([1 / 3, 1 / 3], ndmin=n_inputs)
    stepsize = 2 / 3
    expected = np.array([[1, 1 / 3], [1, 1], [1 / 3, 1]])
    assert_array_equal(
        expected,
        morris_trajectories(
            init_input_pars, n_inputs, stepsize, seed=123, test_D_star_rand_2dim=True
        ),
    )


def lin_portfolio(q1, q2, c1=2, c2=1, *args):
    return c1 * q1 + c2 * q2


def test_elemtary_effect_i():
    assert 2 == round(
        elementary_effect_i(lin_portfolio, 0, [0.5, 1], stepsize=2 / 3), 10
    )

    assert 1 == round(
        elementary_effect_i(lin_portfolio, 1, [0.5, 1], stepsize=2 / 3), 10
    )
