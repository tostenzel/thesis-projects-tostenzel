"""Test ge_menendez_2017.py"""
import sys

# Define parent folder as relative path.
sys.path.append("python")

import numpy as np

from scipy.stats import norm
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from ge_menendez_2017_traj_transformation import reorder_trajectory
from ge_menendez_2017_traj_transformation import reverse_reorder_trajectory
from ge_menendez_2017_traj_transformation import sample_stnormal_paramters
from ge_menendez_2017_traj_transformation import correlate_normalize_row
from ge_menendez_2017_traj_transformation import james_e_gentle_2006
from ERANataf import ERANataf
from ERADist import ERADist

def test_transformations():
    traj = np.array([[0, 0, 0], [1, 0, 0], [2, 3, 0], [4, 5, 6]])

    assert_array_equal(
        reorder_trajectory(traj, p_i_plus_one=True),
        np.array([[0, 0, 0], [0, 0, 1], [0, 2, 3], [4, 5, 6]]),
    )

    assert_array_equal(
        traj,
        reverse_reorder_trajectory(
            reorder_trajectory(traj, p_i_plus_one=True), p_i_plus_one=True
        ),
    )

    assert_array_equal(
        reorder_trajectory(traj, p_i_plus_one=False),
        np.array([[0, 0, 0], [0, 1, 0], [2, 3, 0], [5, 6, 4]]),
    )

    assert_array_equal(
        traj,
        reverse_reorder_trajectory(
            reorder_trajectory(traj, p_i_plus_one=False), p_i_plus_one=False
        ),
    )

def test_correlate_normalize_row():
    """
    Compares the method in steps 1 to 5 on page 38 in Ge/Menendez (2017)
    to the method on page 197 in Gentle (1943).
    """
    row_approx = np.array([0.1, 0.1, 0.2, 0.8, 0.5])
    cov = np.array([
        [1,0,0,0.2,0.5],
        [0,1,0.4,0.15,0],
        [0,0.4,1,0.05,0],
        [0.2,0.15,0.05,1,0],
        [0.5,0,0,0,1]])

    sample_Z_c = sample_stnormal_paramters(5, 100_000)

    expected = james_e_gentle_2006(row_approx, cov)
    gm17 = correlate_normalize_row(row_approx, cov, sample_Z_c)
    assert_array_almost_equal(gm17, expected, 0.01)

def test_Nataf_transformation_standard_normal():
    row_approx = np.array([0.1, 0.1, 0.2, 0.8, 0.5])
    cov = np.array([
        [1,0,0,0.2,0.5],
        [0,1,0.4,0.15,0],
        [0,0.4,1,0.05,0],
        [0.2,0.15,0.05,1,0],
        [0.5,0,0,0,1]])

    expected = james_e_gentle_2006(row_approx, cov)

    M = list()
    M.append(ERADist('normal', 'PAR', [0, 1]))
    M.append(ERADist('normal', 'PAR', [0, 1]))
    M.append(ERADist('normal', 'PAR', [0, 1]))
    M.append(ERADist('normal', 'PAR', [0, 1]))
    M.append(ERADist('normal', 'PAR', [0, 1]))

    z = norm.ppf(row_approx) 
    # The second argument must be the correlation matrix.
    # In the case of a standard normal distribution, this
    # is the covariance matrix.
    T_Nataf = ERANataf(M, cov)

    X = T_Nataf.U2X(z)

    assert_array_almost_equal(X, expected.reshape(5,1), 0.01)
