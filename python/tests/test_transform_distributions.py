"""Test ge_menendez_2017.py"""
import sys

# Define parent folder as relative path.
sys.path.append("python")

import numpy as np

from scipy.stats import norm
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from transform_distributions import sample_stnormal_paramters
from transform_distributions import transform_uniform_stnormal_uncorr
from transform_distributions import covariance_to_correlation
from transform_distributions import transform_stnormal_normal_corr_lemaire09
from transform_distributions import transform_stnormal_normal_corr_gm17

sys.path.append("tests")
from nataf_transformation import nataf_transformation
from distributions import distributions


def test_covariance_to_correlation():
    cov = np.array([
        [10, 0.2, 0.5],
        [0.2, 40, 0],
        [0.5, 0, 50]])
    expected = np.array([
        [1, 0.01, 0.0223],
        [0.01, 1, 0],
        [0.0223, 0, 1]])
    corr = covariance_to_correlation(cov)

    assert_array_almost_equal(corr, expected, 0.001)

#Define shared objects for the next two tests.
# Expectation values.
mu = np.array([10, 10, 10, 10, 10])

# Covariance matrix.
cov = np.array([
        [10,0,0,2,0.5],
        [0,20,0.4,0.15,0],
        [0,0.4,30,0.05,0],
        [2,0.15,0.05,40,0],
        [0.5,0,0,0,50]])

# Draws from U(0,1).
row = np.array([0.1, 0.1, 0.2, 0.8, 0.5])
# Transform draws to uncorrelated N(0,1)
z = transform_uniform_stnormal_uncorr(row)

# Create Nataf transformation from class for many distribution types.
M = list()
M.append(distributions('normal', 'PAR', [mu[0], np.sqrt(cov[0 ,0])]))
M.append(distributions('normal', 'PAR', [mu[1], np.sqrt(cov[1 ,1])]))
M.append(distributions('normal', 'PAR', [mu[2], np.sqrt(cov[2 ,2])]))
M.append(distributions('normal', 'PAR', [mu[3], np.sqrt(cov[3 ,3])]))
M.append(distributions('normal', 'PAR', [mu[4], np.sqrt(cov[4 ,4])]))

Rho = covariance_to_correlation(cov)

T_Nataf = nataf_transformation(M, Rho)

def test_transform_stnormal_normal_corr_lemaire09():
    x_lemaire09 = transform_stnormal_normal_corr_lemaire09(z, cov, mu)
    X = T_Nataf.U2X(z)

    assert_array_almost_equal(x_lemaire09, X.T, 0.0001)


def test_transform_stnormal_normal_corr_gm17():
    sample_Z_c = sample_stnormal_paramters(5, 100_000)

    x_gm17 = transform_stnormal_normal_corr_gm17(z, cov, sample_Z_c, mu)
    X = T_Nataf.U2X(z)

    assert_array_almost_equal(x_gm17, X.T, 0.05)



