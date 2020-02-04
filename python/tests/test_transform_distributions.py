"""Test transform_distributions.py"""
import sys

# Define parent folder as relative path.
sys.path.append("python")

import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from transform_distributions import covariance_to_correlation
from transform_distributions import transform_uniform_stnormal_uncorr
from transform_distributions import transform_stnormal_normal_corr_lemaire09

sys.path.append("python/tests/resources/test_transform_distributions")
from nataf_transformation import nataf_transformation
from distributions import distributions


def test_covariance_to_correlation():
    cov = np.array([[10, 0.2, 0.5], [0.2, 40, 0], [0.5, 0, 50]])
    expected = np.array([[1, 0.01, 0.0223], [0.01, 1, 0], [0.0223, 0, 1]])
    corr = covariance_to_correlation(cov)

    assert_allclose(corr, expected, atol=0.0001)


# Define shared objects for the next two tests.
# Expectation values.
mu = np.array([10, 10, 10, 10, 10])

# Covariance matrix.
cov = np.array(
    [
        [10, 0, 0, 2, 0.5],
        [0, 20, 0.4, 0.15, 0],
        [0, 0.4, 30, 0.05, 0],
        [2, 0.15, 0.05, 40, 0],
        [0.5, 0, 0, 0, 50],
    ]
)

# Draws from U(0,1).
row = np.array([0.1, 0.1, 0.2, 0.8, 0.5])
# Transform draws to uncorrelated N(0,1).
z = transform_uniform_stnormal_uncorr(row)

# Create Nataf transformation from class for many distribution types.
M = list()
M.append(distributions("normal", "PAR", [mu[0], np.sqrt(cov[0, 0])]))
M.append(distributions("normal", "PAR", [mu[1], np.sqrt(cov[1, 1])]))
M.append(distributions("normal", "PAR", [mu[2], np.sqrt(cov[2, 2])]))
M.append(distributions("normal", "PAR", [mu[3], np.sqrt(cov[3, 3])]))
M.append(distributions("normal", "PAR", [mu[4], np.sqrt(cov[4, 4])]))

Rho = covariance_to_correlation(cov)

T_Nataf = nataf_transformation(M, Rho)


def test_transform_stnormal_normal_corr_lemaire09():
    """
    The implementation derived from Lemaire(2009) is more
    precise than the approach in Ge/Menendez for normally distributed
    deviates.

    """
    x_lemaire09, _ = transform_stnormal_normal_corr_lemaire09(z, cov, mu)
    X = T_Nataf.U2X(z)

    assert_allclose(x_lemaire09, X.T, atol=1.0e-14)
