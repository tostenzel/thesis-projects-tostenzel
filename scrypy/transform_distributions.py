"""
Implementation of the inverse Rosenblatt / inverse Nataf transformation
from uniform to normal distribution.

"""
import sys

# Define parent folder as relative path.
sys.path.append("scrypy")

import numpy as np
import scipy.linalg as linalg
from scipy.stats import norm


def covariance_to_correlation(cov):
    """Converts covariance matrix to correlation matrix.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix.

    Returns
    -------
    corr : ndarray
        Correlation matrix.

    """
    # Standard deviations of each variable.
    sd = np.sqrt(np.diag(cov)).reshape(1, len(cov))
    corr = cov / sd.T / sd

    return corr


def transform_uniform_stnormal_uncorr(uniform_deviates, numeric_zero=0.01):
    """
    Converts sample from uniform distribution to standard normal space
    without regarding correlations.

    Parameters
    ----------
    uniform_deviates : ndarray
        Draws from Uniform[0,1].
    numeric_zero : float
        Used to substitute zeros and ones before applying `scipy.stats.norm`
        to not obtain `-Inf` and `Inf`.

    Returns
    -------
    stnormal_deviates : ndarray
        `uniform deviates` converted to standard normal space without correlations.

    See Also
    --------
    `morris_trajectory`

    Notes
    -----
    -This transformation is already applied as option in `morris_trajectory`.
    The reason is that `scipy.stats.norm` transforms the random draws from the
    unit cube non-linearily including the addition of the step. To obtain
    non-distorted screening measures, it is important to also account for this
    transformation of delta in the denumerator to not violate the definition of
    the function derivation.
    -The parameter `numeric_zero` can be highly influential. I prefer it to be
    relatively large to put more proportional, i.e. less weight on the extremes.

    """
    # Need to replace ones, because norm.ppf(1) = Inf and zeros because norm.ppf(0) = -Inf
    approx_uniform_devs = np.where(
        uniform_deviates == 1, 1 - numeric_zero, uniform_deviates
    )
    approx_uniform_devs = np.where(
        approx_uniform_devs == 0, numeric_zero, approx_uniform_devs
    )

    # Inverse cdf of standard normal distribution N(0, 1).
    stnormal_deviates = norm.ppf(approx_uniform_devs)

    return stnormal_deviates


def transform_stnormal_normal_corr(z_row, cov, mu):
    """
    Inverse Rosenblatt/Nataf transformation (from standard normal)
    to multivariate normal space with given correlations following
    [1], page 77-102.

    Step 1) Compute correlation matrix.
    Step 2) Introduce dependencies to standard normal sample.
    Step 3) De-standardize sample to normal space.

    Parameters
    ----------
    z_row : ndarray
        Row of uncorrelated standard normal deviates.
    cov : ndarray
        Covariance matrix of correlated normal deviates.
    mu : ndarray
        Expectation values of correlated normal deviates

    Returns
    -------
    x_norm_row : ndarray
        Row of correlated normal deviates.
    correlate_step : float
        Lower right corner element of the lower Cholesky matrix.

    Notes
    -----
    -Importantly, the step in the numerator of the uncorrelated Elementary Effect
    is multiplied by `correlate_step`. Therefore, this factor has to multiply
    the step in the denominator as well to not violate the definition of the
    function derivation.
    -This method is equivalent to the one in [2], page 199 which uses the Cholesky decomposition
    of the covariance matrix directly. This saves the scaling by SD and expectation.
    -This method is simpler and slightly more precise than the one in [3], page 33, for
    normally distributed paramters.
    -[1] explains how Rosenblatt and Nataf transformation are equal for normally distributed
    deviates.

    References
    ----------
    [1] Lemaire, M. (2013). Structural reliability. John Wiley & Sons.
    [2] Gentle, J. E. (2006). Random number generation and Monte Carlo methods. Springer
    Science & Business Media.
    [3] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global sensitivity
    analysis of models with dependent inputs. Reliability Engineering & System
    Safety 100 (162), 28–39.

    """
    # Convert covariance matrix to correlation matrix
    corr = covariance_to_correlation(cov)

    # Compute lower Cholesky matrix from `corr`.
    chol_low = linalg.cholesky(corr, lower=True)
    correlate_step = chol_low[-1, -1]

    z_corr_stnorm = np.dot(chol_low, z_row.reshape(len(cov), 1))

    x_norm = z_corr_stnorm * np.sqrt(np.diag(cov)).reshape(len(cov), 1) + mu.reshape(
        len(cov), 1
    )
    x_norm_row = x_norm.T

    return x_norm_row, correlate_step