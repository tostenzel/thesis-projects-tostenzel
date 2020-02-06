"""Tests"""
import sys

# Define parent folder as relative path.
sys.path.append("python")

import numpy as np
import pytest
import respy as rp

from numpy.testing import assert_allclose

from jac_estimation_chol import chol_reindex_params


@pytest.mark.skip(
    reason="Speed up Travic CI.This test takes some time due to \
    the simulation of 10000 agents by `respy.get_example_model`."
)
def test_loglikelihood_chol_equals_sdcorr():
    """
    Tests wether the loglikelihood function is equal at equal parameter vectors
    specified by Cholesky factors and Sd-Corr matrix for utility shocks.

    """
    # Init default non-variation parameters and the indices in respy Df format.
    params_sdcorr_default, options, df = rp.get_example_model("kw_94_one")
    params_chol_default = chol_reindex_params(params_sdcorr_default)

    # Log-likelihoods for mean agent
    log_likelihood_sdcorr = rp.get_crit_func(
        params_sdcorr_default, options, df, "log_like"
    )
    # Log-likelihoods for mean agent
    log_likelihood_chol = rp.get_crit_func(params_chol_default, options, df, "log_like")

    # Check loglikelihood for default Dataset One paramters.
    # This is not an optimal test case as the parameters are identical.
    assert log_likelihood_sdcorr(params_sdcorr_default) == log_likelihood_chol(
        params_chol_default
    )

    # Create two identical, non-default params Dfs for both matrix specifications.
    # Init test data.
    chol = np.tril([0.1, 0.2, 0.3, 0.4])
    chol[3, :] = chol[3, :] * (-1)

    # Replace cholesky factors in test Df.
    params_chol_test = params_chol_default.copy(deep=True)

    params_chol_test.at[("shocks_chol", "chol_a"), "value"] = chol[0, 0]
    params_chol_test.at[("shocks_chol", "chol_b"), "value"] = chol[1, 1]
    params_chol_test.at[("shocks_chol", "chol_edu"), "value"] = chol[2, 2]
    params_chol_test.at[("shocks_chol", "chol_home"), "value"] = chol[3, 3]

    params_chol_test.at[("shocks_chol", "chol_b_a"), "value"] = chol[1, 0]

    params_chol_test.at[("shocks_chol", "chol_edu_a"), "value"] = chol[2, 0]
    params_chol_test.at[("shocks_chol", "chol_edu_b"), "value"] = chol[2, 1]

    params_chol_test.at[("shocks_chol", "chol_home_a"), "value"] = chol[3, 0]
    params_chol_test.at[("shocks_chol", "chol_home_b"), "value"] = chol[3, 1]
    params_chol_test.at[("shocks_chol", "chol_home_edu"), "value"] = chol[3, 2]

    # Convert Cholesky factors to Sd-Corr matrix.
    cov = np.dot(chol, chol.T)
    assert np.array_equal(cov, cov.T), "cov matrix must be symmetric"

    sd = np.sqrt(np.diag(cov))
    corr = np.ones((len(cov), len(cov))) * np.nan
    for i in range(0, len(chol)):
        for j in range(0, len(chol)):
            corr[i, j] = cov[i, j] / (sd[i] * sd[j])

    assert np.array_equal(corr, corr.T), "corr matrix must be symmetric"
    assert_allclose(np.diag(corr), np.ones(4), 1e-15)

    # Replace Cholesky matrix in Df.
    params_chol_test = params_chol_default.copy(deep=True)

    params_chol_test.at[("shocks_chol", "chol_a"), "value"] = chol[0, 0]
    params_chol_test.at[("shocks_chol", "chol_b"), "value"] = chol[1, 1]
    params_chol_test.at[("shocks_chol", "chol_edu"), "value"] = chol[2, 2]
    params_chol_test.at[("shocks_chol", "chol_home"), "value"] = chol[3, 3]

    params_chol_test.at[("shocks_chol", "chol_b_a"), "value"] = chol[1, 0]

    params_chol_test.at[("shocks_chol", "chol_edu_a"), "value"] = chol[2, 0]
    params_chol_test.at[("shocks_chol", "chol_edu_b"), "value"] = chol[2, 1]

    params_chol_test.at[("shocks_chol", "chol_home_a"), "value"] = chol[3, 0]
    params_chol_test.at[("shocks_chol", "chol_home_b"), "value"] = chol[3, 1]
    params_chol_test.at[("shocks_chol", "chol_home_edu"), "value"] = chol[3, 2]

    # Replace SD-Corr factors in Df.
    params_sdcorr_test = params_sdcorr_default.copy(deep=True)

    params_sdcorr_test.at[("shocks_sdcorr", "sd_a"), "value"] = sd[0]
    params_sdcorr_test.at[("shocks_sdcorr", "sd_b"), "value"] = sd[1]
    params_sdcorr_test.at[("shocks_sdcorr", "sd_edu"), "value"] = sd[2]
    params_sdcorr_test.at[("shocks_sdcorr", "sd_home"), "value"] = sd[3]

    params_sdcorr_test.at[("shocks_sdcorr", "corr_b_a"), "value"] = corr[1, 0]

    params_sdcorr_test.at[("shocks_sdcorr", "corr_edu_a"), "value"] = corr[2, 0]
    params_sdcorr_test.at[("shocks_sdcorr", "corr_edu_b"), "value"] = corr[2, 1]

    params_sdcorr_test.at[("shocks_sdcorr", "corr_home_a"), "value"] = corr[3, 0]
    params_sdcorr_test.at[("shocks_sdcorr", "corr_home_b"), "value"] = corr[3, 1]
    params_sdcorr_test.at[("shocks_sdcorr", "corr_home_edu"), "value"] = corr[3, 2]

    assert_allclose(
        log_likelihood_sdcorr(params_sdcorr_test),
        log_likelihood_chol(params_chol_test),
        atol=0.001,
    )
