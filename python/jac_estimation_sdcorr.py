"""
Estimates covariance matrix for KW94 Dataset 1 with
Simulated Max. Likelihood.
The code is very similar to jac_estimation_chol.py.
This structure is retained to not commit errors.

"""
import os

import numpy as np
import pandas as pd
import respy as rp
from estimagic.differentiation.differentiation import jacobian
from estimagic.inference.likelihood_covs import cov_jacobian
from estimagic.optimization.optimize import maximize


def jac_estimation_sdcorr(save=False):
    """
    Estimates and stores parameters and covariance matrix for KW94 Dataset 1 with
    Simulated Max. Likelihood.
    The Jacobian matrix is used instead of Hessian because its much yields no inversion
    error.
    The parameters contain  SD-Corr-Matrix elements. The outputs are used for
    comparisons with those containing the Cholesky factors.

    Parameters
    ----------
    save: Bool
        Indicates wether to save data.

    Returns
    -------
    par_estimates_sdcorr_df. DataFrame:
        Containes the estimates parameters and the not estimates fixed parameters in respy
        format.
    rand_par_sdcorr_df: DataFrame
        Df containing variable parameters, SDs and lower and upper bound in estimagic format.
        Can be post-processed with surface/topography plot.
    cov_sdcorr_df: DataFrame
        Df containing the covariance matrix.
    corr_sdcorr_df: DataFrame
        DF containing the correlation matrix.

    Notes
    -----
    Additionally, the given parameters `params_sdcorr`from which the simulation starts
    are stored. These equal the estimate results but are in respy format
    (with the 3 constants parameters). It is handy to use these directly as
    mean parameter estimates for the Uncertainty Propagation. This saves tedious reindexing.
    """
    # Df is sample of 1000 agents in 40 periods.
    sim_params_sdcorr, options, df = rp.get_example_model("kw_94_one")

    # Estimate parameters.
    # log_like = log_like_obs.mean(). Used for consistency with optimizers.
    # Gives log-likelihood function for mean agent.
    crit_func = rp.get_crit_func(sim_params_sdcorr, options, df, "log_like")

    # Get constraint for parameter estimation
    constr_sdcorr = rp.get_parameter_constraints("kw_94_one")

    _, par_estimates_sdcorr_df = maximize(
        crit_func,
        sim_params_sdcorr,
        "scipy_L-BFGS-B",
        db_options={"rollover": 200},
        algo_options={"maxfun": 1},
        constraints=constr_sdcorr,
        dashboard=False,
    )

    # Df of variable parameters. It will take lower and upper bounds
    # after standard error esitmation so that cols fit topography plot requirements.
    rand_par_sdcorr_df = pd.DataFrame(
        data=par_estimates_sdcorr_df["value"].values[:27],
        index=sim_params_sdcorr[:27].index,
        columns=["value"],
    )

    # The rest of this function estimates the variation of the estimates.
    # Log-likelihood function for sample of agents.
    log_like_obs_func = rp.get_crit_func(
        par_estimates_sdcorr_df, options, df, version="log_like_obs"
    )

    # Jacobian matrix.
    jacobian_matrix = jacobian(log_like_obs_func, par_estimates_sdcorr_df, extrapolation=False)

    # Drop zero lines to avoid multicollinearity for matrix inversion.
    jacobian_matrix = jacobian_matrix.loc[:, (jacobian_matrix != 0).any(axis=0)]

    jacobian_cov_matrix = cov_jacobian(jacobian_matrix.to_numpy())

    jacobian_cov_matrix = cov_jacobian(jacobian_matrix.to_numpy())

    cov_sdcorr_df = pd.DataFrame(
        data=jacobian_cov_matrix,
        index=par_estimates_sdcorr_df[:27].index,
        columns=par_estimates_sdcorr_df[:27].index,
    )

    corr_sdcorr_df = cov_sdcorr_df.copy(deep=True)
    for i in range(0, len(cov_sdcorr_df)):
        for j in range(0, len(cov_sdcorr_df)):
            corr_sdcorr_df.iloc[i, j] = cov_sdcorr_df.iloc[i, j] / (
                np.sqrt(cov_sdcorr_df.iloc[i, i] * cov_sdcorr_df.iloc[j, j])
            )

    assert -1 <= corr_sdcorr_df.values.any() <= 1, "Corrs must be inside [-1,1]"

    # Include upper and lower bounds to par_df for topography plot.
    rand_par_sdcorr_df["sd"] = np.sqrt(np.diag(jacobian_cov_matrix))
    rand_par_sdcorr_df["lower"] = (
        rand_par_sdcorr_df["value"] - 2 * rand_par_sdcorr_df["sd"]
    )
    rand_par_sdcorr_df["upper"] = (
        rand_par_sdcorr_df["value"] + 2 * rand_par_sdcorr_df["sd"]
    )

    # Define the script path relative to the jupyter notebook that calls the script.
    abs_dir = os.path.dirname(__file__)
    if save is True:
        # Contains 3 fixed respy parameters.
        par_estimates_sdcorr_df.to_pickle(
            os.path.join(abs_dir, "input/estimation_sdcorr/est_rp_params_sdcorr.uq.pkl")
        )
        # Contains only flexible parametes. Can be used for surface/topography plot.
        rand_par_sdcorr_df.to_pickle(
            os.path.join(abs_dir, "input/estimation_sdcorr/est_rand_params_sdcorr.uq.pkl")
        )
        cov_sdcorr_df.to_pickle(
            os.path.join(abs_dir, "input/estimation_sdcorr/est_cov_sdcorr.uq.pkl")
            )
        corr_sdcorr_df.to_pickle(
            os.path.join(abs_dir, "input/estimation_sdcorr/est_corr_sdcorr.uq.pkl")
            )
    else:
        pass

    return par_estimates_sdcorr_df, rand_par_sdcorr_df, cov_sdcorr_df, corr_sdcorr_df
