"""
Estimates covariance matrix for KW94 Dataset 1 with
Simulated Max. Likelihood.

"""
import numpy as np
import pandas as pd
import respy as rp
from estimagic.differentiation.differentiation import jacobian
from estimagic.inference.likelihood_covs import cov_jacobian
from estimagic.optimization.optimize import maximize
from respy.likelihood import get_crit_func


def jac_estimation_chol(save=False):
    """
    Estimates covariance matrix for KW94 Dataset 1 with Simulated Max. Likelihood.
    The Jacobian matrix is used instead of Hessian because it yields no inversion
    error.
    The parameters contain Cholesky factors instead of SD-Corr-Matrix because
    these factors are unconstrained. Therefore, their distribution can be estimated
    by an unconstrained normal distribution.

    Parameters
    ----------
    save: Bool
        Indicates wether to save data.

    Returns
    -------
    par__chol_df: DataFrame
        Df containing parameters, SDs and lower and upper bound in estimagic format.
    cov_chol_df: DataFrame
        Df containing the covariance matrix.
    corr_chol_df: DataFrame
        DF containing the correlation matrix.

    """
    # Df is sample of 1000 agents in 40 periods.
    params, options, df = rp.get_example_model("kw_94_one")

    # Write params in terms of Cholesky factors instead of SD-Corr-matrix.
    # This transformation holds only true for the parametrization in KW94 Dataset 1.
    # Simply change SD-Corr indices to cholesky indices.
    params_chol = _chol_indexed_params(params)

    # Estimate parameters.
    # log_like = log_like_obs.mean(). Used for consistency with optimizers.
    # Gives log-likelihood function for mean agent.
    crit_func = rp.get_crit_func(params_chol, options, df, "log_like")

    # Get constraint for parameter estimation
    constr = rp.get_parameter_constraints("kw_94_one")
    # Kick out constraints for SD-Corr-Matrix. Cholesky factors are unconstrained.
    constr_chol = constr[1:4]

    _, par_estimates = maximize(
        crit_func,
        params_chol,
        "scipy_L-BFGS-B",
        db_options={"rollover": 200},
        algo_options={"maxfun": 1},
        constraints=constr_chol,
        dashboard=False,
    )

    # df  will take lower and upper bounds after standard error esitmation
    # so that cols fit topography plot requirements.
    par_chol_df = pd.DataFrame(
        data=par_estimates["value"].values[:27],
        index=params_chol[:27].index,
        columns=["value"],
    )

    # The rest of this function estimates the variation of the estimates.
    # Log-likelihood function for sample of agents.
    log_like_obs_func = get_crit_func(params_chol, options, df, version="log_like_obs")

    # Jacobian matrix.
    jacobian_matrix = jacobian(log_like_obs_func, params_chol, extrapolation=False)

    # Drop zero lines to avoid multicollinearity for matrix inversion.
    jacobian_matrix = jacobian_matrix.loc[:, (jacobian_matrix != 0).any(axis=0)]

    jacobian_cov_matrix = cov_jacobian(jacobian_matrix.to_numpy())

    jacobian_cov_matrix = cov_jacobian(jacobian_matrix.to_numpy())

    cov_chol_df = pd.DataFrame(
        data=jacobian_cov_matrix,
        index=params_chol[:27].index,
        columns=params_chol[:27].index,
    )

    corr_chol_df = cov_chol_df.copy(deep=True)
    for i in range(0, len(cov_chol_df)):
        for j in range(0, len(cov_chol_df)):
            corr_chol_df.iloc[i, j] = cov_chol_df.iloc[i, j] / (
                np.sqrt(cov_chol_df.iloc[i, i] * cov_chol_df.iloc[j, j])
            )

    assert -1 <= corr_chol_df.values.any() <= 1, "Corrs must be inside [-1,1]"

    # Estimate parameters.
    # log_like = log_like_obs.mean(). Used for consistency with optimizers.
    # Gives log-likelihood function for mean agent.
    crit_func = rp.get_crit_func(params_chol, options, df, "log_like")

    constr = rp.get_parameter_constraints("kw_94_one")
    # Kick out constraints for SD-Corr-Matrix. Cholesky factors are unconstrained.
    constr_chol = constr[1:4]

    # Include upper and lower bounds to par_df for topography plot.
    par_chol_df["sd"] = np.sqrt(np.diag(jacobian_cov_matrix))
    par_chol_df["lower"] = par_chol_df["value"] - 2 * par_chol_df["sd"]
    par_chol_df["upper"] = par_chol_df["value"] + 2 * par_chol_df["sd"]

    if save is True:
        cov_chol_df.to_pickle("python/input/cov_chol.uq.pkl")
        corr_chol_df.to_pickle("python/input/corr_chol.uq.pkl")
        par_chol_df.to_pickle("python/input/params_chol.uq.pkl")
        # contains 3 fixed respy params
        params_chol.to_pickle("python/input/base_params_chol.uq.pkl")
    else:
        pass

    return par_chol_df, cov_chol_df, corr_chol_df


def _chol_indexed_params(params):
    """Creates the params Df with Cholesky factors and the right indices for
    respy. This transformation holds only true for the parametrization
    in KW94 Dataset 1.
    Thus, this function simply changes SD-Corr indices to cholesky indices.
    Without the slicing and merging, index ('maximum_exp', 'edu') yields
    an uniqueness error for the second index when (..., 'sd_edu') is set to
    (..., 'edu'). Yet, because we have double_indices the indices ARE unique.

    """
    p_chol_slice = params.iloc[17:27, :]
    # Remove unused inherited index levels.
    p_chol_slice.index = p_chol_slice.index.remove_unused_levels()
    # Use the SPECIFIC property of Dataset 1 in KW94 where SD-Corr-Matrix
    # equals Cholesky maxtrix.
    # This mean we just need to, firstly, rename the first index.
    p_chol_slice.index = p_chol_slice.index.set_levels(
        p_chol_slice.index.levels[0].str.replace("shocks_sdcorr", "shocks_chol"),
        level=0,
    )

    # And secondly we need to convert the second index to respy cholesky format.
    dic = {"sd": "chol", "corr": "chol"}
    for i, j in dic.items():
        p_chol_slice.index = p_chol_slice.index.set_levels(
            p_chol_slice.index.levels[1].str.replace(i, j), level=1
        )

    # Insert params_chol with index in params by merging slices.
    part_1 = params.iloc[0:17, :]
    part_1.index = part_1.index.remove_unused_levels()
    part_3 = params.iloc[27:31, :]
    part_3.index = part_3.index.remove_unused_levels()

    parts = [part_1, p_chol_slice, part_3]
    params_chol = pd.concat(parts)

    return params_chol


# Call function.
jac_estimation_chol(save=True)
