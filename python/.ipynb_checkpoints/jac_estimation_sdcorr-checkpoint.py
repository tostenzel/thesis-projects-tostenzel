"""
Estimates covariance matrix for KW94 Dataset 1 with Simulated Max. Likelihood.

"""

import json

import numpy as np
import pandas as pd
import respy as rp

from estimagic.inference.likelihood_covs import cov_jacobian
from estimagic.differentiation.differentiation import jacobian
from estimagic.optimization.optimize import maximize
from respy.likelihood import get_crit_func


def jac_estimation_sdcorr(save=False):
    """
    Estimates covariance matrix for KW94 Dataset 1 with Simulated Max. Likelihood.
    The Jacobian matrix is used instead of Hessian because its much faster.
    The parameters contain sdcorresky factors instead of SD-Corr-Matrix because
    these factors are unconstrained. Therefore, their distribution can be estimated
    by an unconstrained normal distribution.

    Parameters
    ----------
    save: Bool
        Indicates wether to save data.
    
    Returns
    -------
    par_df: DataFrame
        Df containing parameters, SDs and lower and upper bound in estimagic format.
    cov_df: DataFrame
        Df containing the covariance matrix.
    corr_df: DataFrame
        DF containing the correlation matrix.

    """
    # Df is sample of 1000 agents in 40 periods.
    params_sdcorr, options, df = rp.get_example_model("kw_94_one")

    # Estimate parameters.
    # log_like = log_like_obs.mean(). Used for consistency with optimizers.
    # Gives log-likelihood function for mean agent.
    crit_func = rp.get_crit_func(params_sdcorr, options, df, 'log_like')

    # Get constraint for parameter estimation
    constr_sdcorr = rp.get_parameter_constraints('kw_94_one')

    # df  will take lower and upper bounds after standard error esitmation
    # so that cols fit topography plot requirements.
    par_sdcorr_df = pd.DataFrame(
        data=par_estimates['value'].values[:27],
        index=params_sdcorr[:27].index,
        columns=['value']
    )

    # The rest of this function estimates the variation of the estimates.
    # Log-likelihood function for sample of agents.
    log_like_obs_func = get_crit_func(params_sdcorr, options, df, version='log_like_obs')

    # Jacobian matrix.
    jacobian_matrix = jacobian(log_like_obs_func, params_sdcorr, extrapolation=False)

    # Drop zero lines to avoid multicollinearity for matrix inversion.
    jacobian_matrix = jacobian_matrix.loc[:, (jacobian_matrix != 0).any(axis=0)]

    jacobian_cov_matrix = cov_jacobian(jacobian_matrix.to_numpy())

    jacobian_cov_matrix = cov_jacobian(jacobian_matrix.to_numpy())

    cov_sdcorr_df = pd.DataFrame(
    data=jacobian_cov_matrix,
    index=params_sdcorr[:27].index,
    columns=params_sdcorr[:27].index,
    )

    corr_sdcorr_df = cov_sdcorr_df.copy(deep=True)
    for i in range(0,len(cov_sdcorr_df)):
        for j in range(0,len(cov_sdcorr_df)):
            corr_sdcorr_df.iloc[i,j] = cov_sdcorr_df.iloc[i,j]/(
                np.sqrt(cov_sdcorr_df.iloc[i,i]*cov_sdcorr_df.iloc[j,j]))

    assert -1 <= corr_sdcorr_df.values.any() <= 1, "Corrs must be inside [-1,1]"

    # Estimate parameters.
    # log_like = log_like_obs.mean(). Used for consistency with optimizers.
    # Gives log-likelihood function for mean agent.
    crit_func = rp.get_crit_func(params_sdcorr, options, df, 'log_like')

    constr = rp.get_parameter_constraints('kw_94_one')
    # kick out constraints for SD-Corr-Matrix. sdcorresky factors are unconstrained.
    constr_sdcorr = constr[1:4]

    _, par_estimates = maximize(
    crit_func,
    params_sdcorr,
    'scipy_L-BFGS-B',
    db_options={'rollover': 200},
    algo_options={'maxfun': 1},
    constraints=constr_sdcorr,
    dashboard=False
    )

    # Include upper and lower bounds to par_df for topography plot.
    par_sdcorr_df['sd'] = np.sqrt(np.diag(jacobian_cov_matrix))
    par_sdcorr_df['lower'] = par_sdcorr_df['value'] - 2* par_sdcorr_df['sd']
    par_sdcorr_df['upper'] = par_sdcorr_df['value'] + 2* par_sdcorr_df['sd']

    if save is True:
        cov_sdcorr_df.to_pickle("python/input/cov_sdcorr.uq.pkl")
        corr_sdcorr_df.to_pickle("python/input/corr_sdcorr.uq.pkl")
        par_sdcorr_df.to_pickle("python/input/params_sdcorr.uq.pkl")
        # contains 3 fixed respy params
        params_sdcorr.to_pickle("python/input/base_params_sdcorr.uq.pkl")
    else:
        pass
  
    return par_sdcorr_df, cov_sdcorr_df, corr_sdcorr_df


# Call function.
jac_estimation_sdcorr(save=True)










