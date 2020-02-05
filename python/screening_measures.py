"""
Computes the screening measures for correlated inputs by
Ge/Menendez (2017).

"""
import numpy as np
from transform_traj_elementary_effects import trans_ee_corr_trajectories
from transform_traj_elementary_effects import trans_ee_uncorr_trajectories


def screening_measures(
    function, traj_list, step_list, cov, mu
):
    """
    The corr measures are computed on correlated and the uncorrependent
    measures are computed on decorrelated measures.

    Parameters
    ----------
    Returns
    -------

    """
    n_trajs = len(traj_list)
    n_rows = np.size(traj_list[0], 0)
    n_inputs = np.size(traj_list[0], 1)

    trans_pi_i_list, trans_piplusone_i_list, coeff_step = trans_ee_uncorr_trajectories(
        traj_list, cov, mu
    )
    (
        trans_piplusone_iminusone_list,
        original_trans_piplusone_i_list,
    ) = trans_ee_corr_trajectories(traj_list, cov, mu)

    function_evals_pi_i = np.ones([n_rows, n_trajs]) * np.nan
    function_evals_piplusone_i = np.ones([n_rows, n_trajs]) * np.nan
    function_evals_original_piplusone_i = np.ones([n_rows, n_trajs]) * np.nan
    function_evals_piplusone_iminusone = np.ones([n_rows, n_trajs]) * np.nan

    for traj in range(0, n_trajs):
        for row in range(0, n_rows):
            function_evals_pi_i[row, traj] = function(*trans_pi_i_list[traj][row, :])
            function_evals_piplusone_i[row, traj] = function(
                *trans_piplusone_i_list[traj][row, :]
            )
            function_evals_original_piplusone_i[row, traj] = function(
                *original_trans_piplusone_i_list[traj][row, :]
            )
            function_evals_piplusone_iminusone[row, traj] = function(
                *trans_piplusone_iminusone_list[traj][row, :]
            )

    # Init for uncorrependent effects
    ee_uncorr_i = np.ones([n_inputs, n_trajs]) * np.nan
    ee_uncorr = np.ones([n_inputs, 1]) * np.nan
    abs_ee_uncorr = np.ones([n_inputs, 1]) * np.nan
    sd_ee_uncorr = np.ones([n_inputs, 1]) * np.nan
    # Init for corr effects
    ee_corr_i = np.ones([n_inputs, n_trajs]) * np.nan
    ee_corr = np.ones([n_inputs, 1]) * np.nan
    abs_ee_corr = np.ones([n_inputs, 1]) * np.nan
    sd_ee_corr = np.ones([n_inputs, 1]) * np.nan
    for traj in range(0, n_trajs):
        # uncorrependet Elementary Effects for each trajectory (for each parameter).
        ee_uncorr_i[:, traj] = (
            function_evals_piplusone_i[1 : n_inputs + 1, traj]
            - function_evals_pi_i[0:n_inputs, traj]
        ) / (
            step_list[traj]
            * np.squeeze(coeff_step[traj])
            * np.squeeze(np.sqrt(np.diag(cov)))
        )
        # Above, need to account for the decorrelation and the Effects scaling by SD.
        # corr Elementary Effects
        ee_corr_i[:, traj] = (
            function_evals_piplusone_iminusone[1 : n_inputs + 1, traj]
            - function_evals_original_piplusone_i[0:n_inputs, traj]
        ) / (step_list[traj] * np.squeeze(np.sqrt(np.diag(cov))))
        # Above, need to account for Effects scaling by SD.

    ee_uncorr[:, 0] = np.mean(ee_uncorr_i, axis=1)
    abs_ee_uncorr[:, 0] = np.mean(abs(ee_uncorr_i), axis=1)
    sd_ee_uncorr[:, 0] = np.sqrt(np.var(ee_uncorr_i, axis=1))

    ee_corr[:, 0] = np.mean(ee_corr_i, axis=1)
    abs_ee_corr[:, 0] = np.mean(abs(ee_corr_i), axis=1)
    sd_ee_corr[:, 0] = np.sqrt(np.var(ee_corr_i, axis=1))

    return ee_uncorr, ee_corr, abs_ee_uncorr, abs_ee_corr, sd_ee_uncorr, sd_ee_corr
