"""
Computes the screening measures for correlated inputs by
Ge/Menendez (2017).

"""
import numpy as np
from sampling_trajectory import stepsize
from transform_traj_elementary_effects import trans_ee_full_trajectories
from transform_traj_elementary_effects import trans_ee_ind_trajectories


def screening_measures_gm_2017(function, traj_list, cov, n_levels, mu=None):
    """
    The full measures are computed on correlated and the independent
    measures are computed on decorrelated measures.
    
    """
    if mu is None:
        mu = np.zeros(len(cov))
    else:
        pass
    
    n_trajs = len(traj_list)
    n_rows = np.size(traj_list[0], 0)
    n_inputs = np.size(traj_list[0], 1)

    trans_pi_i_list, trans_piplusone_i_list = trans_ee_ind_trajectories(
        traj_list, cov, mu
    )
    trans_piplusone_iminusone_list = trans_ee_full_trajectories(
        traj_list, cov, mu)

    function_evals_pi_i = np.ones([n_rows, n_trajs]) * np.nan
    function_evals_piplusone_i = np.ones([n_rows, n_trajs]) * np.nan
    function_evals_piplusone_iminusone = np.ones([n_rows, n_trajs]) * np.nan

    for traj in range(0, n_trajs):
        for row in range(0, n_rows):
            function_evals_pi_i[row, traj] = function(*trans_pi_i_list[traj][row, :])
            function_evals_piplusone_i[row, traj] = function(
                *trans_piplusone_i_list[traj][row, :]
            )
            function_evals_piplusone_iminusone[row,traj] = function(
                    *trans_piplusone_iminusone_list[traj][row,:])
    
    step = stepsize(n_levels)

    # Init for independent effects
    ee_ind_i = np.ones([n_inputs, n_trajs]) * np.nan
    abs_ee_ind = np.ones([n_inputs, 1]) * np.nan
    sd_ee_ind = np.ones([n_inputs, 1]) * np.nan
    # Init for full effects
    ee_full_i = np.ones([n_inputs, n_trajs]) * np.nan
    abs_ee_full = np.ones([n_inputs, 1]) * np.nan
    sd_ee_full = np.ones([n_inputs, 1]) * np.nan
    for traj in range(0, n_trajs):
        # Independet Elementary Effects for each trajectory (for each parameter).
        ee_ind_i[:, traj] = (
            function_evals_piplusone_i[1 : n_inputs + 1, traj]
            - function_evals_pi_i[0:n_inputs, traj]
            ) / step
        # Full Elementary Effects
        ee_full_i[:, traj] = (
                function_evals_piplusone_iminusone[1 : n_inputs + 1, traj]
                - function_evals_piplusone_i[0:n_inputs, traj]
                ) /step
        
    abs_ee_ind[:, 0] = np.mean(abs(ee_ind_i), axis=1)
    sd_ee_ind[:, 0] = np.sqrt(np.var(ee_ind_i, axis=1))
    
    abs_ee_full[:, 0] = np.mean(abs(ee_full_i), axis=1)
    sd_ee_full[:, 0] = np.sqrt(np.var(ee_full_i, axis=1))
    
    return abs_ee_ind, abs_ee_full, sd_ee_ind, sd_ee_full