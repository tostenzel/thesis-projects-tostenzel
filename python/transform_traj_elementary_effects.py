"""Functions to compute the elementary effects in Ge/Menendez (2017)."""
import numpy as np
from transform_distributions import transform_stnormal_normal_corr
from transform_distributions import transform_uniform_stnormal_uncorr
from transform_reorder import ee_full_reorder_trajectory
from transform_reorder import ee_ind_reorder_trajectory
from transform_reorder import inverse_ee_full_reorder_trajectory
from transform_reorder import inverse_ee_ind_reorder_trajectory
from transform_reorder import inverse_reorder_cov
from transform_reorder import inverse_reorder_mu
from transform_reorder import reorder_cov
from transform_reorder import reorder_mu


def trans_ee_ind_trajectories(
    sample_traj_list, cov, mu
):
    """
    Transforms list of trajectories to two lists of transformed trajectories
    for the computation of the independent Elementary Effects. As explained in
    Ge/Menendez (2017), pages 33 and 34, the rows in the two different transformed
    trajectories equal to T(p_{i}, i) and T(p_{i+1}, i), respectively.
    Understanding the transformations requires to write up the first transformation
    from p_i and p_{i+1} to T_1(p_{i}, i) and T_1(p_{i+1}, i).
    T_1 shifts the first i elements to the end for each row i.

    REMARK: This function creates list of transformations of whole trajectories.
    The rows in the trajectories for T(p_{i+1}, i) that are to be subtracted from
    T(p_{i}, i), are still positioned one below compared to the trajectories for
    T(p_{i}, i).
    Therefore one needs to compare each row in a traj from `trans_pi_i`
    with the respective row one below in `trans_piplusone_i`.

    Parameters
    ----------
    Returns
    -------

    """
    assert len(mu) == len(cov) == np.size(sample_traj_list[0], 1)

    n_traj_sample = len(sample_traj_list)
    n_rows = np.size(sample_traj_list[0], 0)
    zero_idx_diff = []
    one_idx_diff = []

    # Transformation 1 including taking the cdf from Transformation 2.
    for traj in range(0, n_traj_sample):
        z = sample_traj_list[traj]
        zero_idx_diff.append(ee_ind_reorder_trajectory(z, p_i_plus_one=False))
        one_idx_diff.append(ee_ind_reorder_trajectory(z))

    # Transformation 2 for p_i.
    # Need to reorder mu and covariance according to the zero index difference.
    for traj in range(0, n_traj_sample):
        # Needs to be set up again for each traj because otherwise it'd be one too much.
        mu_zero = reorder_mu(mu)
        cov_zero = reorder_cov(cov)
        for row in range(0, n_rows):
            zero_idx_diff[traj][row, :], _ = transform_stnormal_normal_corr(
                zero_idx_diff[traj][row, :], cov_zero, mu_zero
            )
            mu_zero = reorder_mu(mu_zero)
            cov_zero = reorder_cov(cov_zero)

    # Transformation 2 for p_{i+1}.
    # No re-arrangement needed as the first transformation for p_{i+1}
    # is using the original order of mu and cov.
    coeff_step = []
    for traj in range(0, n_traj_sample):
        # Needs to be set up again for each traj because otherwise it'd be one too much.
        mu_one = mu
        cov_one = cov
        # We do not need the coefficient of the first row as it is not used
        c_step = np.ones([n_rows - 1, 1]) * np.nan
        for row in range(0, n_rows):
            (
                one_idx_diff[traj][row, :],
                correlate_step,
            ) = transform_stnormal_normal_corr(
                one_idx_diff[traj][row, :], cov_one, mu_one
            )
            if row > 0:
                c_step[row - 1, 0] = correlate_step
            else:
                pass
            mu_one = reorder_mu(mu_one)
            cov_one = reorder_cov(cov_one)
        coeff_step.append(c_step)

    # Transformation 3.
    trans_pi_i = []
    trans_piplusone_i = []
    for traj in range(0, n_traj_sample):
        trans_pi_i.append(
            inverse_ee_ind_reorder_trajectory(zero_idx_diff[traj], p_i_plus_one=False)
        )
        trans_piplusone_i.append(inverse_ee_ind_reorder_trajectory(one_idx_diff[traj]))

    return trans_pi_i, trans_piplusone_i, coeff_step


def trans_ee_full_trajectories(
    sample_traj_list, cov, mu
):
    """
    Transforms a list of trajectories such that their rows correspond to
    T(p_{i+1}, i-1). To create T(p_{i}, i-1) is not needed as this is done by
    `trans_ee_ind_trajectories`.

    REMARK: It is important that from the rows `trans_piplusone_iminusone`
    one subtracts one row above in `trans_piplusone_i`.
    This implies that one does not need the first row of each
    trajectory in `trans_piplusone_iminusone` as there is no counter-part available
    in `trans_piplusone_i`.

    Parameters
    ----------
    Returns
    -------

    """
    assert len(mu) == len(cov) == np.size(sample_traj_list[0], 1)

    n_traj_sample = len(sample_traj_list)
    n_rows = np.size(sample_traj_list[0], 0)
    one_idx_diff = []
    two_idx_diff = []

    # Transformation 1 for p_{i+1} including taking the cdf from Transformation 2.
    for traj in range(0, n_traj_sample):
        z = sample_traj_list[traj]
        two_idx_diff.append(ee_full_reorder_trajectory(z))
        one_idx_diff.append(ee_ind_reorder_trajectory(z))

    # Transformation 2 for p_{i+1}.
    # Need to reorder mu and covariance according to the two index difference by
    # using the invese function as for p_i in `the function for the independent EEs.
    for traj in range(0, n_traj_sample):
        # Needs to be set up again for each traj because otherwise it'd be one too much.
        mu_two = inverse_reorder_mu(mu)
        cov_two = inverse_reorder_cov(cov)
        for row in range(0, n_rows):
            two_idx_diff[traj][row, :], _ = transform_stnormal_normal_corr(
                two_idx_diff[traj][row, :], cov_two, mu_two
            )
            mu_two = reorder_mu(mu_two)
            cov_two = reorder_cov(cov_two)

    # Transformation 2 for p_{i} = T2 for p_{i+1} in function above.
    # No re-arrangement needed as the first transformation for p_{i+1}
    # is using the original order of mu and cov.
    for traj in range(0, n_traj_sample):
        # Needs to be set up again for each traj because otherwise it'd be one too much.
        mu_one = mu
        cov_one = cov
        for row in range(0, n_rows):
            (
                one_idx_diff[traj][row, :],
                correlate_step,
            ) = transform_stnormal_normal_corr(
                one_idx_diff[traj][row, :], cov_one, mu_one
            )
            mu_one = reorder_mu(mu_one)
            cov_one = reorder_cov(cov_one)
    # Transformation 3 for p_i and p_{i+1}.
    trans_piplusone_iminusone = []
    trans_piplusone_i = []
    for traj in range(0, n_traj_sample):
        trans_piplusone_i.append(inverse_ee_ind_reorder_trajectory(one_idx_diff[traj]))
        trans_piplusone_iminusone.append(
            inverse_ee_full_reorder_trajectory(two_idx_diff[traj])
        )

    return trans_piplusone_iminusone, trans_piplusone_i
