"""Functions to compute the elementary effects in Ge/Menendez (2017)."""
import numpy as np
from sampling_trajectory import morris_trajectory
from transform_distributions import transform_stnormal_normal_corr_lemaire09
from transform_reorder import ee_full_reorder_trajectory
from transform_reorder import ee_ind_reorder_trajectory
from transform_reorder import inverse_ee_full_reorder_trajectory
from transform_reorder import inverse_ee_ind_reorder_trajectory
from transform_reorder import inverse_reorder_cov
from transform_reorder import inverse_reorder_mu
from transform_reorder import reorder_cov
from transform_reorder import reorder_mu


def trans_ee_ind_trajectories(sample_traj_list, mu, cov):
    """
    Transforms list of trajectories to two lists of transformed trajectories
    for the computation of the independent Elementary Effects. As explained in
    Ge/Menendez (2017), pages 33 and 34, the rows equal to T(p_{i}, i) and
    T(p_{i+1}, i), respectively.

    REMARK: This function creates list of transformations of whole trajectories.
    The rows in the trajectories for T(p_{i+1}, i) that are to be subtracted from
    T(p_{i}, i), are still positioned one below compared to the trajectories for
    T(p_{i}, i).
    Therefore one needs to compare each row in a traj from `trans_pi_i`
    with the respective row one below in `trans_piplusone_i`.

    """
    assert len(mu) == len(cov) == np.size(sample_traj_list[0], 1)

    n_traj_sample = len(sample_traj_list)
    n_rows = np.size(sample_traj_list[0], 0)
    zero_idx_diff = []
    one_idx_diff = []

    # Transformation 1.
    for traj in range(0, n_traj_sample):
        zero_idx_diff.append(
            ee_ind_reorder_trajectory(sample_traj_list[traj], p_i_plus_one=False)
        )
        one_idx_diff.append(ee_ind_reorder_trajectory(sample_traj_list[traj]))

    # Transformation 2 for p_i
    # Need to reorder mu and covariance according to the zero index difference.
    mu_zero = reorder_mu(mu)
    cov_zero = reorder_cov(cov)
    for traj in range(0, n_traj_sample):
        for row in range(0, n_rows):
            zero_idx_diff[traj][row, :] = transform_stnormal_normal_corr_lemaire09(
                zero_idx_diff[traj][row, :], cov_zero, mu_zero
            )
            mu_zero = reorder_mu(mu_zero)
            cov_zero = reorder_cov(cov_zero)

    # Transformation 2 for p_{i+1}
    # No re-arrangement needed as the first transformation for p_{i+1}
    # is using the original order of mu and cov.
    mu_one = mu
    cov_one = cov
    for traj in range(0, n_traj_sample):
        for row in range(0, n_rows):
            one_idx_diff[traj][row, :] = transform_stnormal_normal_corr_lemaire09(
                one_idx_diff[traj][row, :], cov_one, mu_one
            )
            mu_one = reorder_mu(mu_one)
            cov_one = reorder_cov(cov_one)

    # Transformation 3.
    trans_pi_i = []
    trans_piplusone_i = []
    for traj in range(0, n_traj_sample):
        trans_pi_i.append(
            inverse_ee_ind_reorder_trajectory(zero_idx_diff[traj], p_i_plus_one=False)
        )
        trans_piplusone_i.append(inverse_ee_ind_reorder_trajectory(one_idx_diff[traj]))

    return trans_pi_i, trans_piplusone_i


def trans_ee_full_trajectories(sample_traj_list, mu, cov):
    """Transforms a list of trajectories such that their rows correspond to
    T(p_{i+1}, i-1). To create T(p_{i}, i-1) is not needed as this is done by
    `trans_ee_ind_trajectories`.
    REMARK: It is important that from the rows `trans_piplusone_iminusone`
    one subtracts one row above in `trans_piplusone_i`.

    """
    assert len(mu) == len(cov) == np.size(sample_traj_list[0], 1)

    n_traj_sample = len(sample_traj_list)
    n_rows = np.size(sample_traj_list[0], 0)
    two_idx_diff = []

    # Transformation 1 for p_{i+1}.
    for traj in range(0, n_traj_sample):
        two_idx_diff.append(ee_full_reorder_trajectory(sample_traj_list[traj]))

    # Transformation 2 for p_{i+1}.
    # Need to reorder mu and covariance according to the two index difference.
    mu_two = inverse_reorder_mu(mu)
    cov_two = inverse_reorder_cov(cov)
    for traj in range(0, n_traj_sample):
        for row in range(0, n_rows):
            two_idx_diff[traj][row, :] = transform_stnormal_normal_corr_lemaire09(
                two_idx_diff[traj][row, :], cov_two, mu_two
            )
            mu_two = reorder_mu(mu_two)
            cov_two = reorder_cov(cov_two)

    # Transformation 3 for p_{i+1}.
    trans_piplusone_iminusone = []
    for traj in range(0, n_traj_sample):
        trans_piplusone_iminusone.append(
            inverse_ee_full_reorder_trajectory(two_idx_diff[traj])
        )

    return trans_piplusone_iminusone


"""As written in Ge/Menendez (2017), page 34: The elements in vectors T(p_{i}, i) and
T(p_{i+1}, i) are the same except of the ith element."""

mu = np.array([10, 11, 12, 13, 14])

cov = np.array(
    [
        [10, 0, 0, 2, 0.5],
        [0, 20, 0.4, 0.15, 0],
        [0, 0.4, 30, 0.05, 0],
        [2, 0.15, 0.05, 40, 0],
        [0.5, 0, 0, 0, 50],
    ]
)

n_traj_sample = 100
sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj_list.append(morris_trajectory(n_inputs=5, n_levels=6, seed=seed))

trans_zero, trans_one = trans_ee_ind_trajectories(sample_traj_list, mu, cov)

trans_two = trans_ee_full_trajectories(sample_traj_list, mu, cov)
