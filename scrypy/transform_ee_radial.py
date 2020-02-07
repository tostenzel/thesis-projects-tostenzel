"""
Functions to compute the arguments for the function evaluations in the numerator
of the individual uncorrelated and correlated Elementary Effects following [1],
page 33 and 34, and coefficients that scale the step.

References
----------
[1] Ge, Q. and M. Menendez (2017). Extending morris method for qualitative global
sensitivityanalysis of models with dependent inputs. Reliability Engineering &
System Safety 100 (162), 28–39.

"""
import numpy as np
from transform_distributions import transform_stnormal_normal_corr
from transform_reorder import ee_corr_reorder_trajectory
from transform_reorder import ee_uncorr_reorder_trajectory
from transform_reorder import reorder_cov
from transform_reorder import reorder_mu
from transform_reorder import reverse_ee_corr_reorder_trajectory
from transform_reorder import reverse_ee_uncorr_reorder_trajectory
from transform_reorder import reverse_reorder_cov
from transform_reorder import reverse_reorder_mu


def trans_ee_uncorr_radial(sample_traj_list, cov, mu):

    assert len(mu) == len(cov) == np.size(sample_traj_list[0], 1)

    n_traj_sample = len(sample_traj_list)
    n_rows = np.size(sample_traj_list[0], 0)
    zero_idx_diff = []
    one_idx_diff = []

    # Transformation 1.
    for traj in range(0, n_traj_sample):
        z = sample_traj_list[traj]
        one_idx_diff.append(ee_uncorr_reorder_trajectory(z))
        # Only use first row for subtration in EE numerator for radial design.
        z = np.tile(z[0, :], (n_rows, 1))
        zero_idx_diff.append(ee_uncorr_reorder_trajectory(z, row_plus_one=False))

    # Transformation 2 for p_{i+1}.
    # No re-arrangement needed as the first transformation for p_{i+1}
    # is using the original order of mu and cov.
    # ´coeff_step` saves the coefficient from the last element in the Cholesky matrix
    # that transforms the step.
    coeff_step = []
    for traj in range(0, n_traj_sample):
        # Needs to be set up again for each traj because otherwise it'd be one `i`too much.
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

    # Transformation 2 for p_i.
    # Need to reorder mu and covariance according to the zero idx difference.
    for traj in range(0, n_traj_sample):
        # Needs to be set up again for each traj because otherwise it'd be `i` too much.
        mu_zero = reorder_mu(mu)
        cov_zero = reorder_cov(cov)
        for row in range(0, n_rows):
            zero_idx_diff[traj][row, :], _ = transform_stnormal_normal_corr(
                zero_idx_diff[traj][row, :], cov_zero, mu_zero
            )
            mu_zero = reorder_mu(mu_zero)
            cov_zero = reorder_cov(cov_zero)

    # Transformation 3: Undo Transformation 1.
    trans_pi_i = []
    trans_piplusone_i = []
    for traj in range(0, n_traj_sample):
        trans_pi_i.append(
            reverse_ee_uncorr_reorder_trajectory(
                zero_idx_diff[traj], row_plus_one=False
            )
        )
        trans_piplusone_i.append(
            reverse_ee_uncorr_reorder_trajectory(one_idx_diff[traj])
        )

    return trans_piplusone_i, trans_pi_i, coeff_step


def trans_ee_corr_radial(sample_traj_list, cov, mu):

    assert len(mu) == len(cov) == np.size(sample_traj_list[0], 1)

    n_traj_sample = len(sample_traj_list)
    n_rows = np.size(sample_traj_list[0], 0)
    one_idx_diff = []
    two_idx_diff = []

    # Transformation 1 for p_{i+1} 2.
    for traj in range(0, n_traj_sample):
        z = sample_traj_list[traj]
        two_idx_diff.append(ee_corr_reorder_trajectory(z))
        # Only use first row for subtration in EE numerator for radial design.
        z = np.tile(z[0, :], (n_rows, 1))
        one_idx_diff.append(ee_uncorr_reorder_trajectory(z))

    # Transformation 2 for p_{i+1}.
    # Need to reorder mu and covariance according to the two uncorrex difference by
    # using the invese function as for p_i in `the function for the uncorrependent EEs.
    for traj in range(0, n_traj_sample):
        # Needs to be set up again for each traj because otherwise it'd be one too much.
        mu_two = reverse_reorder_mu(mu)
        cov_two = reverse_reorder_cov(cov)
        for row in range(0, n_rows):
            two_idx_diff[traj][row, :], _ = transform_stnormal_normal_corr(
                two_idx_diff[traj][row, :], cov_two, mu_two
            )
            mu_two = reorder_mu(mu_two)
            cov_two = reorder_cov(cov_two)

    # Transformation 2 for p_{i+1}.
    # No re-arrangement needed as the first transformation for p_{i+1}
    # is using the original order of mu and cov.
    for traj in range(0, n_traj_sample):
        # Needs to be set up again for each traj because otherwise it'd be one `i`too much.
        mu_one = mu
        cov_one = cov
        for row in range(0, n_rows):
            (
                one_idx_diff[traj][row, :],
                _
            ) = transform_stnormal_normal_corr(
                one_idx_diff[traj][row, :], cov_one, mu_one
            )
            mu_one = reorder_mu(mu_one)
            cov_one = reorder_cov(cov_one)

    # # Transformation 3: Undo Transformation 1.
    trans_piplusone_iminusone = []
    trans_piplusone_i = []
    for traj in range(0, n_traj_sample):
        trans_piplusone_iminusone.append(
            reverse_ee_corr_reorder_trajectory(two_idx_diff[traj])
        )
        trans_piplusone_i.append(
        reverse_ee_uncorr_reorder_trajectory(one_idx_diff[traj])
        )

    return trans_piplusone_iminusone, trans_piplusone_i
