"""
Re-ordering transformations

IMPORTANT REMARK: ee_ind_reorder_trajectory(traj, p_i_plus_one=True)
is equivalent to ee_full_reorder_trajectory(traj, p_i_plus_one=False).
The same holds true for the two respective inverse functions.

"""
import numpy as np


def ee_ind_reorder_trajectory(traj, p_i_plus_one=True):
    """
    Transformation 1 for the independent Elementary Effect.
    Move the first i elements to the back of the ith row.

    """
    traj_trans_one = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        if p_i_plus_one is False:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i + 1))
        if p_i_plus_one is True:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i))
    return traj_trans_one


def inverse_ee_ind_reorder_trajectory(traj, p_i_plus_one=True):
    """
    Transformation 3 for the independent Elementary Effect.
    Inverse of Transformation 1.

    """
    traj_trans_three = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        if p_i_plus_one is False:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i + 1)))
        if p_i_plus_one is True:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i)))
    return traj_trans_three


def ee_full_reorder_trajectory(traj, p_i_plus_one=True):
    """
    Transformation 1 for the full Elementary Effect.
    Move the first i-1 elements to the back of the ith row.

    """
    traj_trans_one = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        if p_i_plus_one is False:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i))
        if p_i_plus_one is True:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i - 1))
    return traj_trans_one


def inverse_ee_full_reorder_trajectory(traj, p_i_plus_one=True):
    """
    Transformation 3 for the full Elementary Effect.
    Inverse of Transformation 1.

    """
    traj_trans_three = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        if p_i_plus_one is False:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i)))
        if p_i_plus_one is True:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i - 1)))
    return traj_trans_three


def reorder_mu(mu):
    """Put the first element of the expectation vector to the end."""
    return np.roll(mu, -1)


def reorder_cov(cov):
    """Arrange covariance matrix according to the expectation vector when
    the first element is moved to the end.

    """
    cov_new = np.ones(cov.shape) * np.nan
    # Put untouched square one up and one left
    cov_new[0 : len(cov) - 1, 0 : len(cov) - 1] = cov[1 : len(cov), 1 : len(cov)]
    # Put [0,0] to [n,n]
    cov_new[len(cov) - 1, len(cov) - 1] = cov[0, 0]
    # Put [0, 1:n] to [n, 0:n-1] and same for the column.
    cov_new[len(cov) - 1, 0 : len(cov) - 1] = cov[0, 1 : len(cov)]
    cov_new[0 : len(cov) - 1, len(cov) - 1] = cov[0, 1 : len(cov)]

    return cov_new

def inverse_reorder_mu(mu):
    """
    Inverse function of `reorder_mu`
    Used to intialize the loop for
    `inverse_ee_full_reorder_trajectory(traj, p_i_plus_one=True)`
    """
    return np.roll(mu, +1)


def inverse_reorder_cov(cov):
    """
    Inverse function of `reorder_cov`
    Used to intialize the loop for
    `inverse_ee_full_reorder_trajectory(traj, p_i_plus_one=True)`
    """
    cov_old = np.ones(cov.shape) * np.nan
    cov_old[1 : len(cov) , 1 : len(cov)] = cov[0 : len(cov) - 1, 0 : len(cov) - 1]
    cov_old[0, 0] = cov[len(cov) - 1, len(cov) - 1]
    
    cov_old[0, 1 : len(cov)] = cov[len(cov) - 1, 0 : len(cov) - 1]
    cov_old[1 : len(cov), 0] = cov[0 : len(cov) - 1, len(cov) - 1]
    
    return cov_old