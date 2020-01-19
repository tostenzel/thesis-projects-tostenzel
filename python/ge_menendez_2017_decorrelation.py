"""Ge/Menendez (2017)"""
import random

import numpy as np
from sampling_trajectory import morris_trajectory
from sampling_trajectory import stepsize

traj = morris_trajectory(n_inputs=5, n_levels=6, step_function=stepsize)

# Transformation 1: Shift the first \omega elements to the back to generate
# an independent vector.
def transformation_one(traj, p_i_plus_one=False):
    traj_trans_one = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        # move FIRST w elements to the BACK
        if p_i_plus_one is False:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i + 1))
        if p_i_plus_one is True:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i))
    return traj_trans_one


# Transformation 3: Undo Transformation 1.
def transformation_three(traj, p_i_plus_one=False):
    traj_trans_three = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        # move LAST w elements to the FRONT
        if p_i_plus_one is False:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i + 1)))
        if p_i_plus_one is True:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i)))
    return traj_trans_three


traj_trans_one = transformation_one(traj)
traj_trans_one_compare = transformation_one(traj, p_i_plus_one=True)
traj_trans_three = transformation_three(traj_trans_one)
traj_trans_three_compare = transformation_three(
    traj_trans_one_compare, p_i_plus_one=True
)


"""Try tranformation 2"""
import scipy.special
from scipy.stats import multivariate_normal

cov = np.eye(np.size(traj, 1))
# cov[0, 0] = 1
# cov[1, 1] = 1

mean = np.array([0, 0, 0, 0, 0])
# Need to replace ones, because erfinv(1) = inf and zeros because erfinv(0) = -inf
traj_trans_approx = np.where(traj_trans_one == 1, 0.99, traj_trans_one)
traj_trans_approx = np.where(traj_trans_approx == 0, 0.01, traj_trans_approx)
