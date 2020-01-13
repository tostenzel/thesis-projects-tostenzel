"""Winding stairs sampling + Morris(1991) improvement + Campolongo(2007) improvement"""
import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom


def stepsize(n_levels):
    """
    Book recommendation:
    Leads not to equiprobable sampling from the input distribution.
    and is HALFWAY EQUISPACED.

    """
    return n_levels / (2 * (n_levels - 1))


def stepsize_equidistant(n_levels):
    """
    Leads to concentration at middle-sized values in the sample.
    The reason is the following:
    Imagine the levels are linspace(0,1,5).
    Also imagine the first parameter starts as zero. Then stepsize 0.2 is added.
    This yields to an additional number of 0.2s.
    The number of additional 0.2 in this case equals the number of paramters.
    This can not happen to 0.0s. If the stepsize does not yield repetitive
    values, there is no equidistance.

    """
    return 1 / (n_levels - 1)


def morris_trajectories(
    n_inputs, n_levels, step_function, stairs=True, seed=123, test=False
):
    """
    Returns n parameter vectors, Dim n x Theta.
    n is also Theta+1.
    Uses stepsize function.

    IMPORTANT:
    - Shuffling of identity matrix is turned off to obtain stairs
    for the Qiao We / Menendez (2016) paper for correlated paramters.
    - The levels have not to be equidistant because the first level after zero
    is the step. Thereafter, the levels are equispaced. Therefore, one may want
    to adjust the stepsize to the number of levels by hand instead of following
    the above function.
    - If the stepsize equals the distance between the other levels,
    then level 0 is less frequent.

    """
    np.random.seed(seed)
    step = step_function(n_levels)
    #  B is (p+1)*p strictily lower triangular matrix of ones.
    B = np.tril(np.ones([n_inputs + 1, n_inputs]), -1)
    # J is (p+1)*p matrix of ones.
    J = np.ones([n_inputs + 1, n_inputs])
    # Matrix of zeros with random choices between -1 and 1 on main
    # diagonal.
    if test is False:
        # Choose a random value from the differenent Elementary Effects.
        # Must be lower than 1 - step otherwise one could sample values > 1.
        # base_value rand \in [i/(1-step)]
        # !!!The upper bound must be 1 - step, because step is added on top of the
        # values!!!
        value_grid = [0, 1 - step]
        idx = 1
        while idx / (n_levels - 1) < 1 - step:
            value_grid.append(idx / idx / (n_levels - 1))
            idx = idx + 1
        base_value_vector_rand = np.array(random.choices(value_grid, k=n_inputs))
        # Influenced by seed.
        P_star_rand = np.identity(n_inputs)
        D_star_rand = np.zeros([n_inputs, n_inputs])
        np.fill_diagonal(D_star_rand, random.choices([-1, 1], k=n_inputs))
        """Shuffle columns: Commented Out to get simple stairs form!!!"""
        if stairs is False:
            np.random.shuffle(P_star_rand.T)
        else:
            pass
    else:
        base_value_vector_rand = [1 / 3] * 2
        P_star_rand = np.identity(n_inputs)
        D_star_rand = np.array([[1, 0], [0, -1]])
    # Be careful with np.dot vs. np.matmul vs. *.
    B_star_rand = np.dot(
        J * base_value_vector_rand
        + (step / 2) * (np.dot((2 * B - J), D_star_rand) + J),
        P_star_rand,
    )
    return B_star_rand



traj = morris_trajectories(
    n_inputs=5,
    n_levels=6,
    step_function=stepsize_equidistant,
)

# Transformation 1: Shift the first \omega elements to the back to generate
# an independent vector.
def transformation_one(traj, p_i_plus_one=False):
    traj_trans_one = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
            # move FIRST w elements to the BACK
            if p_i_plus_one is False:
                traj_trans_one[i,:] = np.roll(traj[i,:], -(i+1))
            if p_i_plus_one is True:
                traj_trans_one[i,:] = np.roll(traj[i,:], -(i))
    return traj_trans_one

# Transformation 3: Undo Transformation 1.
def transformation_three(traj, p_i_plus_one=False):
    traj_trans_three = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        # move LAST w elements to the FRONT
        if p_i_plus_one is False:
            traj_trans_three[i,:] = np.roll(traj[i,:], -(np.size(traj, 1) - (i+1)))
        if p_i_plus_one is True:
            traj_trans_three[i,:] = np.roll(traj[i,:], -(np.size(traj, 1) - (i)))
    return traj_trans_three

traj_trans_one =  transformation_one(traj)
traj_trans_one_compare =  transformation_one(traj, p_i_plus_one=True)
traj_trans_three =  transformation_three(traj_trans_one)
traj_trans_three_compare =  transformation_three(traj_trans_one_compare, p_i_plus_one=True)



"""Try tranformation 2"""
import scipy.special
from scipy.stats import multivariate_normal

cov = np.eye(np.size(traj, 1))
#cov[0, 0] = 1
#cov[1, 1] = 1

mean = np.array([0, 0, 0, 0, 0])
# Need to replace ones, because erfinv(1) = inf and zeros because erfinv(0) = -inf
traj_trans_wo_ones = np.where(traj_trans_one==1, 0.99, traj_trans_one)
traj_trans_wo_ones = np.where(traj_trans_wo_ones==0, 0.01, traj_trans_wo_ones) 




rbt_0_0 = mean[0] + (cov[0,0] * np.sqrt(2) * scipy.special.erfinv(2*traj_trans_wo_ones[0,0] - 1)) 

rbt_1_1 = (
        mean[1] + np.dot((cov[1,1] * np.sqrt(2)), scipy.special.erfinv(2*traj_trans_wo_ones[0,1] - 1))
        ) * multivariate_normal.pdf(rbt_0_0, mean[0], cov[0,0])

rbt_2_2 = (
        mean[2] + np.dot((cov[2,2] * np.sqrt(2)), scipy.special.erfinv(2*traj_trans_wo_ones[0,2] - 1))
        ) * multivariate_normal.pdf(np.array([rbt_0_0, rbt_1_1]), mean[0:2], cov[0:2,0:2])






