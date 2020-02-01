"""Tests for `screening_measures_gm_2017`."""
import numpy as np

from sampling_trajectory import morris_trajectory
from screening_measures_gm_2017 import screening_measures_gm_2017



def lin_portfolio(q1, q2, c1=2, c2=1, *args):
    """Simple function with analytic EE solution to support testing."""
    return c1 * q1 + c2 * q2




cov = np.array([
        [1, 0],
        [0, 9]])

# mu does not matter because the function is linear. You subtract what you add.
mu = np.array([0, 0])

numeric_zero = 0.01
seed = 2020
n_levels = 10
n_inputs = 2
n_traj_sample = 10_000

traj_list = list()
step_list = list()
for traj in range(0, n_traj_sample):
    seed = seed + traj
    m_traj, step = morris_trajectory(n_inputs, n_levels, seed, True, cov, mu, numeric_zero)
    traj_list.append(m_traj)
    step_list.append(step)


ee_ind, ee_full, abs_ee_ind, abs_ee_full, sd_ee_ind, sd_ee_full = screening_measures_gm_2017(
        lin_portfolio,
        traj_list,
        step_list,
        n_levels,
        cov, mu,
        numeric_zero=0.01,
        normal=True
    )
