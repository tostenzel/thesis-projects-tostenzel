"""Tests for `screening_measures_gm_2017`."""
import numpy as np

from sampling_trajectory import morris_trajectory
from screening_measures_gm_2017 import screening_measures_gm_2017



def lin_portfolio(q1, q2, c1=2, c2=1, *args):
    """Simple function with analytic EE solution to support testing."""
    return c1 * q1 + c2 * q2

seed = 2020
n_levels = 10
n_inputs = 3
n_traj_sample = 10_000
sample_traj_list = list()


cov = np.array([
        [1, 0],
        [0, 9]])

# mu does not matter because the function is linear. You subtract what you add.
mu = np.array([0, 0])

seed = 2020
n_levels = 6
n_inputs = 2
n_traj_sample = 10_000

sample_traj_list = list()

for traj in range(0, n_traj_sample):
    seed = seed + traj
    sample_traj_list.append(morris_trajectory(n_inputs, n_levels, seed, True, cov, mu, numeric_zero=0.00000000001))

ee_ind, ee_full, abs_ee_ind, abs_ee_full, sd_ee_ind, sd_ee_full = screening_measures_gm_2017(
        lin_portfolio,
        sample_traj_list,
        n_levels,
        cov, mu,
        numeric_zero=0.01,
        normal=True
    )
