"""empty."""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sampling_schemes import trajectory_sample
from sampling_schemes import radial_sample
from screening_measures import screening_measures
from transform_reorder import reorder_cov
from transform_distributions import covariance_to_correlation

"""Example from Ge/Menendez (2017)"""
def linear_function(a, b, c, *args):
    return a + b + c

mu = np.array([0, 0, 0])
"""
cov = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
"""
# Number oflevels rises if there are different covarianzes

cov = np.array(
    [
        [1.0, 0.9, 0.4],
        [0.9, 1.0, 0.0],
        [0.4, 0.0, 1.0],
    ]
)
numeric_zero = 0.01
seed = 2020
n_levels = 10
n_inputs = 3
n_sample = 10_0

#traj_list, step_list = trajectory_sample(n_sample, n_inputs, n_levels, seed, True, numeric_zero)
traj_list, step_list = radial_sample(n_sample, n_inputs, seed, True, numeric_zero)

ee_ind, ee_full, abs_ee_ind, abs_ee_full, sd_ee_ind, sd_ee_full = screening_measures(linear_function, traj_list, step_list, cov, mu, radial = True)



from transform_ee import trans_ee_corr

a, b = trans_ee_corr(traj_list, cov, mu, radial=True)