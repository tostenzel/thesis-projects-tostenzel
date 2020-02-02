"""Test sampling_trajectory.py"""
import sys

# Define parent folder as relative path.
sys.path.append("..")

import numpy as np

from sampling_trajectory import stepsize
from sampling_trajectory import morris_trajectory


def test_morris_trajectory_value_grid():
    n_levels = 10
    # Many inputs for high probability to catch all grid points in trajectory.
    n_inputs = 100

    traj, _ = morris_trajectory(n_inputs, n_levels, seed=123)

    # Round the elements in both sets.
    grid_flat_list = [round(item, 6) for sublist in traj.tolist() for item in sublist]
    grid = set(grid_flat_list)

    expected = np.around((np.linspace(0, 9, 10) / (n_levels - 1)), 6)
    expected = set(expected.tolist())

    assert grid == expected
