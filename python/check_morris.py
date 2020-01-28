"""Check Morris screening."""
from collections import Counter


import numpy as np

from sampling_trajectory import stepsize
from sampling_trajectory import morris_trajectory

n_levels = 10
n_inputs=10


step = stepsize(n_levels)

traj = morris_trajectory(n_inputs, n_levels, seed=123)

value_grid = [0, 1 - step]
idx = 1
while idx / (n_levels - 1) < 1 - step:
    value_grid.append(idx / (n_levels - 1))
    idx = idx + 1
    
flat_list = [item for sublist in traj.tolist() for item in sublist]

print(set(flat_list))

assert float(6/2).is_integer()
