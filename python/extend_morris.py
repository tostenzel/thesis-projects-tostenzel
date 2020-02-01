"""
Extend function `morris_screening` to return a vector of transformed delta
with length equals number of input paramters and a vector of elements in {-1,1}
indicating if delta is added or subtracted.

"""
import numpy as np

from sampling_trajectory import morris_trajectory


n_inputs = 3
n_levels = 10

mu = np.array([0, 0, 0])

cov = np.array(
    [
        [2.0, 0.9, 0.4],
        [0.9, 2.0, 0.0],
        [0.4, 0.0, 2.0],
    ]
)

traj, steps = morris_trajectory(
    n_inputs, n_levels, seed=123, normal=True, cov=cov, mu=mu, numeric_zero=0.0001)
    




