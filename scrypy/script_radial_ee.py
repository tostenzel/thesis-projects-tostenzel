"""From Radial samples, compute EE-based measures for single QoI from KW94."""
import pickle
import time

import pandas as pd
from multi_quantities_of_interest import multi_quantities_of_interest
from sampling_schemes import radial_sample
from screening_measures import screening_measures


params = pd.read_pickle("input/est_rand_params_chol.pkl")
mean = params["value"].to_numpy()
cov = pd.read_pickle("input/est_cov_chol.pkl").to_numpy()


def wrapper_qoi(*args):
    """
    Transform different single arguments for `screening_measures` to a list for
    `multi_quantities_of_interest`.
    Also, return only the first main qoi.
    
    """

    args_list = [args]
    return multi_quantities_of_interest(*args_list)[0]


n_inputs = len(mean)

# Rad-exclusive params.
n_sample_rad = 100


# traj_list, step_list_traj = trajectory_sample(n_sample_traj, n_inputs, n_levels, seed, normal=True, numeric_zero)
rad_list, step_list_rad = radial_sample(n_sample_rad, n_inputs, normal=True)

# Save measures and steps to results dir.
with open("input/sample_radial.pkl", "wb") as f:
    pickle.dump(rad_list, f)

with open("input/steps_radial.pkl", "wb") as f:
    pickle.dump(step_list_rad, f)


start = time.time()

measures_list_rad, obs_list_rad = screening_measures(
    wrapper_qoi, rad_list, step_list_rad, cov, mean, radial=True
)

end = time.time()
print(end - start)
comp_time = (end - start) / 60


# Save observations to results dir.
with open("results/ee_obs_radial.pkl", "wb") as f:
    pickle.dump(obs_list_rad, f)

