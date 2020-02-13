"""From Trajectory samples, compute EE-based measures for QoI from KW94."""

import pickle
import pandas as pd

import time


from multi_quantities_of_interest import  multi_quantities_of_interest
from sampling_schemes import trajectory_sample
from select_sample_set import select_sample_set_normal
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

# Traj-exclusive params.
n_sample_traj = 200
n_sample_select = 100
n_levels = 100
numeric_zero = 0.005
seed = 123



# Normal = False because we need to have stnormal ones for post-selection
traj_list, _ = trajectory_sample(n_sample_traj, n_inputs, n_levels, seed, normal=False, numeric_zero = numeric_zero)


# Implement sample post-selection and convert elements from [0,1] to stnormal space.
traj_list, step_list_traj = select_sample_set_normal(traj_list, n_sample_select, numeric_zero)


# Save post-selected trajectories and steps in input dir.
with open('input/selected_trajs.pkl', 'wb') as f:
  pickle.dump(traj_list, f)
  
with open('input/selected_steps_traj.pkl', 'wb') as f:
  pickle.dump(step_list_traj, f)


start = time.time()

measures_list_traj, obs_list_traj = screening_measures(wrapper_qoi, traj_list, step_list_traj, cov, mean, radial=False)

end = time.time()
print(end - start)
comp_time= (end - start)/60


# Save observations to results dir.
with open('results/ee_obs_traj.pkl', 'wb') as f:
  pickle.dump(obs_list_traj, f)
