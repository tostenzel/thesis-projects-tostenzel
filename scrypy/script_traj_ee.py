"""From Trajectory samples, compute EE-based measures for QoI from KW94."""

import pickle
import pandas as pd

import time


from multi_quantities_of_interest import  multi_quantities_of_interest
from sampling_schemes import trajectory_sample
from select_sample_set import intermediate_ge_menendez_2014

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
n_sample_select = 300
n_sample_traj = 150
n_levels = 100
numeric_zero = 0.005
seed = 123




traj_list, step_list_traj = trajectory_sample(n_sample_select, n_inputs, n_levels, seed, normal=True, numeric_zero = numeric_zero)

# Implement sample post-selection.
traj_list, _, select_indices = intermediate_ge_menendez_2014(traj_list, n_sample_traj)
step_list_traj = [step_list_traj[idx] for idx in select_indices]


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


# Save measures and observations to results dir.
with open('results/measures_traj.pkl', 'wb') as f:
  pickle.dump(measures_list_traj, f)

with open('results/ee_obs_traj.pkl', 'wb') as f:
  pickle.dump(obs_list_traj, f)

# Load results to check.
with open('results/measures_traj.pkl', 'rb') as f:
  read_meas = pickle.load(f)
  
with open('results/ee_obs_traj.pkl', 'rb') as f:
  read_ee_obs = pickle.load(f)
  
