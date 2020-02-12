"""From Trajectory samples, compute EE-based measures for QoI from KW94."""

import pickle
import pandas as pd
import numpy as np
import time


from multi_quantities_of_interest import  multi_quantities_of_interest
from sampling_schemes import trajectory_sample
from select_sample_set import intermediate_ge_menendez_2014


from sampling_schemes import radial_sample
from screening_measures import screening_measures



params = pd.read_pickle("input/est_rand_params_chol.pkl")
mean = params["value"].to_numpy()
cov = pd.read_pickle("input/est_cov_chol.pkl").to_numpy()

n_inputs = len(mean)

# Traj-exclusive params.
n_sample_select = 300
n_sample_traj = 150
n_levels = 100
numeric_zero = 0.005
#n_sample_rad = 2
seed = 123


# Rad-exclusive params.
n_sample_rad = 2


traj_list, step_list_traj = trajectory_sample(n_sample_select, n_inputs, n_levels, seed, normal=True, numeric_zero = numeric_zero)

# Implement sample post-selection.
traj_list, _, select_indices = intermediate_ge_menendez_2014(traj_list, n_sample_traj)
step_list_traj = [step_list_traj[idx] for idx in select_indices]
#rad_list, step_list_rad = radial_sample(n_sample_rad, n_inputs, normal=True, numeric_zero=numeric_zero)


# Transform diffferent single arguments for `screening_measures` to a list for
#
def wrapper_qoi(*args):
    """
    Transform different single arguments for `screening_measures` to a list for
    `multi_quantities_of_interest`.
    Also, return only the first main qoi.
    
    """
    
    args_list = [args]
    return multi_quantities_of_interest(*args_list)[0]


start = time.time()


measures_list_traj, obs_list_traj = screening_measures(wrapper_qoi, traj_list, step_list_traj, cov, mean, radial=False)
#measures_list_rad, obs_list_rad = screening_measures(wrapper_qoi, rad_list, step_list_rad, cov, mean, radial = True)


end = time.time()
print(end - start)
comp_time= (end - start)/60



with open('results/measures_traj.pkl', 'wb') as f:
  pickle.dump(measures_list_traj, f)

with open('results/ee_obs_traj.pkl', 'wb') as f:
  pickle.dump(obs_list_traj, f)




with open('results/measures_traj.pkl', 'rb') as f:
  read_meas = pickle.load(f)
  
with open('results/ee_obs_traj.pkl', 'rb') as f:
  read_ee_obs = pickle.load(f)
  
