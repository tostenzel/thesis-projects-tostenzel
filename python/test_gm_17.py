"""Ge/Menendez (2017) test function standard normal"""

"""As written in Ge/Menendez (2017), page 34: The elements in vectors T(p_{i}, i) and
T(p_{i+1}, i) are the same except of the ith element."""

import pickle

import numpy as np

from sampling_trajectory import stepsize
from sampling_trajectory import morris_trajectory
from sampling_trajectory import intermediate_ge_menendez_2014
from transform_traj_elementary_effects import trans_ee_ind_trajectories
from transform_traj_elementary_effects import trans_ee_full_trajectories




n_levels = 10
n_inputs=3

n_traj_sample = 10000
sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj
    sample_traj_list.append(morris_trajectory(n_inputs, n_levels, seed=seed))
"""
_, opt_traj_list, _ =  intermediate_ge_menendez_2014(sample_traj_list, 150)


# Convert into string representation and save it as pickled .txt
with open("results/test_opt_traj_list_gm17.txt", "wb") as fp:   #Pickling
    pickle.dump(opt_traj_list, fp)


with open("results/test_opt_traj_list_gm17.txt", "rb") as fp:   # Unpickling
    opt_traj_list = pickle.load(fp)
""" 
  
cov = np.array(
    [
        [1.0, 0.9, 0.4],
        [0.9, 1.0, 0.01],
        [0.4, 0.01, 1.0],
    ]
)

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


opt_traj_list = sample_traj_list
#n_trajs = 9
n_trajs = 10000

opt_traj_list = opt_traj_list[0:n_trajs]

trans_pi_i_list, trans_piplusone_i_list = trans_ee_ind_trajectories(opt_traj_list, cov, mu)

#trans_piplusone_iminusone_list = trans_ee_full_trajectories(opt_traj_list, cov)


n_rows = np.size(opt_traj_list[0], 0)

def linear_function(a, b, c, *args):
    return 2*a + b + c

function_evals_pi_i = np.ones([n_rows, n_trajs]) * np.nan
function_evals_piplusone_i = np.ones([n_rows, n_trajs]) * np.nan
#function_evals_piplusone_iminusone = np.ones([n_rows, n_trajs]) * np.nan




for traj in range(0, n_trajs):
    for row in range(0, n_rows):
        function_evals_pi_i[row,traj] = linear_function(*trans_pi_i_list[traj][row,:])
        function_evals_piplusone_i[row,traj] = linear_function(*trans_piplusone_i_list[traj][row,:])
        #function_evals_piplusone_iminusone[row,traj] = linear_function(*trans_piplusone_iminusone_list[traj][row,:])

step = stepsize(n_levels)
""" / step commented out """
ee_ind_i = np.ones([n_inputs, n_trajs]) * np.nan
ee_ind = np.ones([n_inputs, 1]) * np.nan
for traj in range(0, n_trajs):
    ee_ind_i[:, traj] = (function_evals_piplusone_i[1:n_inputs + 1, traj] - function_evals_pi_i[0:n_inputs, traj]) #/ step
ee_ind[:,0] = np.mean(abs(ee_ind_i), axis = 1)   
