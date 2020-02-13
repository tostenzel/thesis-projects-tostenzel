"""Analyse results."""
import pickle
import numpy as np

from screening_measures import compute_measures

# Loadings.

# Load standard deviations of input paramters.
with open('results/10_000_draws/mc_change_mean_edu_df.pkl', 'rb') as f:
  sample_qoi = pickle.load(f)

sd_y = np.sqrt(np.var(sample_qoi.loc['change_avg_schooling']))

# Load standard deviations of input paramters.
with open('input/est_rand_params_chol.pkl', 'rb') as f:
  params = pickle.load(f)

sd_x = params['sd'].to_numpy()

# Load Trajectory EE obs..  
with open('results/ee_obs_traj.pkl', 'rb') as f:
  traj_ee_obs = pickle.load(f)
  
# Load Radial EE obs..
with open("results/ee_obs_radial.pkl", "rb") as f:
    rad_ee_obs = pickle.load(f)

ee_mean_traj_uncorr, abs_ee_mean_traj_uncorr, sd_ee_traj_uncorr = compute_measures(traj_ee_obs[0], sd_x, sd_y, sigma_norm=False, ub=True)

ee_mean_traj_corr, abs_ee_mean_traj_corr, sd_ee_traj_corr = compute_measures(traj_ee_obs[1], sd_x, sd_y, sigma_norm=False, ub=True)


#ee_mean_rad_uncorr, abs_ee_mean_rad_uncorr, sd_ee_rad_uncorr = compute_measures(rad_ee_obs[0], sd_vector, sd_y, sigma_norm=True)

#ee_mean_rad_corr, abs_ee_mean_rad_corr, sd_ee_rad_corr = compute_measures(rad_ee_obs[1], sd_vector, sd_y, sigma_norm=True)
#n_inputs = np.size(traj_ee_obs[0], 0)

#norm = (sd_x / sd_y).reshape(n_inputs, 1)


