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


_, abs_ee_mean_traj_uncorr, _ = compute_measures(traj_ee_obs[0], sd_x, sd_y, sigma_norm=True, ub=False)

_, abs_ee_mean_traj_corr, _ = compute_measures(traj_ee_obs[1], sd_x, sd_y, sigma_norm=True, ub=False)


_, abs_ee_mean_rad_uncorr, _ = compute_measures(rad_ee_obs[0], sd_x, sd_y, sigma_norm=True)

_, abs_ee_mean_rad_corr, _ = compute_measures(rad_ee_obs[1], sd_x, sd_y, sigma_norm=True)


measures_to_plot = [abs_ee_mean_traj_uncorr, abs_ee_mean_traj_corr, abs_ee_mean_rad_uncorr, abs_ee_mean_rad_corr]

# Save observations to results dir.
with open('results/measures_to_plot.pkl', 'wb') as f:
  pickle.dump(measures_to_plot, f)
