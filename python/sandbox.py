"""
n_traj_sample = 20
sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj_list.append(morris_trajectory(n_inputs=5, n_levels=6))


traj = sample_traj_list[0]

traj_trans_one = ee_ind_reorder_trajectory(traj)
traj_trans_one_subtract = ee_ind_reorder_trajectory(traj, p_i_plus_one=False)
traj_trans_rev = reverse_ee_ind_reorder_trajectory(traj_trans_one)
traj_trans_rev_subtract = reverse_ee_ind_reorder_trajectory(traj_trans_one, p_i_plus_one=False)


full = ee_full_reorder_trajectory(traj)
full_subtract = ee_full_reorder_trajectory(traj, p_i_plus_one=False)

rev_full = reverse_ee_full_reorder_trajectory(full)
rev_full_subtract = reverse_ee_full_reorder_trajectory(full, p_i_plus_one=True)
"""
