"""Re-ordering transformations"""
import numpy as np
from sampling_trajectory import morris_trajectory


# Transformation 1: Shift the first \omega elements to the back to generate
# an independent vector.
def reorder_trajectory(traj, p_i_plus_one=False):
    traj_trans_one = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        # move FIRST w elements to the BACK
        if p_i_plus_one is False:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i + 1))
        if p_i_plus_one is True:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i))
    return traj_trans_one


# Transformation 3: Undo Transformation 1.
def reverse_reorder_trajectory(traj, p_i_plus_one=False):
    traj_trans_three = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        # move LAST w elements to the FRONT
        if p_i_plus_one is False:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i + 1)))
        if p_i_plus_one is True:
            traj_trans_three[i, :] = np.roll(traj[i, :], -(np.size(traj, 1) - (i)))
    return traj_trans_three


n_traj_sample = 20
sample_traj_list = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj_list.append(morris_trajectory(n_inputs=5, n_levels=6))


traj = reorder_trajectory(sample_traj_list[0])

row_traj_reordered = traj[0, :]
# traj_trans_one = reorder_trajectory(traj)
# traj_trans_one_compare = reorder_trajectory(traj, p_i_plus_one=True)
# traj_trans_rev = reverse_reorder_trajectory(traj_trans_one)
# traj_trans_rev_compare = reverse_reorder_trajectory(
#    traj_trans_one_compare, p_i_plus_one=True
# )
