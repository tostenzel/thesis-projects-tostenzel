

n_inputs = 4
n_levels = 5
n_traj_sample = 5
n_traj = 3

sample_traj = list()
for traj in range(0, n_traj_sample):
    seed = 123 + traj

    sample_traj.append(
        morris_trajectory(n_inputs, n_levels, step_function=stepsize, seed=seed)
    )
traj_dist_matrix = distance_matrix(sample_traj)

assert np.all(np.abs(traj_dist_matrix - traj_dist_matrix.T) < 1e-8)
indices = list(np.arange(0, np.size(traj_dist_matrix, 1)))
combi = combi_wrapper(indices, n_traj_sample - 1)
assert len(combi) == binom(np.size(traj_dist_matrix, 1), n_traj_sample - 1)
# leave last column open for aggregate distance
combi_distance = np.ones([len(combi), len(combi)]) * np.nan
combi_distance[:, 0 : n_traj_sample - 1] = np.array(combi)
for row in range(0, len(combi)):
    # Assign last column
    combi_distance[row, n_traj_sample - 1] = 0
    pair_combi = combi_wrapper(combi[row], 2)
    for pair in pair_combi:

        combi_distance[row, n_traj_sample - 1] += (
            traj_dist_matrix[int(pair[0])][int(pair[1])] ** 2
        )
combi_distance[:, n_traj_sample - 1] = np.sqrt(
    combi_distance[
        :, n_traj_sample - 1
    ]  # Here was 0.5 * combi_distance. This might be wrong.
)
# Indices of combination that yields highest distance figure.
max_dist_indices_row = (
    combi_distance[:, n_traj_sample - 1].argsort()[-1:][::-1].tolist()
)
max_dist_indices = combi_distance[max_dist_indices_row, 0 : n_traj_sample - 1]
# Convert list of float indices to list of ints.
max_dist_indices = [int(i) for i in max_dist_indices.tolist()[0]]
select_trajs_iter = [sample_traj[j] for j in max_dist_indices]

combi_new = combi_wrapper(max_dist_indices, n_traj_sample - 2)
# leave last column open for aggregate distance
combi_distance_new = np.ones([len(combi_new), len(combi_new)]) * np.nan
combi_distance_new[:, 0 : n_traj_sample - 2] = np.array(combi_new).astype(int)
lost_traj_idx = [idx for idx in indices if idx not in max_dist_indices][0]


for row in range(0, np.size(combi_distance_new, 0)):
    sum_dist_squared = 0
    for col in range(0, np.size(combi_distance_new, 1) - 1):
        # Get the distance between lost index trajectory and present ones in row.
        sum_dist_squared += (
            traj_dist_matrix[int(combi_distance_new[row, col]), lost_traj_idx]
        ) ** 2
    # Search for the old combination of trajs with the lost index
    # to compute the new aggregate distance with the above distances.
    for row_old in range(0, np.size(combi_distance, 0)):
        old_indices = [
            float(x) for x in combi_distance_new[row, 0 : n_traj_sample - 2].tolist()
        ]
        old_indices.append(float(lost_traj_idx))
        if set(combi_distance[row_old, 0 : n_traj_sample - 1]) == set(old_indices):
            combi_distance_new[row, n_traj_sample - 2] = np.sqrt(
                combi_distance[row_old, n_traj_sample - 1] ** 2 - sum_dist_squared
            )
        else:
            pass