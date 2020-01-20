"""Ge/Menendez (2017)"""
import numpy as np

from scipy.stats import norm
import scipy.linalg as linalg

from sampling_trajectory import morris_trajectory
from sampling_trajectory import intermediate_ge_menendez_2014


# Transformation 1: Shift the first \omega elements to the back to generate
# an independent vector.
def transformation_one(traj, p_i_plus_one=False):
    traj_trans_one = np.ones([np.size(traj, 0), np.size(traj, 1)]) * np.nan
    for i in range(0, np.size(traj, 0)):
        # move FIRST w elements to the BACK
        if p_i_plus_one is False:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i + 1))
        if p_i_plus_one is True:
            traj_trans_one[i, :] = np.roll(traj[i, :], -(i))
    return traj_trans_one


# Transformation 3: Undo Transformation 1.
def rev_transformation_one(traj, p_i_plus_one=False):
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

    sample_traj_list.append(
        morris_trajectory(n_inputs=3, n_levels=6)
    )
    
gm14_array, gm14_list, gm14_pairs_dist = intermediate_ge_menendez_2014(
    sample_traj_list, n_traj=5
    )


traj = sample_traj_list[0]

traj_trans_one = transformation_one(traj)
traj_trans_one_compare = transformation_one(traj, p_i_plus_one=True)
traj_trans_rev = rev_transformation_one(traj_trans_one)
traj_trans_rev_compare = rev_transformation_one(
    traj_trans_one_compare, p_i_plus_one=True
)


"""Try tranformation 2"""
# Check in mu and cov.
cov = np.array(
          [[1, 0.9, 0.4],
           [0.9, 1, 0.01],
           [0.4, 0.01, 1]])
mu = np.zeros(np.size(cov, 0))

# Need to replace ones, because erfinv(1) = inf and zeros because erfinv(0) = -inf
traj_trans_approx = np.where(traj_trans_one == 1, 0.999, traj_trans_one)
traj_trans_approx = np.where(traj_trans_approx == 0, 0.001, traj_trans_approx)

# Step 1: Inverse cdf of standard normal distribution (N(0, 5)).
z = norm.ppf(traj_trans_approx)

# Step 2. Skipped transformation of covariance matrix for normally distributed paramters.
r_z = cov


# Step 3: Perform Cholesky decoposition of (transformed) covariance matrix for
# upper triangular matrix.
n_par = np.size(r_z, 0)
n_draws = 10000

m = linalg.cholesky(r_z, lower=False)


# Step 4: Draw random vector from standard normal distribution for each paramter
# and calculate the corresponding correlation matrix. Then do the same as in
# Step 3 for the correlation matrix.
c_1 = np.corrcoef(r_z)

sample_random_normal_paramters = np.random.normal(0, 1, n_par*n_draws).reshape(n_par, n_draws)

c = np.corrcoef(sample_random_normal_paramters)

q = linalg.cholesky(c, lower=False)

# Step 5: Derive the dependent normally distributed vector z_c = z*c^(-1)*m.
z_c = np.ones(traj.shape) * np.nan
for row in range(0, n_par + 1):
    zq = np.dot(z[row, :], linalg.inv(q))
    z_c[row, :] = np.dot(zq, m)

# Step 6: Apply inverse Nataf transformation.
phi_z_c = norm.cdf(z_c)

x = np.ones(traj.shape) * np.nan
for row in range(0, n_par + 1):
    x[row, :] = mu + np.dot(norm.ppf(phi_z_c)[row, :], np.sqrt(cov))



