"""Experiment with Nataf transformation."""
import numpy as np
import scipy.linalg as linalg
from ERADist import ERADist
from ERANataf import ERANataf
from ge_menendez_2017_traj_transformation import james_e_gentle_2006
from scipy.stats import norm


"""(Forward) Nataf Transformation"""

row_approx = np.array([0.1, 0.1, 0.2, 0.8, 0.5])
# Transform uniform variable to standard normal space.
z = norm.ppf(row_approx) 

M = list()
M.append(ERADist('normal', 'PAR', [0, 1]))
M.append(ERADist('normal', 'PAR', [0, 1]))
M.append(ERADist('normal', 'PAR', [0, 1]))
M.append(ERADist('normal', 'PAR', [0, 1]))
M.append(ERADist('normal', 'PAR', [0, 1]))


cov = np.array([
        [1,0,0,0.2,0.5],
        [0,1,0.4,0.15,0],
        [0,0.4,1,0.05,0],
        [0.2,0.15,0.05,1,0],
        [0.5,0,0,0,1]])


# Correlation matrix.
Rho = cov

# Applying Nataf transformation
T_Nataf = ERANataf(M, Rho)


# Transform sample from INDEPENDENT standard normal to DEPENDENT actual/physical space.
X = T_Nataf.U2X(z)

# Transform sample from uniform space to multivariate normal space, not necessarily independet or standard.
X_check = james_e_gentle_2006(row_approx, cov)

"""backward james e gentle"""
# Transform sample from DEPENDENT standard normal space to INDEPENDENT standard normal space.
M_prime = linalg.cholesky(cov, lower=True)
inv_M_prime = linalg.inv(M_prime)

indie_z = np.dot(inv_M_prime,X)



"""(Backward) Inverse Nataf Transformation"""
z_nataf = T_Nataf.X2U(X)





"""Try wether reverse order of paramaters yields different results."""

rev_row_approx = np.array([0.5, 0.8, 0.2, 0.1, 0.1])

rev_cov = np.array([
        [1, 0, 0, 0, 0.5],
        [0, 1, 0.05, 0.15, 0.2],
        [0, 0.05, 1, 0.4, 0],
        [0, 0.15, 0.4, 1, 0],
        [0.5, 0.2, 0, 0, 1]])
rev_z = norm.ppf(rev_row_approx) 

# Correlation matrix.
rev_Rho = rev_cov

# Applying Nataf transformation
T_Nataf = ERANataf(M, rev_Rho)


# Transform sample from INDEPENDENT standard normal to DEPENDENT actual/physical space.
rev_X = T_Nataf.U2X(rev_z)

# Transform sample from uniform space to multivariate normal space, not necessarily independet or standard.
rev_X_check = james_e_gentle_2006(rev_row_approx, rev_cov)

"""
Result: The order matters! The first element stays untouched. Also the previous paramters
influence the following but not vice versa!!!
--> The "Rosenblatt Transformation is not unique.
"""



"""(Backward) james e gentle (to standard normal!)"""
# Transform sample from DEPENDENT standard normal space to INDEPENDENT standard normal space.
rev_M_prime = linalg.cholesky(rev_cov, lower=True)
rev_inv_M_prime = linalg.inv(rev_M_prime)

rev_indie_z = np.dot(rev_inv_M_prime, rev_X)


"""(Backward) Inverse Nataf Transformation (to standard normal!)"""
rev_z_nataf = T_Nataf.X2U(rev_X)
