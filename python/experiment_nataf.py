"""Experiment with Nataf transformation."""
import numpy as np
import scipy.linalg as linalg
from ERADist import ERADist
from ERANataf import ERANataf
from ge_menendez_2017_traj_transformation import james_e_gentle_2005
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
X_check = james_e_gentle_2005(row_approx, cov)

"""backward james e gentle"""
# Transform sample from DEPENDENT standard normal space to INDEPENDENT standard normal space.
M_prime = linalg.cholesky(cov, lower=True)
inv_M_prime = linalg.inv(M_prime)

indie_z = np.dot(inv_M_prime,X)



"""(Backward) Inverse Nataf Transformation"""
