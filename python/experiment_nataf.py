"""Experiment with Nataf transformation."""
import numpy as np
from scipy.stats import norm

from ERANataf import ERANataf
from ERADist import ERADist

cov = np.array([
        [1,0,0,0,0.45],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0.45,0,0,0,1]])


mu = np.array([0,0,0,0,0])
row_traj_reordered = np.array([0.9, 0.1, 0.1, 0.1, 0.8])
expected = np.dot((norm.ppf(row_traj_reordered) + mu), np.sqrt(cov))

M = list()
M.append(ERADist('normal', 'PAR', [0, 1]))
M.append(ERADist('normal', 'PAR', [0, 1]))
M.append(ERADist('normal', 'PAR', [0, 1]))
M.append(ERADist('normal', 'PAR', [0, 1]))
M.append(ERADist('normal', 'PAR', [0, 1]))



# Correlation matrix.
Rho = cov

# applying Nataf transformation
T_Nataf = ERANataf(M, Rho)

# samples in physical space
X = T_Nataf.U2X(row_traj_reordered)