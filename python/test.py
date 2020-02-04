"""Tests for `screening_measures_gm_2017`."""
import numpy as np
from screening_measures_gm_2017 import screening_measures_gm_2017

def sobol_model(a, b, c, d, e, f, *args):
    """
    Tested by comparing graphs for 3 specifications to book.
    Arguments are lists. Strongly nonlinear, nonmonotonic, and nonzero interactions.
    Analytic results for Sobol Indices.

    """
    input_pars = np.array([a, b, c, d, e, f])
    coeffs = np.array([78, 12, 0.5, 2, 97, 33])

    def g_i(input_pars, coeffs):
        return (abs(4 * input_pars - 2) + coeffs) / (1 + coeffs)

    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs[i])

    return y




def sobol_model_one(f, b, a, d, c, e, *args):
    """Reordered because trajectory in primer shuffled the trajectory columns"""
    input_pars = np.array([f, b, a, d, c, e])
    coeffs = np.array([33, 12, 78, 2, 0.5, 97])

    def g_i(input_pars, coeffs):
        return (abs(4 * input_pars - 2) + coeffs) / (1 + coeffs)

    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs[i])

    return y

steps_one = np.array([2/3, -2/3, -2/3, 2/3, 2/3, 2/3])


def sobol_model_two(b, c, a, f, e, d, *args):
    """Reordered because trajectory in primer shuffled the trajectory columns"""
    input_pars = np.array([b, c, a, f, e, d])
    coeffs = np.array([12, 0.5, 78, 33, 97, 2])

    def g_i(input_pars, coeffs):
        return (abs(4 * input_pars - 2) + coeffs) / (1 + coeffs)

    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs[i])

    return y

steps_two = np.array([2/3, 2/3, 2/3, -2/3, -2/3, -2/3])




def sobol_model_three(d, a, e, b, c, f, *args):
    """Reordered because trajectory in primer shuffled the trajectory columns"""
    input_pars = np.array([d, a, e, b, c, f])
    coeffs = np.array([2, 78, 97, 12, 0.5, 33])

    def g_i(input_pars, coeffs):
        return (abs(4 * input_pars - 2) + coeffs) / (1 + coeffs)

    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs[i])

    return y

steps_three = np.array([-2/3, -2/3, 2/3, -2/3, -2/3, +2/3])

def sobol_model_four(f, c, d, e, b, a, *args):
    """Reordered because trajectory in primer shuffled the trajectory columns"""
    input_pars = np.array([f, c, d, e, b, a])
    coeffs = np.array([33, 0.5, 2, 97, 12, 78])

    def g_i(input_pars, coeffs):
        return (abs(4 * input_pars - 2) + coeffs) / (1 + coeffs)

    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs[i])

    return y

steps_four = np.array([-2/3, -2/3, -2/3, -2/3, 2/3, -2/3])



cov = np.zeros(36).reshape(6, 6)
np.fill_diagonal(cov, np.ones(5))

# Not the expectation for x \in U[0,1]. Yet, prevents transformation.
mu = np.array([0, 0, 0, 0, 0, 0])

numeric_zero = 0.00001
seed = 2020
n_levels = 4
n_inputs = 6


traj_one = np.array([[0, 2/3, 1, 0, 0, 1/3],
                     [0, 2/3, 1, 0, 0, 1],
                     [0, 0, 1, 0, 0, 1],
                     [2/3, 0, 1, 0, 0, 1],
                     [2/3, 0, 1, 2/3, 0, 1],
                     [2/3, 0, 1/3, 2/3, 0, 1],
                     [2/3, 0, 1/3, 2/3, 2/3, 1]])
    
idx_one = [5,1,0,3,2,4]
traj_one = traj_one[:, idx_one]

traj_two = np.array([[0, 1/3, 1/3, 1, 1, 2/3],
                    [0, 1, 1/3, 1, 1, 2/3],
                    [0, 1, 1, 1, 1, 2/3],
                    [2/3, 1, 1, 1, 1, 2/3],
                    [2/3, 1, 1, 1, 1, 0],
                    [2/3, 1, 1, 1, 1/3, 0],
                    [2/3, 1, 1, 1/3, 1/3, 0]])

idx_two = [1,2,0,5,4,3]
traj_two = traj_two[:, idx_two]    
    
    
    
traj_three = np.array([[1, 2/3, 0, 2/3, 1, 0],
                    [1, 2/3, 0, 0, 1, 0],
                    [1/3, 2/3, 0, 0, 1, 0],
                    [1/3, 2/3, 0, 0, 1/3, 0],
                    [1/3, 0, 0, 0, 1/3, 0],
                    [1/3, 0, 2/3, 0, 1/3, 0],
                    [1/3, 0, 2/3, 0, 1/3, 2/3]])

idx_three = [3,0,4,1,2,5]
traj_three = traj_three[:, idx_three]    
    
    
traj_four = np.array([[1, 1/3, 2/3, 1, 0, 1/3],
                    [1, 1/3, 2/3, 1, 0, 1],
                    [1, 1/3, 0, 1, 0, 1],
                    [1, 1/3, 0, 1/3, 0, 1],
                    [1, 1/3, 0, 1/3, 2/3, 1],
                    [1, 1, 0, 1/3, 2/3, 1],
                    [1/3, 1, 0, 1/3, 2/3, 1]])


idx_four = [5,2,3,4,1,0]
traj_four = traj_four[:, idx_four]   

(
    one_ee_ind,
    one_ee_full,
    one_abs_ee_ind,
    one_abs_ee_full,
    one_sd_ee_ind,
    one_sd_ee_full,
) = screening_measures_gm_2017(
    sobol_model_one,
    [traj_one],
    [steps_one],
    n_levels,
    cov,
    mu,
    numeric_zero=0.00,
)



(
    two_ee_ind,
    two_ee_full,
    two_abs_ee_ind,
    two_abs_ee_full,
    two_sd_ee_ind,
    two_sd_ee_full,
) = screening_measures_gm_2017(
    sobol_model_two,
    [traj_two],
    [steps_two],
    n_levels,
    cov,
    mu,
    numeric_zero=0.00,
)


(
    three_ee_ind,
    three_ee_full,
    three_abs_ee_ind,
    three_abs_ee_full,
    three_sd_ee_ind,
    three_sd_ee_full,
) = screening_measures_gm_2017(
    sobol_model_three,
    [traj_three],
    [steps_three],
    n_levels,
    cov,
    mu,
    numeric_zero=0.00,
)

(
    four_ee_ind,
    four_ee_full,
    four_abs_ee_ind,
    four_abs_ee_full,
    four_sd_ee_ind,
    four_sd_ee_full,
) = screening_measures_gm_2017(
    sobol_model_four,
    [traj_four],
    [steps_four],
    n_levels,
    cov,
    mu,
    numeric_zero=0.00,
)


ee_one = np.array(one_ee_ind).reshape(6,1)

ee_two = np.array([two_ee_ind[1], two_ee_ind[2], two_ee_ind[0], two_ee_ind[5], two_ee_ind[4], two_ee_ind[3]]).reshape(6,1)[idx_two]

ee_three = np.array([three_ee_ind[1], three_ee_ind[3], three_ee_ind[4], three_ee_ind[0], three_ee_ind[2], three_ee_ind[5]]).reshape(6,1)[idx_three]

ee_four = np.array([four_ee_ind[5], four_ee_ind[4], four_ee_ind[1], four_ee_ind[2], four_ee_ind[3], four_ee_ind[0]]).reshape(6,1)[idx_four]

ee_i = np.concatenate((ee_one, ee_two, ee_three, ee_four), axis=1)



ee = np.ones(6).reshape(6, 1) * np.nan

ee = np.mean(ee_i, axis=1).reshape(6,1)

ee_abs = np.mean(abs(ee_i), axis=1).reshape(6,1)

ee_sd = np.sqrt(np.var(ee_i, axis=1)).reshape(6,1)



for i in range(0,7):
    print(sobol_model_four(*traj_four[i, :]))



coeffs = np.array([78, 12, 0.5, 2, 97, 33])

#coeffs_one = np.array([33, 12, 78, 2, 0.5, 97])
#coeffs_two = coeffs[idx_two]





