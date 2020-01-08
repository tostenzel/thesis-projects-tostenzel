"""Morris Screening"""
import numpy as np


n_inputs = 2
n_levels = 4


def stepsize(n_levels):
    """Leads to relatively equiprobable sampling from the input distribution"""
    return n_levels / (2 * (n_levels - 1))


def morris_trajectories(
        init_input_pars, stepsize, seed=123, test_D_star_rand_2dim=False
        ):
    """Returns n parameter vectors, Dim n x Theta.
    n is also Theta+1."""
    n_inputs = len(init_input_pars)
    np.random.seed(seed)
    #  B is (p+1)*p strictily lower triangular matrix of ones.
    B = np.tril(np.ones([n_inputs + 1, n_inputs]), -1)
    # J is (p+1)*p matrix of ones.
    J = np.ones([n_inputs + 1, n_inputs])
    # Influenced by seed. Seed 123 generates P_star in book.
    P_star_rand = np.identity(n_inputs)
    # Shuffle columns
    np.random.shuffle(P_star_rand.T)

    if test_D_star_rand_2dim is True:
        D_star_rand = np.array([[1, 0], [0, -1]])
    else:
        D_star_rand = np.array(np.random.uniform(-1, 1, n_inputs * n_inputs)).reshape(
            (n_inputs, n_inputs)
        )

    B_star_rand = np.matmul(
        J * init_input_pars
        + (stepsize / 2) * (np.matmul((2 * B - J), D_star_rand) + J),
        P_star_rand,
    )
    return B_star_rand


def elementary_effect_i(model, i_python, init_input_pars, stepsize):
    vector_e = np.zeros(len(init_input_pars))
    vector_e[i_python] = 1
    step_input_pars = init_input_pars + (vector_e * stepsize)

    return (model(*step_input_pars.tolist()) - model(*init_input_pars)) / stepsize


def scaled_elementary_effect_i(
    model, i_python, init_input_pars, stepsize, sd_i, sd_model
    ):
    """Scales EE by (SD_i / SD_M)"""
    ee_i = elementary_effect_i(model, i_python, init_input_pars, stepsize)

    return ee_i * (sd_i / sd_model)

def sobol_model(input_pars, coeffs_a):
    """
    Arguments are lists. Strongly nonlinear, nonmonotonic, and nonzero interactions.
    Analytic results for Sobol Indices.
    """
    def g_i(input_par_i, coeffs_a_i):
        return (abs(4 * input_par_i - 2) + coeffs_a_i) / (1 + coeffs_a_i)
    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs_a[i])
    return y


# afterwards implement this for all elements of a traj list.



def compute_trajectory_distance(traj_0, traj_1):
    """
    Computes the sum of the root of the square distance between each
    parameter vector of one trajectory to each vector of the other trajectory.
    Trajectories are np.Arrays with iterations as rows and parameters as vectors.
    
    """
    distance = 0
    
    assert np.size(traj_0, 0) == np.size(traj_0, 1) + 1
    assert traj_0.shape == traj_1.shape
    if np.any(np.not_equal(traj_0, traj_1)):
        for col_0 in range(0,  np.size(traj_0, 1)):
            for col_1 in range(0,  np.size(traj_1, 1)):
                distance += np.sqrt(sum((traj_0[:,col_0] - traj_1[:,col_1])**2))
    else:
        pass
    
    return distance

# afterwards implement this for all elements of a traj list.
def campolongo_2007(trajectory_list):
    distance_matrix = np.nan * np.ones(shape=(len(trajectory_list),len(trajectory_list)))
    for i in range(0, len(trajectory_list)):
        for j in range(0, len(trajectory_list)):
            distance_matrix[i,j] = compute_trajectory_distance(
                    trajectory_list[i],
                    trajectory_list[j])
            return distance_matrix

traj_0 = np.ones((3,2))
traj_1 = np.zeros((3,2))

compute_trajectory_distance(traj_0, traj_1)
campolongo_2007([traj_0, traj_1])
    
    
    





