"""Morris Screening"""
import numpy as np


n_inputs = 2
n_levels = 4
stepsize = n_levels / (2 * (n_levels - 1))


def morris_trajectories(
    init_input_pars, n_inputs, stepsize, seed=123, test_D_star_rand_2dim=False
):
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
    ee_i = elementary_effect_i(model, i_python, init_input_pars, stepsize)

    return ee_i * (sd_i / sd_model)
