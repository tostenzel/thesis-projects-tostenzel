"""Screening functions not directly used for sampling."""
import numpy as np

a = [78, 12, 0.5, 2, 97, 33]


def sobol_model(input_pars, coeffs_a):
    """
    Tested by comparing graphs for 3 specifications to book.
    Arguments are lists. Strongly nonlinear, nonmonotonic, and nonzero interactions.
    Analytic results for Sobol Indices.

    """
    assert len(input_pars) == len(coeffs_a)

    def g_i(input_par_i, coeffs_a_i):
        return (abs(4 * input_par_i - 2) + coeffs_a_i) / (1 + coeffs_a_i)

    y = 1
    for i in range(0, len(input_pars)):
        y *= g_i(input_pars[i], coeffs_a[i])

    return y


def elementary_effect_i(model, i_python, init_input_pars, stepsize):
    """EE"""
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
