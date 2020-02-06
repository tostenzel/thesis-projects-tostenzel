"""Screening functions not directly used for sampling."""
import numpy as np

a = [78, 12, 0.5, 2, 97, 33]


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


def lin_portfolio(q1, q2, c1=2, c2=1, *args):
    """Simple function with analytic EE solution to support testing."""
    return c1 * q1 + c2 * q2


def test_elemtary_effect_i():
    """Tests EE"""
    assert 2 == round(
        elementary_effect_i(lin_portfolio, 0, [0.5, 1], stepsize=2 / 3), 10
    )

    assert 1 == round(
        elementary_effect_i(lin_portfolio, 1, [0.5, 1], stepsize=2 / 3), 10
    )
