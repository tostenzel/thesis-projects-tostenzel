"""
Model "mathcal{M}(pmb{theta}).
This represents the Quantity of Interest q as a function of the
vector of input paramters theta.
"""

import os

import numpy as np
import respy as rp
import pandas as pd

def get_quantity_of_interest(input_params):
    """
    Computes the Quantity of Interest.

    Paramters
    ---------
    input_params: np.array
        unindexed input parameters
    Returns
    -------
    qoi: float
        Quantity of Interest
    """
    # We need the baseline options and a grid for the indices. It does not matter which of the
    # three KW94 specifications we use here.
    _, base_options = rp.get_example_model("kw_94_one", with_data=False)
    # Define the script path relative to the jupyter notebook that calls the script.
    abs_dir = os.path.dirname(__file__)
    # Estimated parameters with Choleskies plus 3 fixed respy parameters.
    base_params = pd.read_pickle(os.path.join(abs_dir, "input/rp_params_chol.uq.pkl"))

    params_idx = pd.Series(data=input_params, index=base_params.index[0:27])
    params_idx_respy = transform_params_kw94_respy(params_idx)

    policy_edu, _ = model_wrapper_kw_94(params_idx_respy, base_options, 500.0)
    base_edu, _ = model_wrapper_kw_94(params_idx_respy, base_options, 0.0)

    return policy_edu - base_edu


def model_wrapper_kw_94(input_params, base_options, tuition_subsidy):
    """
    Wrapper around respy to compute the mean number of years in education.

    Parameters
    ----------
    input_params: Dataframe
        Contains the complete respy vector of input parameters with indices.
    base_options: dict
        Contains the options for the sumulation of agents.
    tuition_subsidy: float
        tuition subsidy that is added to the respective paramter.

    Returns
    -------
    edu: float
        mean number of years in education.
    policy_df: Dataframe
        Dataframe of occupation choices of sample of agents of time.
    """
    simulate = rp.get_simulate_func(input_params, base_options)

    policy_params = input_params.copy()
    policy_params.loc[("nonpec_edu", "at_least_twelve_exp_edu"), "value"] += tuition_subsidy
    policy_df = simulate(policy_params)

    edu = policy_df.groupby("Identifier")["Experience_Edu"].max().mean()

    return edu, policy_df


def transform_params_kw94_respy(params_idx):
    """Converts indexed Series of non-constant input paramters to a Dataframe
    and adds three constant factors to achieve respy format.

    Parameters
    ----------
    params_idx: Series
        Non-constant input paramters with correct indices

    Returns
    --------
    rp_params_df: Dataframe
        Input parameters in respy format.
    """ 
    assert len(params_idx) == 27, "Length of KW94 vector must be 27."
    part_1 = params_idx

    rp_params, _ = rp.get_example_model("kw_94_one", with_data=False)
    part_2 = rp_params.iloc[27:31,0]

    parts = [part_1, part_2]
    rp_params_series = pd.concat(parts)
    rp_params_df = pd.DataFrame(rp_params_series, columns=["value"])

    return rp_params_df
