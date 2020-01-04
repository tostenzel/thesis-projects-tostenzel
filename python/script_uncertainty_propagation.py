import os

# Use multiprocessing for parall computing. Needs to be set up at the beginning.
update = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
os.environ.update(update)

import chaospy as cp
import pandas as pd
import numpy as np
import respy as rp

from multiprocessing import Pool

from quantity_of_interest import get_quantity_of_interest
from quantity_of_interest import model_wrapper_kw_94

# Define the script path relative to the jupyter notebook that calls the script.
# abs_dir = os.path.dirname(__file__)


def propagate_mean_estimates(save=False):
    """Evaluates the QoI at the mean estimates"""
    # Init base_options because its part of the model wrapper argument
    _, base_options = rp.get_example_model("kw_94_one", with_data=False)

    # Read correctly indexed estimation results in respy format.
    base_params = pd.read_pickle("input/rp_params_chol.uq.pkl")

    policy_edu, _ = model_wrapper_kw_94(base_params, base_options, 500)
    base_edu, _ = model_wrapper_kw_94(base_params, base_options, 0)
    base_quantity = policy_edu - base_edu

    base_quantity = pd.DataFrame(base_quantity, columns=["avg_schooling"], index=[0])
    if save is True:
        base_quantity.to_pickle("results/base_quantity.uq.pkl")
    else:
        pass

    return base_quantity


def run():
    """
    Function that protects the "entry point" of new programs to not produce endlessly
    many entries. See:
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """
    # Call function.
    propagate_mean_estimates(save=True)

    # Global variables.
    seed = 123
    number_draws = 10

    # Init estimates of parameters and their covariance matrix as nummpy arrays.
    params = pd.read_pickle("input/params_chol.uq.pkl")
    mean = params["value"].to_numpy()
    cov = pd.read_pickle("input/cov_chol.uq.pkl").to_numpy()

    # Draw the sample of random input parameters.
    np.random.seed(seed)
    distribution = cp.MvNormal(loc=mean, scale=cov)

    sample_input_parameters = list()
    for _ in range(number_draws):
        sample_input_parameters.append(distribution.sample())
    # Check for errors.
    temp_array = np.array(sample_input_parameters)
    assert np.isinf(temp_array.any()) == 0
    assert np.isnan(temp_array.any()) == 0

    # Parallelized: The default number of worker processes is the number of CPUs.
    # Evaluate the QoI at the randomly drawn input paramter vectors.
    pool = Pool(8)
    mc_quantities = pool.map(get_quantity_of_interest, sample_input_parameters)
    # Close worker processes.
    pool.close()
    # Wait until these are terminated.
    pool.join()

    # Check for errors.
    temp_array = np.array(mc_quantities)
    assert np.isinf(temp_array.any()) == 0
    assert np.isnan(temp_array.any()) == 0

    # Save the random parameters and the quantity of interest.
    index = pd.read_pickle("input/params_chol.uq.pkl").index
    mc_params_ser = pd.Series(np.column_stack(sample_input_parameters), index=index)

    mc_quantities_df = pd.DataFrame(
        mc_quantities, columns=["avg_schooling"], index=range(number_draws)
    )
    mc_quantities_df.index.name = "iteration"

    mc_quantities_df.to_pickle("results/mc_quantity.uq.pkl")
    mc_params_df.to_pickle("results/mc_params.uq.pkl")


# Avoid mp Runtime error.
if __name__ == "__main__":
    run()
