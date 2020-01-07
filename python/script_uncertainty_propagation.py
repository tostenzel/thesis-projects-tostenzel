import os

# Define the script path relative to the location from where the script is called.
abs_dir = os.path.dirname(__file__)

# Use multiprocessing for parallel computing. Needs to be set up at the beginning.
# Restrict number of threads to one for each library.
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
import argparse

from multi_quantities_of_interest import model_wrapper_kw_94
from multi_quantities_of_interest import get_quantity_of_interest


def propagate_mean_estimates():
    """Evaluates the QoI at the mean estimates"""
    # Init base_options because its part of the model wrapper argument
    _, base_options = rp.get_example_model("kw_94_one", with_data=False)

    # Read correctly indexed estimation results in respy format.
    mean_params = pd.read_pickle(os.path.join(abs_dir, "input/est_rp_params_chol.pkl"))

    policy_edu, base_occ_shares_df, _ = model_wrapper_kw_94(
        mean_params, base_options, 500
    )
    base_edu, policy_occ_shares_df, _ = model_wrapper_kw_94(
        mean_params, base_options, 0
    )
    qoi_mean_params_edu = policy_edu - base_edu

    mean_edu_df = pd.DataFrame(
        qoi_mean_params_edu, columns=["change_mean_schooling"], index=[0]
    )
    # Store results.
    mean_edu_df.to_pickle(
        os.path.join(abs_dir, "results/qoi_mean_params_change_mean_edu_df.pkl")
    )
    base_occ_shares_df.to_pickle(
        os.path.join(abs_dir, "results/qoi_mean_params_base_occ_shares_df.pkl")
    )
    policy_occ_shares_df.to_pickle(
        os.path.join(abs_dir, "results/qoi_mean_params_policy_occ_shares_df.pkl")
    )

    return mean_edu_df, base_occ_shares_df, policy_occ_shares_df


def run(args):
    """
    Function that protects the "entry point" of new programs to not produce endlessly
    many entries. See:
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """
    # Call function.
    _, _, _ = propagate_mean_estimates(save=True)

    # Global variables.

    # Init estimates of parameters and their covariance matrix as nummpy arrays.
    params = pd.read_pickle("input/est_rand_params_chol.pkl")
    mean = params["value"].to_numpy()
    cov = pd.read_pickle("input/est_cov_chol.pkl").to_numpy()

    # Draw the sample of random input parameters.
    np.random.seed(args.seed)
    distribution = cp.MvNormal(loc=mean, scale=cov)

    mc_sample_input_parameters = list()
    for _ in range(args.number_draws):
        mc_sample_input_parameters.append(distribution.sample())
    # Check for errors.
    temp_array = np.array(mc_sample_input_parameters)
    assert np.isinf(temp_array.any()) == 0
    assert np.isnan(temp_array.any()) == 0

    # Parallelized: The default number of worker processes is the number of CPUs.
    # Evaluate the QoI at the randomly drawn input paramter vectors.
    pool = Pool(8)
    mc_change_mean_edu = list()
    mc_policy_occ_shares = list()
    mc_base_occ_shares = list()

    # Pool returns lists. Need Loop to handle these lists.
    for i, j, k in pool.map(get_quantity_of_interest, mc_sample_input_parameters):
        mc_change_mean_edu.append(i)
        mc_policy_occ_shares.append(j)
        mc_base_occ_shares.append(k)
    # Close worker processes.
    pool.close()
    # Wait until these are terminated.
    pool.join()

    # Check for errors in main qoi.
    temp_array = np.array(mc_change_mean_edu)
    assert np.isinf(temp_array.any()) == 0
    assert np.isnan(temp_array.any()) == 0

    # Store the random parameters and the quantity of interest.
    # Paramter x iteration
    tmp_idx = pd.read_pickle("input/est_rand_params_chol.pkl").index
    mc_input_parameters_df = pd.DataFrame(
        np.column_stack(mc_sample_input_parameters), index=tmp_idx
    )
    # The shares for each iteration are stacked along the vertical axis.
    # Therefore, indices 16-65 are not unique. Dim.: (65-15)*n_iter x 4.
    mc_base_occ_shares_df = pd.concat(mc_base_occ_shares)
    mc_policy_occ_shares_df = pd.concat(mc_policy_occ_shares)

    # Dim.: 1 x Iteration
    mc_change_mean_edu_df = pd.DataFrame(
        mc_change_mean_edu,
        columns=["change_mean_schooling"],
        index=range(args.number_draws),
    ).T

    mc_input_parameters_df.to_pickle(
        os.path.join(abs_dir, "results/mc_input_parameters_df.pkl")
    )
    mc_base_occ_shares_df.to_pickle(
        os.path.join(abs_dir, "results/mc_base_occ_shares_df.pkl")
    )
    mc_policy_occ_shares_df.to_pickle(
        os.path.join(abs_dir, "results/mc_policy_occ_shares_df.pkl")
    )
    mc_change_mean_edu_df.to_pickle(
        os.path.join(abs_dir, "results/mc_change_mean_edu_df.pkl")
    )


# Avoid mp Runtime error.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create results for Uncertainty Propagation."
    )

    parser.add_argument(
        "-d",
        "--number_draws",
        action="store",
        dest="number_draws",
        default=5,
        type=int,
        help="set number of random input parameter draws",
    )

    parser.add_argument(
        "-s",
        "--seed",
        action="store",
        dest="seed",
        default=123,
        type=int,
        help="set seed for the random input parameter draws",
    )

    args = parser.parse_args()

    run(args)
