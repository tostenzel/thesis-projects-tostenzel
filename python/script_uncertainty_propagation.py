import os

# In this script we only have explicit use of MULTIPROCESSING as our level of parallelism. This
# needs to be done right at the beginning of the script.
update = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
os.environ.update(update)
number_processes = 2

import chaospy as cp
import pandas as pd
import numpy as np
import respy as rp

from multiprocessing import Pool

from quantity_of_interest import get_quantity_of_interest
from quantity_of_interest import model_wrapper_kw_94

# Define the script path relative to the jupyter notebook that calls the script.
#abs_dir = os.path.dirname(__file__)

def propagate_mean_estimates(save=False):
    # Init base_options because its part of the model wrapper argument
    _, base_options = rp.get_example_model("kw_94_one", with_data=False)
    
    
    # Read correctly indexed estimation results in respy format.
    base_params = pd.read_pickle("input/rp_params_chol.uq.pkl")
    
    policy_edu, _ = model_wrapper_kw_94(base_params, base_options, 500)
    base_edu, _ = model_wrapper_kw_94(base_params, base_options, 0)
    base_quantity = policy_edu - base_edu
    
    base_quantity = pd.DataFrame(base_quantity, columns=['avg_schooling'], index=[0])
    if save is True:
        base_quantity.to_pickle("results/base_quantity.uq.pkl")
    else:
        pass
    
    return base_quantity

propagate_mean_estimates(save=True)

seed = 123
number_draws = 100   
# Init estimates of parameters and their covariance matrix as nummpy arrays.
params = pd.read_pickle("input/params_chol.uq.pkl")
mean = params['value'].to_numpy()
cov = pd.read_pickle("input/cov_chol.uq.pkl").to_numpy()

# Draw the sample of random input parameters.
np.random.seed(seed)
distribution = cp.MvNormal(loc=mean, scale=cov)

sample_input_parameters = list()
for _ in range(number_draws):
    sample_input_parameters.append(distribution.sample())

quantities = Pool(number_processes).map(get_quantity_of_interest, sample_input_parameters)

# We now store the random parameters and the quantity of interest for further processing.
index = pd.read_pickle("input/params_chol.uq.pkl").index

mc_params = pd.DataFrame(np.column_stack(samples), index=index)

mc_quantities = pd.DataFrame(quantities, columns=['avg_schooling'], index=range(args.num_draws))
mc_quantities.index.name = 'iteration'

assert np.any(np.isinf(mc_params.to_numpy())) == False
assert mc_params.isnull().values.any() == False

assert np.any(np.isinf(mc_quantities.to_numpy())) == False
assert mc_quantities.isnull().values.any() == False

mc_quantities.to_pickle(RSLT_DIR / "mc_quantity.uq.pkl")
mc_params.to_pickle(RSLT_DIR / "mc_params.uq.pkl")