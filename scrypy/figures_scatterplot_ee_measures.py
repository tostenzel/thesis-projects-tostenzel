"""Scatterplot results."""

import pickle


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load Measures.  
with open('results/measures_to_plot.pkl', 'rb') as f:
  measures_to_plot = pickle.load(f)

"""
# Load standard deviations of input paramters.
with open('input/est_rand_params_chol.pkl', 'rb') as f:
  params = pickle.load(f)
 
params.index
"""

  
param_labels = np.array(
    [
        r"$\hat{\delta}$",
        r"$\hat{\beta^{b}}$",
        r"$\hat{\beta_{e}^{b}}$",
        r"$\hat{\beta_{b}^{b}}$",
        r"$\hat{\beta_{bb}^{b}}$",
        r"$\hat{\beta_{w}^{b}}$",
        r"$\hat{\beta_{ww}^{b}}$",
        r"$\hat{\beta^{w}}$",
        r"$\hat{\beta_{e}^{w}}$",
        r"$\hat{\beta_{w}^{w}}$",
        r"$\hat{\beta_{ww}^{w}}$",
        r"$\hat{\beta_{b}^{w}}$",
        r"$\hat{\beta_{bb}^{w}}$",
        r"$\hat{\beta^{e}}$",
        r"$\hat{\beta_{col}^{e}}$",
        r"$\hat{\beta_{re}^{e}}$",
        r"$\hat{\beta^{h}}$",
        r"$\hat{c_{1}}$",
        r"$\hat{c_{2}}$",
        r"$\hat{c_{3}}$",
        r"$\hat{c_{4}}$",
        r"$\hat{c_{1,2}}$",
        r"$\hat{c_{1,3}}$",
        r"$\hat{c_{2,3}}$",
        r"$\hat{c_{1,4}}$",
        r"$\hat{c_{2,4}}$",
        r"$\hat{c_{3,4}}$",
    ]
)

param_groups = np.array(
    [
        r"Discount factor",
        r"Blue-collar",
        r"Blue-collar",
        r"Blue-collar",
        r"Blue-collar",
        r"Blue-collar",
        r"Blue-collar",
        r"White-collar",
        r"White-collar",
        r"White-collar",
        r"White-collar",
        r"White-collar",
        r"White-collar",
        r"Education",
        r"Education",
        r"Education",
        r"Home",
        r"Choleskies",
        r"Choleskies",
        r"Choleskies",
        r"Choleskies",
        r"Choleskies",
        r"Choleskies",
        r"Choleskies",
        r"Choleskies",
        r"Choleskies",
        r"Choleskies",
    ]
)

#index = pd.MultiIndex.from_arrays(param_groups, param_labels)

abs_ee_mean = pd.DataFrame(np.hstack(measures_to_plot), index=[param_groups, param_labels],
                    columns= [r"$\mu_{traj}^{*,u}$", r"$\mu_{traj}^{*,c}$",
                           r"$\mu_{rad}^{*,u}$", r"$\mu_{rad}^{*,c}$"])


plt.style.use('_mplstyle/uq.mplstyle') # wanna include that in funciton.

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"
current_palette = sns.color_palette("deep")
sns.set_palette(current_palette)

 
  
  
sns.scatterplot(

    x=abs_ee_mean.iloc[:,1], y=abs_ee_mean.iloc[:,0], hue=abs_ee_mean.index.get_level_values(0)

)