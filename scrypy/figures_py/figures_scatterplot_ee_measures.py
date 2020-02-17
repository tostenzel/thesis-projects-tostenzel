"""Scatterplot results."""

import pickle


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from cycler import cycler
# Load Measures.  
with open('results/measures_to_plot.pkl', 'rb') as f:
  measures_to_plot = pickle.load(f)

"""
# Load standard deviations of input paramters.
with open('input/est_rand_params_chol.pkl', 'rb') as f:
  params = pickle.load(f)
 
params.index
"""

def scatter_plot(df, iloc_col_x, iloc_col_y, xlim, ylim):

    plt.style.use('_mplstyle/uq.mplstyle') # wanna include that in funciton.
    
    
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "STIXGeneral"
    current_palette = sns.color_palette("deep")
    sns.set_palette(current_palette)
    
    colorz = ["c", current_palette[0], current_palette[1],
        current_palette[3], current_palette[2], current_palette[5]]
    
    plt.rcParams["axes.prop_cycle"] = cycler('color', colorz)
    fig, ax = plt.subplots(figsize=(12, 9))
      
    ax = sns.scatterplot(
            df.iloc[:, iloc_col_x], df.iloc[:, iloc_col_y], s=250, hue=abs_ee_mean.index.get_level_values(0))

    
    # Reorder legend.
    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[3], handles[1], handles[2], handles[4], handles[5]]
    labels = [labels[0], labels[3], labels[1], labels[2], labels[4], labels[5]]
    ax.legend(handles,labels,
                              frameon=True,
                framealpha=1.0,
                fontsize=22,
                edgecolor="black",
                fancybox=False,
                borderpad=0.5,
                handletextpad=-0.06,
                markerscale=2.4)
    
    ax.grid(True, linestyle="-", alpha=0.5)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.tick_params(axis="x", which="major", pad=12)
    ax.tick_params(axis="y", which="major", pad=8)

    
    for line in range(0,abs_ee_mean.shape[0]):
         ax.text(df.iloc[line, iloc_col_x]+0.015, df.iloc[line, iloc_col_y]+0.00, 
         param_labels[line], horizontalalignment='left', 
         size=24, color='black', weight='semibold')   
    
    ax.tick_params(axis="both", direction="out", length=6, width=2, labelsize=22)
    ax.set_ylabel(r"$\mu^{*,u}_{\sigma}$", labelpad=+35, rotation=0, fontsize=28)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel(r"$\mu^{*,c}_{\sigma}$", labelpad=+25, fontsize=28)



param_labels = np.array(
    [
        r"$\delta$",
        r"$\beta^{b}$",
        r"$\beta_{e}^{b}$",
        r"$\beta_{b}^{b}$",
        r"$\beta_{bb}^{b}$",
        r"$\beta_{w}^{b}$",
        r"$\beta_{ww}^{b}$",
        r"$\beta^{w}$",
        r"$\beta_{e}^{w}$",
        r"$\beta_{w}^{w}$",
        r"$\beta_{ww}^{w}$",
        r"$\beta_{b}^{w}$",
        r"$\beta_{bb}^{w}$",
        r"$\beta^{e}$",
        r"$\beta_{col}^{e}$",
        r"$\beta_{re}^{e}$",
        r"$\beta^{h}$",
        r"$c_{1}$",
        r"$c_{2}$",
        r"$c_{3}$",
        r"$c_{4}$",
        r"$c_{1,2}$",
        r"$c_{1,3}$",
        r"$c_{2,3}$",
        r"$c_{1,4}$",
        r"$c_{2,4}$",
        r"$c_{3,4}$",
    ]
)

param_groups = np.array(
    [
        "Discount factor",
        "Blue-collar",
        "Blue-collar",
        "Blue-collar",
        "Blue-collar",
        "Blue-collar",
        "Blue-collar",
        "White-collar",
        "White-collar",
        "White-collar",
        "White-collar",
        "White-collar",
        "White-collar",
        "Education",
        "Education",
        "Education",
        "Home",
        "Choleskies",
        "Choleskies",
        "Choleskies",
        "Choleskies",
        "Choleskies",
        "Choleskies",
        "Choleskies",
        "Choleskies",
        "Choleskies",
        "Choleskies",
    ]
)

#index = pd.MultiIndex.from_arrays(param_groups, param_labels)

abs_ee_mean = pd.DataFrame(np.hstack(measures_to_plot), index=[param_groups, param_labels],
                    columns= ["traj_uncorr", "traj_corr",
                           "rad_uncorr", "rad_corr"])

scatter_plot(abs_ee_mean, 1, 0, 0.9, 6)

plt.savefig("figures/scatter_traj.png", bbox_inches="tight")

scatter_plot(abs_ee_mean, 3, 2, 0.9, 12)

plt.savefig("../figures/scatter_rad.png", bbox_inches="tight")
