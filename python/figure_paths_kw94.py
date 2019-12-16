"""Figure: Occupation choice paths KW94"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import respy as rp
import warnings
import seaborn as sns


from mpl_toolkits.mplot3d import Axes3D
#sns.set_style("white")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
current_palette = sns.color_palette("deep")
sns.set_palette(current_palette)

def choice_paths(save=False):
    # Build simulate function as only parameters change, it can be reused.
    params, options, _ = rp.get_example_model("kw_94_one")
    options["simulation_agents"] = 4_000
    simulate = rp.get_simulate_func(params, options)

    models = np.repeat(["one"], 2)
    tuition_subsidies = [0, 500]

    data_frames = []

    for model, tuition_subsidy in zip(models, tuition_subsidies):
        params, _, _ = rp.get_example_model(f"kw_94_{model}")
        params.loc[("nonpec_edu", "at_least_twelve_exp_edu"), "value"] += tuition_subsidy
        data_frames.append(simulate(params))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for df, ax, model, tuition_subsidy in zip(data_frames, axs, models, tuition_subsidies):
        df['Age'] = df['Period']+16
        data_frames.append(simulate(params))    
        shares = df.groupby("Age").Choice.value_counts(normalize=True).unstack()[["home", "edu", "a", "b"]]
        shares = shares[["edu", "a", "b", "home"]]
        with sns.axes_style('whitegrid'):
            sns.set_palette("deep")

            shares["edu"].plot(ax=ax, legend=True, linewidth=4.0, color=current_palette[3])
            shares["a"].plot(ax=ax, linestyle='-.', linewidth=2, color=current_palette[0])
            shares["b"].plot(ax=ax, linestyle='-.', linewidth=2,  color=current_palette[1])
            shares["home"].plot(ax=ax, linestyle='-.', linewidth=2, color=current_palette[2])

            
            
            ax.set_ylim(0, 0.85)
            ax.set_xticks([16, 20, 25, 30, 35, 40, 45, 50, 55])
            ax.set_xticklabels([16, 20, 25, 30, 35, 40, 45, 50, 55], rotation="horizontal")
            ax.set_ylabel('Share of population', labelpad=10, fontdict=dict(size=16))
            ax.set_xlabel("Age", labelpad=10, fontdict=dict(size=16))
            
            ax.tick_params(direction='out', length=4, width=1.1, labelsize=14)
            ax.tick_params(axis='x', which='major', pad=6)
            
            
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            if tuition_subsidy:
                label = f"with a tuition subsidy of {tuition_subsidy:,} USD"
            else:
                label = "without a tuition subsidy"
            ax.set_title(f"Occupational choices \n {label}", size=16)

        legend = fig.legend(
            handles,
            ["Education", "Blue Collar", "White Collar", "Home"],
            loc="lower center",
            bbox_to_anchor=(0.460, -0.010),
            ncol=4, frameon=True, framealpha=1.0, fontsize=14, edgecolor='black', fancybox=False, borderpad=0.5)
        frame = legend.get_frame()
        frame.set_linewidth(0.5)

    fig.subplots_adjust(bottom=0.24)

    if save is True:
        plt.savefig("figures/occ_paths.png", bbox_inches="tight")
    else:
        pass

    return fig, ax
