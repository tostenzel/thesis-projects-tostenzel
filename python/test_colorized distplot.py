"""Test"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#def cone_plot_choices():
tuition_subsidies = [0, 500]

mc_base_shares_occ_df = pd.read_pickle("results/mc_base_occ_shares_df.pkl")
mc_policy_occ_shares_df = pd.read_pickle("results/mc_policy_occ_shares_df.pkl")

# Set some plt and sns properties: Latex font and custom colors.
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"
current_palette = sns.color_palette("deep")
sns.set_palette(current_palette)

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
scenarios = [mc_base_shares_occ_df, mc_policy_occ_shares_df]

for scenario, idx in zip(scenarios, range(0, 2)):
    with sns.axes_style("whitegrid"):
        sns.set_palette("deep")

        scenarios[idx]['edu'].T.quantile(0.01).plot(
            ax=axs[idx], legend=True, linewidth=1.0, color='white'
        )
        scenarios[idx]['edu'].T.quantile(0.99).plot(
            ax=axs[idx], legend=True, linewidth=1.0, color='white'
        )
        scenarios[idx]['edu'].T.mean().plot(
            ax=axs[idx], legend=True, linewidth=1.0, color=current_palette[3]
        )
        axs[idx]['edu'].fill_between(scenarios[idx].index,
                scenarios[idx].T.quantile(0.01),scenarios[idx].T.quantile(0.99),
                facecolor=current_palette[3], alpha=0.2)

        axs[idx].set_ylim(-0.05, 0.65)
        axs[idx].set_xticks([16, 20, 25, 30, 35, 40, 45, 50, 55])
        axs[idx].set_xticklabels(
            [16, 20, 25, 30, 35, 40, 45, 50, 55], rotation="horizontal"
        )
        axs[idx].set_ylabel(
            "Share of population", labelpad=10, fontdict=dict(size=16)
        )
        axs[idx].set_xlabel("Age", labelpad=10, fontdict=dict(size=16))

        axs[idx].tick_params(direction="out", length=4, width=1.1, labelsize=14)
        axs[idx].tick_params(axis="x", which="major", pad=6)

        handles, labels = axs[idx].get_legend_handles_labels()
        axs[idx].get_legend().remove()
        if idx == 1:
            label = "with a tuition subsidy of {} USD".format(
                tuition_subsidies[idx]
            )
        else:
            label = "without a tuition subsidy"
        axs[idx].set_title(f"Occupational choices \n {label}", size=16)

    legend = fig.legend(
        handles,
        ["Education"],
        loc="lower center",
        bbox_to_anchor=(0.460, -0.010),
        ncol=4,
        frameon=True,
        framealpha=1.0,
        fontsize=14,
        edgecolor="black",
        fancybox=False,
        borderpad=0.5,
    )
    frame = legend.get_frame()
    frame.set_linewidth(0.5)

fig.subplots_adjust(bottom=0.24)

if save is True:
    # Define the script path relative to the jupyter notebook that calls the script.
    abs_dir = os.path.dirname(__file__)
    plt.savefig(
        os.path.join(abs_dir, "figures/cone_plot_choices.png"), bbox_inches="tight"
    )
else:
    pass

