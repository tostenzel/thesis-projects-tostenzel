"""Cone Plot"""
import os

# Define the script path relative to the jupyter notebook that calls the script.
abs_dir = os.path.dirname(__file__)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

# Set some plt and sns properties: Latex font and custom colors.
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"
current_palette = sns.color_palette("deep")
sns.set_palette(current_palette)


def cone_plot_choice_shares(save=True):
    """
    Uses sample of randomly drawn occupation choices shares over time
    of a sample of agents from the input paramter distribution to plot
    confidence intervals for these shares around their means.

    Parameters
    ----------
    save : bool
        Indicates if the graph is saved as png-file.

    """
    tuition_subsidies = [0, 500]

    mc_base_shares_occ_df = pd.read_pickle(
        os.path.join(abs_dir, "results/10_000_draws/mc_base_occ_shares_df.pkl")
    )
    mc_policy_occ_shares_df = pd.read_pickle(
        os.path.join(abs_dir, "results/10_000_draws/mc_policy_occ_shares_df.pkl")
    )

    occupations = ["edu", "a", "b", "home"]

    means_base = mc_base_shares_occ_df.groupby("Age").mean()
    percentile_99_base = mc_base_shares_occ_df.groupby("Age").quantile(0.99)
    percentile_1_base = mc_base_shares_occ_df.groupby("Age").quantile(0.01)

    means_policy = mc_policy_occ_shares_df.groupby("Age").mean()
    percentile_99_policy = mc_policy_occ_shares_df.groupby("Age").quantile(0.99)
    percentile_1_policy = mc_policy_occ_shares_df.groupby("Age").quantile(0.01)

    means = [means_base, means_policy]
    percentile_99 = [percentile_99_base, percentile_99_policy]
    percentile_1 = [percentile_1_base, percentile_1_policy]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for idx in range(0, len(tuition_subsidies)):
        with sns.axes_style("whitegrid"):
            sns.set_palette("deep")
            sort_colour = [3, 0, 1, 2]
            colour_idx = 0
            for occ in occupations:
                percentile_99[idx][occ].plot(
                    ax=axs[idx], legend=True, linewidth=1.0, color="white"
                )
                percentile_1[idx][occ].plot(
                    ax=axs[idx], legend=True, linewidth=1.0, color="white"
                )
                axs[idx].fill_between(
                    means[idx].index,
                    percentile_99[idx][occ],
                    percentile_1[idx][occ],
                    facecolor=current_palette[sort_colour[colour_idx]],
                    alpha=0.2,
                )
                if colour_idx == 0:
                    means[idx][occ].plot(
                        ax=axs[idx],
                        legend=True,
                        linewidth=1.0,
                        linestyle="-",
                        color=current_palette[sort_colour[colour_idx]],
                    )
                else:
                    means[idx][occ].plot(
                        ax=axs[idx],
                        legend=True,
                        linewidth=1.0,
                        linestyle="-.",
                        color=current_palette[sort_colour[colour_idx]],
                    )
                colour_idx += 1

            axs[idx].set_ylim(0, 0.85)
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

            handles = [
                Line2D([0], [0], color=current_palette[c], lw=1, linestyle="-.")
                for c in sort_colour[1:4]
            ][::-1]
            handles.append(
                Line2D([0], [0], color=current_palette[sort_colour[0]], lw=1)
            )
            axs[idx].get_legend().remove()
            if idx == 1:
                label = "with a tuition subsidy of {} USD".format(
                    tuition_subsidies[idx]
                )
            else:
                label = "without a tuition subsidy"
            axs[idx].set_title(f"Occupational choices \n {label}", size=16)

        legend = fig.legend(
            handles[::-1],
            ["Education", "Blue-collar", "White-collar", "Home"],
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
        plt.savefig(
            os.path.join(abs_dir, "figures/cone_plot_choice_shares.png"),
            bbox_inches="tight",
        )
    else:
        pass

    return fig, axs
