import os

import matplotlib.pyplot as plt
import respy as rp
import seaborn as sns

# Set some plt and sns properties: Latex font and custom colors.
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"
current_palette = sns.color_palette("deep")
sns.set_palette(current_palette)


def figure_choice_shares_over_time(save=False):
    """
    Create figures of shares of occupation choices over time for a
    sample of 1000 agents in KW94.
    Parameters
    ----------
    Returns
    -------
    """
    # Build simulate function. As only parameters change, it can be reused.
    params, options, _ = rp.get_example_model("kw_94_one")
    options["simulation_agents"] = 4_000
    simulate = rp.get_simulate_func(params, options)

    # One policy and one base policy implies two models.
    tuition_subsidies = [0, 500]

    # Generate data based on a simulation of 1000 agents.
    shares_dfs_list = []
    for tuition_subsidy in tuition_subsidies:
        params.loc[
            ("nonpec_edu", "at_least_twelve_exp_edu"), "value"
        ] += tuition_subsidy
        df = simulate(params)
        df["Age"] = df["Period"] + 16
        shares = (
            df.groupby("Age")
            .Choice.value_counts(normalize=True)
            .unstack()[["home", "edu", "a", "b"]]
        )
        # Set 0 NaNs in edu shares to 0.
        shares["edu"].fillna(0, inplace=True)
        shares_dfs_list.append(shares[["edu", "a", "b", "home"]])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for idx in range(0, len(tuition_subsidies)):
        with sns.axes_style("whitegrid"):
            sns.set_palette("deep")

            shares_dfs_list[idx]["edu"].plot(
                ax=axs[idx], legend=True, linewidth=4.0, color=current_palette[3]
            )
            shares_dfs_list[idx]["a"].plot(
                ax=axs[idx], linestyle="-.", linewidth=2, color=current_palette[0]
            )
            shares_dfs_list[idx]["b"].plot(
                ax=axs[idx], linestyle="-.", linewidth=2, color=current_palette[1]
            )
            shares_dfs_list[idx]["home"].plot(
                ax=axs[idx], linestyle="-.", linewidth=2, color=current_palette[2]
            )

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
        # Define the script path relative to the jupyter notebook that calls the script.
        abs_dir = os.path.dirname(__file__)
        plt.savefig(
            os.path.join(abs_dir, "figures/occ_choice_shares.png"), bbox_inches="tight"
        )
    else:
        pass

    return fig, axs
