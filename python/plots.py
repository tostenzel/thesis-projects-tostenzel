import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def heatmap_corr_chol(corr_df, save=False):
    """
    Creates the heatmap for the correlations between important
    input parameters.

    """
    # Mask to select the important parameters.
    select = corr_df.iloc[[0, 1, 7, 13, 16, 19, 20], [0, 1, 7, 13, 16, 19, 20]]
    mask = np.zeros_like(select)
    mask[np.triu_indices_from(select, 1)] = True

    labels = np.array(
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
            r"$\hat{c_{3,4}$}",
        ]
    )

    fig = plt.figure(figsize=(15, 10))

    # Aspects for heigth, pad for whitespace
    ax = sns.heatmap(
        np.round(select, 2),
        mask=mask,
        cmap="RdBu_r",
        linewidths=0.0,
        square=False,
        vmin=-1,
        vmax=1,
        annot=True,
    )

    ax.tick_params(axis="both", direction="out", length=6, width=2)
    ax.set_yticklabels(labels[[0, 1, 7, 13, 16, 19, 20]], ha="left", rotation=0)
    ax.set_xticklabels(labels[[0, 1, 7, 13, 16, 19, 20]], rotation=0)
    ax.set_ylabel(r"$\hat{\theta}$", labelpad=+35, rotation=0)
    ax.set_xlabel(r"$\hat{\theta}$", labelpad=+25)

    cbar = ax.collections[0].colorbar

    # Positioning at -1 needs vmin.
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["-1.0", " -0.5", "0.0", "0.5", "1.0"])
    cbar.ax.tick_params(direction="out", length=6, width=2)

    # A bit more space for xlabels.
    ax.tick_params(axis="x", which="major", pad=8)

    # Left-align y labels.
    yax = ax.get_yaxis()
    pad = max(tick.label.get_window_extent().width for tick in yax.majorTicks) + 5
    yax.set_tick_params(pad=pad)

    if save is True:
        plt.savefig("figures/heatmap.png", bbox_inches="tight")
    else:
        pass

    return fig, ax


def distplot(sample, qoi_name, save=False):
    """
    This function is a custom-made wrapper around seaborn.distplot.

    Parameters
    ----------
    sample: Series, 1d-array, or list.
        A vector of random observations.
    qoi_name: str
        Name of Quantity of interest used for x label and png-name.
    save: bool
        Indicate whether to save the plot as png.

    Returns
    -------
    dp: Figure
        Returns Figure object setting figure-level attributes.

    """
    fig, ax = plt.subplots()

    # Plot mean as vertical line.
    mean = ax.axvline(
        np.mean(sample), color="#1245A8", linestyle="--", lw=4, label="Sample Mean"
    )

    # Call seaborn.distplot and set options.
    dp = sns.distplot(
        sample,
        hist=True,
        kde=True,
        bins=100,
        norm_hist=True,
        hist_kws=dict(alpha=0.4, color="#1245A8", edgecolor="#1245A8"),
        kde_kws=dict(color="#1245A8", linewidth=5),
    )

    ax.set_title("Distribution of {}".format(qoi_name), y=1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Kernel Density Estimate", labelpad=30)
    ax.legend(handles=[mean], edgecolor="white")

    if save is True:
        plt.savefig("figures/distplot_{}.png".format(qoi_name), bbox_inches="tight")
    else:
        pass

    return dp, ax


def convergence_plot(sample, expected, qoi_name, absolute_deviation=False, save=False):
    """
    This function is a custom-made convergence plot for some Monte-Carlo
    sample.

    Parameters
    ----------
    sample: Series, 1d-array, or list.
        A vector of random observations.
    expected: float, int.
        Expected value of sample mean.
    qoi_name: str
        Label of y-axis.
    absolute_deviation: bool
        Plots absolute deviation of means to zero expectation value.
    save: bool
        Indicates whether to save the plot as png.

    Returns
    -------
    dp: Figure
        Returns Figure object setting figure-level attributes.

    """
    df = pd.DataFrame(sample, columns=["qoi_realization"])
    df["cum_sum"] = df["qoi_realization"].cumsum()
    df["mean_iteration"] = df["cum_sum"] / (df.index.to_series() + 1)

    fig, ax = plt.subplots()

    if absolute_deviation is not True:
        # Compute sample mean for each iteration
        title = "Convergence of Monte-Carlo Uncertainty Propagation (level)"
        file_str = "level"
        legend_loc = "lower right"

        conv_plot, = ax.plot(
            df.index + 1,
            df["mean_iteration"],
            color="#1245A8",
            lw=3.0,
            label="Sample Mean",
        )

    else:
        title = "Convergence of MC Uncertainty Propagation (absolute deviation)"
        file_str = "abs_dev"
        legend_loc = "upper right"

        conv_plot, = ax.plot(
            df.index + 1,
            abs(df["mean_iteration"] - expected),
            color="#1245A8",
            lw=3.0,
            label="Sample Mean",
        )
        expected = 0

    # Plot expected value.
    exp_plot = ax.hlines(
        expected,
        1,
        len(sample),
        lw=2.5,
        linestyle="--",
        label="Under Mean Parametrization",
    )

    ax.set_title(title, y=1.05)
    ax.set_xlim(1, len(sample))
    ax.grid(True, linestyle=(0, (5, 10)))
    ax.set_ylabel(qoi_name, labelpad=14)
    ax.set_xlabel("Number of iterations", labelpad=14)
    ax.tick_params(axis="both")
    ax.legend(
        handles=[exp_plot, conv_plot], loc=legend_loc, edgecolor="black", fancybox=False
    )

    if save is True:
        plt.savefig(
            "figures/convergence_plot_{}_{}.png".format(file_str, qoi_name),
            bbox_inches="tight",
        )
    else:
        pass

    return plt, ax
