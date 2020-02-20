import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Set some plt and sns properties: Latex font and custom colors.
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"


def heatmap_corr_chol(corr_df, save=False):
    """
    Creates the heatmap for the correlations between important
    input parameters of the KW94 model.

    Parameters
    ----------
    corr_df : Dataframe
        Correlation matrix.
    save : bool
        Indicates if the graph is saved as png-file.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

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
        # Define the script path relative to the jupyter notebook that calls the script.
        abs_dir = os.path.dirname(__file__)
        plt.savefig(os.path.join(abs_dir, "../figures/heatmap.png"), bbox_inches="tight")
    else:
        pass

    return fig, ax


def distplot(sample, qoi_name, save=False):
    """
    This function is a custom-made wrapper around seaborn.distplot for the QoI.

    Parameters
    ----------
    sample: Series, 1d-array, or list.
        A vector of random observations in vertical format.
    qoi_name : str
        Name of Quantity of interest used for x label and png-file.
    save : bool
        Indicate whether to save the plot as png.

    Returns
    -------
    dp : matplotlib Axes
    ax : matplotlib Axes

    """
    # Init colors.
    cmap = cm.get_cmap("RdYlBu_r")
    colour_mid = cmap(0.1)
    colour_outer = cmap(0.2)
    colour_out = cmap(0.999)
    # Colours for different percentiles.
    perc_2dot5_colour = colour_out
    perc_16_colour = colour_outer
    perc_50_colour = colour_mid
    perc_84_colour = colour_outer
    perc_97dot5_colour = colour_out

    # Plot the Histogram from the random data.
    fig, ax = plt.subplots(figsize=(10, 8))

    counts, bins, patches = ax.hist(
        sample, bins=100, facecolor=perc_50_colour, density=True
    )

    # Change the colors of bars at the edges.
    twodotfive, sixteen, eightyfour, ninetyseventdotfive = np.percentile(
        sample, [2.5, 16, 84, 97.5]
    )
    for patch, leftside, rightside in zip(patches, bins[:-1], bins[1:]):
        if rightside < twodotfive:
            patch.set_facecolor(perc_2dot5_colour)
        elif leftside > ninetyseventdotfive:
            patch.set_facecolor(perc_97dot5_colour)
        elif leftside > eightyfour:
            patch.set_facecolor(perc_84_colour)
        elif rightside < sixteen:
            patch.set_facecolor(perc_16_colour)

    # Plot mean as vertical line.
    mean = ax.axvline(np.mean(sample), color="gainsboro", linestyle="--", lw=4)

    # Call seaborn.distplot and set options.
    dp = sns.distplot(
        sample, hist=False, kde=True, kde_kws=dict(color=cmap(0.0), linewidth=5)
    )

    kde_handle = Line2D([0], [0], color=cmap(0.0), lw=5)
    handles = [
        Line2D(
            [],
            [],
            color=c,
            marker="|",
            linestyle="None",
            markersize=22,
            markeredgewidth=3.5,
        )
        for c in [perc_2dot5_colour, perc_16_colour, perc_50_colour]
    ]
    handles.append(mean)
    handles.append(kde_handle)
    labels = [
        "KDE",
        r"Sample mean $\overline{Y}$",
        r"$Y_n \in [\overline{Y} \pm \sigma_Y]$",
        r"$Y_n \in [\overline{Y} \pm 2\sigma_Y]$",
        r"$Y_n ~ \notin [\overline{Y} \mp 2\sigma_Y]$",
    ]
    # Reverse list order.
    ax.legend(handles[::-1], labels, edgecolor="white", fontsize=20)

    ax.grid(True, linestyle="-", alpha=0.5)
    ax.tick_params(axis="both", direction="out", length=6, width=2)
    # A bit more space for xlabels.
    ax.tick_params(axis="x", which="major", pad=8)
    ax.set_xlabel("{}".format(qoi_name), labelpad=20)
    ax.set_ylabel("Probability density", labelpad=15)

    if save is True:
        # Define the script path relative to the jupyter notebook that calls the script.
        abs_dir = os.path.dirname(__file__)
        plt.savefig(os.path.join(abs_dir, "../figures/distplot.png"), bbox_inches="tight")
    else:
        pass

    return dp, ax


def convergence_plot(sample, expected, qoi_name, absolute_deviation=False, save=False):
    """
    This function is a custom-made convergence plot for some Monte-Carlo
    sample.

    Parameters
    ----------
    sample : Series, 1d-array, or list.
        A vector of random observations.
    expected : float, int.
        Expected value of sample mean.
    qoi_name : str
        Label of y-axis.
    absolute_deviation : bool
        Plots absolute deviation of means to zero expectation value.
    save : bool
        Indicates whether to save the plot as png.

    Returns
    -------
    dp : pyplot.figure
    ax : pyplot.axes
    """
    df = pd.DataFrame(list(sample), columns=["qoi_realization"])
    df["cum_sum"] = df["qoi_realization"].cumsum()
    df["mean_iteration"] = df["cum_sum"] / (df.index.to_series() + 1)

    fig, ax = plt.subplots()

    if absolute_deviation is not True:
        # Compute sample mean for each iteration
        title = "Convergence of Monte-Carlo Uncertainty Propagation (level)"
        file_str = "level"
        legend_loc = "lower right"

        (conv_plot,) = ax.plot(
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

        (conv_plot,) = ax.plot(
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
        # Define the script path relative to the jupyter notebook that calls the script.
        abs_dir = os.path.dirname(__file__)
        plt.savefig(
            os.path.join(abs_dir, "../figures/convergence_plot_{}.png").format(file_str),
            bbox_inches="tight",
        )
    else:
        pass

    return plt, ax
