"""Plot the quantile /inverse CDF function"""


import numpy as np

from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

grid = np.linspace(0,1, 1000).tolist()
num_zero = 0.00001
grid[0] = num_zero / 1000000
grid[999] = 1- num_zero/1000000



x = [num_zero, 1/3, 2/3, 1-num_zero]
y = norm.ppf(x)

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "STIXGeneral"
current_palette = sns.color_palette("deep")
sns.set_palette(current_palette)
plt.style.use('_mplstyle/uq.mplstyle')
plt.rcParams['axes.linewidth'] = 2

fig, ax = plt.subplots(figsize=(12, 9))

ax = sns.lineplot(grid, norm.ppf(grid), markevery=x, label="Quantile function")
plt.plot([x], [y], marker='o', markersize=15, color=current_palette[0], markeredgecolor=current_palette[3], markerfacecolor=current_palette[3], label="Grid points GM'17")

ax.grid(True, linestyle="-", alpha=0.5)


handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[1]]
labels = [labels[0], labels[1]]




ax.legend(handles,labels,
          frameon=True,
          framealpha=1.0,
          fontsize=22,
          edgecolor="black",
          fancybox=False,
          borderpad=0.5,
          handletextpad=+0.5,
          markerscale=1.0,
          loc="lower right")




ax.tick_params(axis="x", which="major", pad=12)
ax.tick_params(axis="y", which="major", pad=8)
ax.tick_params(axis="both", direction="out", length=9, width=3, labelsize=22)

ax.set_ylabel(r"Standard normal sample space", labelpad=+25, rotation=90, fontsize=28)
ax.set_xlabel(r"Uniform sample space", labelpad=+25, fontsize=28)



plt.savefig("../figures/quantile_fct.png", bbox_inches="tight")