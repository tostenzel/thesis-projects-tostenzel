"""Test"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

mc_change_avg_edu = pd.read_pickle("results/mc_change_avg_edu_df.pkl")


# define globals for dissolved fct.
sample = mc_change_avg_edu.values
qoi_name = r"$\Delta$ in mean years of education"
save = False



data = sample.T

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Generate random data
data = np.random.randn(10000)

# Colours for different percentiles
perc_2dot5_colour = 'red'
perc_16_colour = 'deepskyblue'
perc_50_colour = 'mediumaquamarine'
perc_84_colour = 'deepskyblue'
perc_97dot5_colour = 'red'

# Plot the Histogram from the random data
fig, ax = plt.subplots(figsize=(8,8))

counts, bins, patches = ax.hist(data, bins=100, facecolor=perc_50_colour, edgecolor='gray', density=True)

# Change the colors of bars at the edges
twodotfive, sixteen, eightyfour, ninetyseventdotfive = np.percentile(data, [2.5, 16, 84, 97.5])
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
mean = ax.axvline(
    np.mean(data), color="#1245A8", linestyle="--", lw=4, label="Sample mean"
)


#create legend      
handles =  [Rectangle((0,0),1,1,color=c,ec="k") for c in [perc_50_colour, perc_16_colour, perc_2dot5_colour]]
handles.append(mean)
labels = ["Sample mean", r"$\in [\gamma \mp \sigma$]", r"$\in [\gamma \mp 2\sigma]$", r"$\notin ~[\gamma \mp 2\sigma$]"]
ax.legend(handles[::-1], labels, edgecolor="white")
#bbox_to_anchor=(0.5, 0., 0.80, 0.99)



# Call seaborn.distplot and set options.
dp = sns.distplot(
    data,
    hist=False,
    kde=True,
    norm_hist=True,
    hist_kws=dict(alpha=0.4, color="#1245A8", edgecolor="#1245A8"),
    kde_kws=dict(color="#1245A8", linewidth=5)
)
    



ax.grid(True, linestyle='-', alpha=0.5)
ax.tick_params(axis="both", direction="out", length=6, width=2)
# A bit more space for xlabels.
ax.tick_params(axis="x", which="major", pad=8)
ax.set_xlabel("{}".format(qoi_name), labelpad=20)
#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)
ax.set_ylabel("Kernel density estimate", labelpad=15)

plt.show()
