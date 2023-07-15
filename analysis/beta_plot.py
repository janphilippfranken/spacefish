from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_beta(a, b, ax, label=None, **kwargs):

    plt.rcParams['font.family'] = "Avenir"
    plt.rcParams['font.size'] = "50"

    x = np.linspace(0, 1, 1000)
    ax.plot(x, beta.pdf(x, a, b), label=label, **kwargs, color='grey')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, zorder=-100)
    sns.despine(left=True, bottom=False)

    colors = ['#0173B2', '#0173B2', '#0173B2', '#D9D9D9', '#E06666', '#E06666', '#E06666']
    # split the range on the x-axis in 7 equally sized parts and fill in between
    split_points = np.linspace(0, 1, 8)
    for i in range(1, 8):
        ax.fill_between(x, beta.pdf(x, a, b), where=(x < split_points[i]) & (x >= split_points[i-1]), alpha=0.8, color=colors[i-1], zorder=-200)

    return ax

fig, ax = plt.subplots(figsize=(8, 6))

plot_beta(2, 3, ax, label='1, 1')

xticks = np.linspace(0, 1, 8)[:-1] + 1/14  # shift ticks to be at the center of each bin
ax.set_xticks(xticks)
ax.set_xticklabels([3, 2, 1, 0, 1, 2, 3])
ax.set_yticks([])
ax.set_xlabel(r'$Beta(\alpha=2, \beta=3)$')
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, zorder=-100)
plt.tight_layout()
plt.savefig('beta_2_3.pdf')
plt.show()
