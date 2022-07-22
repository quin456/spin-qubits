import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib import transforms
import numpy as np

fig = plt.figure(figsize=(5, 2), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 3])
axes = [fig.add_subplot(gs[i]) for i in range(3)]

ylabels = ["flat label", "bigger\nlabel", "even\nbigger\nlabel"]
labels = ["A", "B", "C"]

scaledtrans = transforms.ScaledTranslation(-0.4, 0, fig.dpi_scale_trans)

for ax, ylabel, label in zip(axes, ylabels, labels):
    ax.set_ylabel(ylabel)
    ax.text(0, 1, label, fontsize=12, fontweight="bold", va="bottom", ha="left",
           transform=ax.transAxes + scaledtrans)

plt.show()