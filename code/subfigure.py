
import matplotlib as mpl
import numpy as np
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a 3x3 grid of subplots
grid = GridSpec(3, 3)

sub_grid1 = GridSpec(2,3)

# Specify the position and size of the new subfigure
subfig_spec = grid.new_subplotspec((1, 0), rowspan=2, colspan=3)
# Create a main figure
fig = plt.figure()

# Add a subfigure using the subplotspec
subfig1 = fig.add_subfigure(subplotspec=subfig_spec)

m1=2; n1=3
ax1 = np.array([[None]*n1 for _ in range(m1)])

for i in range(m1):
    for j in range(n1):
        ax1[i,j] = subfig1.add_subplot(sub_grid1[i,j])

# Plot some data on the subfigure
ax1[0,0].plot([0, 1, 2], [0, 1, 0])

# Show the final plot
plt.show()


