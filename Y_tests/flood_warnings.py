import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Define ranges
flood_extent = np.linspace(0, 0.825, 100)
D = np.linspace(1, 6, 100)

# Create 2D grid
F, DD = np.meshgrid(flood_extent, D)

# Flood Index
FI = F * np.minimum(DD / 6, 1)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    F,
    DD,
    FI,
    cmap="viridis",
    edgecolor="none",
    alpha=0.9
)

# Axis labels
ax.set_xlabel("Flood extent")
ax.set_ylabel("Number of flooded dekads (D)")
ax.set_zlabel("Flood Index (FI)")

# Limits
ax.set_xlim(0, 0.825)
ax.set_ylim(1, 6)
ax.set_zlim(0, 0.825)

# Colorbar
fig.colorbar(surf, ax=ax, shrink=0.6, label="Flood Index (FI)")

# Constant threshold
FI_threshold = 0.625 * 2 / 6
FI_threshold_string = 'Threshold -> Flood extent (0.625) * n_deks (2), ' + str(round(FI_threshold, 2))

# Show the threshold plane only where it is above the FI surface
FI_plane = np.full_like(FI, FI_threshold)
FI_plane_masked = np.where(FI_plane >= FI, FI_plane, np.nan)

ax.plot_surface(
    F, DD, FI_plane_masked,
    color="red",
    alpha=0.45,
    edgecolor="none"
)
# Legend
threshold_legend = Patch(
    facecolor="red",
    alpha=0.45,
    label=FI_threshold_string
)

ax.legend(handles=[threshold_legend], loc="upper left")
plt.tight_layout()
plt.show()