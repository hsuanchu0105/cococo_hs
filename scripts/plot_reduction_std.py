import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = np.array([2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20])

# (reduction, std) per layout
single = np.array([
    (0.0406, 0.0481), (0.1182, 0.0202), (0.1428, 0.0372), (0.1620, 0.0322),
    (0.2179, 0.0292), (0.2422, 0.0143), (0.2731, 0.0171), (0.2771, 0.0191),
    (0.2809, 0.0219), (0.2809, 0.0219), (0.2809, 0.0219),
])
pair = np.array([
    (0.0530, 0.0320), (0.1287, 0.0305), (0.1616, 0.0228), (0.1881, 0.0239),
    (0.2055, 0.0276), (0.2343, 0.0383), (0.2566, 0.0401), (0.2617, 0.0372),
    (0.2591, 0.0367), (0.2591, 0.0367), (0.2591, 0.0367),
])
triple = np.array([
    (0.0902, 0.0466), (0.2052, 0.0271), (0.2351, 0.0279), (0.2453, 0.0268),
    (0.2724, 0.0306), (0.2856, 0.0292), (0.2940, 0.0228), (0.2902, 0.0248),
    (0.2902, 0.0248), (0.2902, 0.0248), (0.2902, 0.0248),
])
hex_ = np.array([
    (0.1025, 0.0371), (0.2191, 0.0357), (0.2510, 0.0335), (0.2681, 0.0447),
    (0.2956, 0.0395), (0.3014, 0.0387), (0.3036, 0.0364), (0.3036, 0.0364),
    (0.3036, 0.0364), (0.3036, 0.0364), (0.3036, 0.0364),
])

series = [
    ("single", single, "#4472C4", "none"),   # blue, open
    ("pair",   pair,   "#ED7D31", "#ED7D31"), # orange, filled
    ("triple", triple, "#595959", "none"),    # dark grey, open
    ("hex",    hex_,   "#FFC000", "#FFC000"),  # yellow, filled
]

fig, ax = plt.subplots(figsize=(9, 5.5))

# small horizontal offset so error bars don't overlap
offsets = [-0.18, -0.06, 0.06, 0.18]

for (label, data, color, face), dx in zip(series, offsets):
    y, err = data[:, 0], data[:, 1]
    ax.errorbar(
        x + dx, y, yerr=err,
        fmt="o", color=color, markerfacecolor=face,
        markeredgecolor=color, markeredgewidth=1.4, markersize=7,
        ecolor=color, elinewidth=1.2, capsize=3, capthick=1.2,
        label=label, zorder=3,
    )

ax.set_xlabel("max path", fontsize=12)
ax.set_ylabel("reduction ratio", fontsize=12)
ax.set_title("Reduction for different layout and max paths (fg)", fontsize=14)
ax.set_xlim(0, 25)
ax.set_ylim(0, 0.36)
ax.set_xticks(range(0, 26, 5))
ax.grid(True, color="#D9D9D9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="lower right")

fig.tight_layout()
fig.savefig("reduction_std.png", dpi=200)
print("saved reduction_std.png")
