import numpy as np

data = {
    5: {
        "pure steiner": [4, 13, 7, 14, 11, 8, 6, 8, 4, 7],
        "pure idle":    [0, 10, 4, 11, 9, 6, 3, 5, 4, 7],
        "mixed":        [2, 11, 8, 14, 10, 6, 6, 8, 3, 8],
    },
    10: {
        "pure steiner": [4, 10, 7, 16, 11, 7, 7, 8, 4, 9],
        "pure idle":    [0, 12, 4, 14, 6, 6, 5, 5, 3, 5],
        "mixed":        [4, 11, 3, 16, 9, 6, 4, 4, 3, 11],
    },
    15: {
        "pure steiner": [4, 11, 6, 15, 4, 6, 6, 9, 5, 10],
        "pure idle":    [1, 10, 2, 13, 6, 6, 8, 5, 2, 6],
        "mixed":        [2, 10, 6, 16, 8, 7, 6, 8, 4, 7],
    },
}

def avg_std(values):
    arr = np.array(values, dtype=float)
    avg = np.mean(arr)
    std = np.std(arr, ddof=1)   # sample standard deviation
    return avg, std

summary = {}

# Average/std for each radius, combining all three modes and all seeds.
for radius, mode_data in data.items():
    #print(radius, mode_data)
    values = []
    for mode_values in mode_data.values():
        values.extend(mode_values)
    #print(values)
    summary[f"r = {radius}"] = avg_std(values)

# Average/std for each mode, combining all three radii and all seeds.
for mode in ["pure steiner", "pure idle", "mixed"]:
    values = []
    for radius in data.keys():
        values.extend(data[radius][mode])

    summary[mode] = avg_std(values)

print(f"{'case':<15} {'avg':>10} {'std':>10}")
print("-" * 37)

for name, (avg, std) in summary.items():
    print(f"{name:<15} {avg:>10.3f} {std:>10.3f}")
