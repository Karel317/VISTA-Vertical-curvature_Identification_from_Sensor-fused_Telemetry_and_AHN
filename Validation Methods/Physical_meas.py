import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "Physical_meas_data.csv")

with open(filename, 'r') as f:
    labels = [h.strip() for h in f.readline().split(",")]

data = np.genfromtxt(filename, delimiter=",", skip_header=1, usecols=(0, 1, 2))

z = {}
x = {}
fig, axes = plt.subplots(3, 1, figsize=(9, 9))
for i, ax in enumerate(axes):
    col = data[:, i] - data[0, i]
    col = col[~np.isnan(col)]
    z[labels[i]] = -np.cumsum(col)
    x[labels[i]] = np.arange(len(z[labels[i]])) * 0.222
    ax.plot(x[labels[i]], -z[labels[i]])
    ax.set_title(labels[i])
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Cumulative height (cm)")
    ax.grid(True)

ax.set_xlabel("Distance (m)")
ax.set_ylabel("Height relative to first measurement (cm)")
ax.legend()
ax.grid(True)
plt.tight_layout()
#plt.show()

# ============================================================
# Save results to external drive


SAVE_DIR = r"D:\Validation_results\Physical_validation_data"
os.makedirs(SAVE_DIR, exist_ok=True)


TIME = 1779439914.817100912
method = "Physical_meas_stationsbrug"
np.savez_compressed(
    os.path.join(SAVE_DIR, f"{method}_{TIME}.npz"),
    x      = None,
    y      = None,
    z      = z[labels[1]],
    s      = x[labels[1]],
    kappa  = None,
    t      = None,
    method = method,

)
fpath = os.path.join(SAVE_DIR, f"{method}_{TIME}.npz")
print(f"Saved to {fpath}")
print(f"  Core   : x, y, z, s, t, method")



