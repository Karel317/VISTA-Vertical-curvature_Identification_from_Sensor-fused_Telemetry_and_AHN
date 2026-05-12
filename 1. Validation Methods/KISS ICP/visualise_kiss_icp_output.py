from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS  — point this at the run folder you want to inspect
# =============================================================================

SCRIPT_DIR  = Path(__file__).parent
RUN_NAME    = "28 april 17_30_00"   # subfolder inside "KISS ICP results"

# =============================================================================
# LOAD
# =============================================================================

RESULTS_DIR = SCRIPT_DIR / "KISS ICP results" / RUN_NAME

poses      = np.load(RESULTS_DIR / "poses.npy")       # (N, 4, 4)
timestamps = np.load(RESULTS_DIR / "timestamps.npy")   # (N,)

print(f"Loaded {len(poses)} poses from {RESULTS_DIR}/")

# =============================================================================
# PLOT
# =============================================================================

trajectory = poses[:, :3, 3]   # (N, 3)  x, y, z
x, y, z    = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
t          = timestamps

fig = plt.figure(figsize=(12, 3))
fig.suptitle(RUN_NAME, fontsize=12)

# Top-down x/y
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x, y, linewidth=0.8)
ax1.set_title("Top-down trajectory (x/y)")
ax1.set_aspect("equal")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

# Z over time — useful for spotting z drift
#ax2 = fig.add_subplot(1, 3, 2)
#ax2.plot(t, z, linewidth=0.8, color="coral")
#ax2.set_title("Z height over time")
#ax2.set_xlabel("time (s)")
#ax2.set_ylabel("z (m)")
#ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

# 3D trajectory
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(x, y, z, linewidth=0.8)
ax2.set_title("3D trajectory")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_zlabel("z (m)")

plt.tight_layout()
plt.show()
