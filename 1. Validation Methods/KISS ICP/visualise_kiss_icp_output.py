from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================

# MODE: "npy"   — KISS-ICP output (poses.npy + timestamps.npy)
#        "kitti" — KITTI-format pose file (12 values per line, no timestamps)
MODE = "npy"

SCRIPT_DIR = Path(__file__).parent
RUN_NAME   = "2026_05_22 10_35_00"   # subfolder inside "KISS ICP results" (npy mode)
KITTI_FILE = r"C:\Users\Leons\Downloads\data_odometry_poses\dataset\poses\00.txt"  # kitti mode

# =============================================================================
# LOAD
# =============================================================================

if MODE == "npy":
    RESULTS_DIR = SCRIPT_DIR / "KISS ICP results" / RUN_NAME
    poses      = np.load(RESULTS_DIR / "poses.npy")       # (N, 4, 4)
    timestamps = np.load(RESULTS_DIR / "timestamps.npy")   # (N,)
    trajectory = poses[:, :3, 3]                           # (N, 3)
    title = f"KISS-ICP Trajectory  [{RUN_NAME}]"
elif MODE == "kitti":
    # Each line: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz  (3×4 matrix, row-major)
    data  = np.loadtxt(KITTI_FILE)           # (N, 12)
    mats  = data.reshape(-1, 3, 4)           # (N, 3, 4)
    raw   = mats[:, :, 3]                    # (N, 3): [tx, ty, tz] in camera frame
    # KITTI camera frame: x=right, y=down, z=forward
    # Remap to standard visualisation frame: x=right, y=forward, z=up
    trajectory = np.column_stack([raw[:, 0], raw[:, 2], -raw[:, 1]])
    timestamps = np.arange(len(trajectory), dtype=float)
    title = f"KITTI Trajectory  [{Path(KITTI_FILE).name}]"

else:
    raise ValueError(f"Unknown MODE: {MODE!r}. Use 'npy' or 'kitti'.")

print(f"Loaded {len(trajectory)} poses  [{MODE} mode]  — {title}")

# =============================================================================
# PLOT
# =============================================================================

x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

fig = plt.figure(figsize=(12, 5))
fig.suptitle(title, fontsize=12)

xlabel = "x / right (m)"
ylabel = "y / forward (m)" if MODE == "kitti" else "y (m)"
zlabel = "z / up (m)"      if MODE == "kitti" else "z (m)"

# Top-down
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x, y, linewidth=0.8)
ax1.set_title("Top-down trajectory (x/y)")
ax1.set_aspect("equal")
ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)

# 3D trajectory
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(x, y, z, linewidth=0.8)
ax2.set_title("3D trajectory")
ax2.set_xlabel(xlabel)
ax2.set_ylabel(ylabel)
ax2.set_zlabel(zlabel)

plt.tight_layout()
plt.show()
