# Running this does take a few minutes

from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from kiss_icp.pipeline import OdometryPipeline
from kiss_icp.datasets import dataset_factory

# =============================================================================
# SETTINGS
# =============================================================================

DATASET_PATH = Path(r"D:\Data_gathered\2026_05_22\Rosbag\10_40_00\rosbag\rosbag_0.mcap")   # path to mcap file
TOPIC        = "/rslidar/helios_L"   # Lidar topic

CUSTOM_PARAMETERS = False # Set to True to override individual parameters below instead of using kiss_icp.yaml
LIVE_VISUALIZE = False   # set to True to see a live visualization of the odometry results as they are computed (can slow down processing)

# =============================================================================
# LOAD CONFIG FROM YAML  (default is edit kiss_icp.yaml)
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
YAML_PATH  = SCRIPT_DIR / "kiss_icp.yaml"

with open(YAML_PATH, "r") as f:
    config = yaml.safe_load(f)

# =============================================================================
# CUSTOM PARAMETERS  (only applied when CUSTOM_PARAMETERS = True)
# =============================================================================

if CUSTOM_PARAMETERS:
    config["data"]["deskew"]    = False
    config["data"]["max_range"] = 40.0
    config["data"]["min_range"] = 1.0

    config["mapping"]["voxel_size"]           = 0.3
    config["mapping"]["max_points_per_voxel"] = 20

    config["registration"]["max_num_iterations"]    = 500
    config["registration"]["convergence_criterion"] = 0.0001
    config["adaptive_threshold"]["min_motion_th"]   = 0.05

# =============================================================================
# OUTPUT DIRECTORY  — "KISS ICP results/<dataset_name>/"
# =============================================================================

# Derive folder name from path:
_date = DATASET_PATH.parts[-5] 
_time = DATASET_PATH.parts[-3]
RESULTS_DIR = SCRIPT_DIR / "KISS ICP results" / f"{_date} {_time}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Write the (possibly modified) config to a temp file for the pipeline
_active_config = SCRIPT_DIR / "_active_config.yaml"
with open(_active_config, "w") as f:
    yaml.dump(config, f)

# Disable latest folder, kiss_icp creates this but you need extra permission
# and we don't need it
import os as _os
_os.symlink = lambda *_: None

# =============================================================================
# RUN KISS-ICP
# =============================================================================

dataset = dataset_factory(
    dataloader="mcap",
    data_dir=DATASET_PATH,
    topic=TOPIC,
)

pipeline = OdometryPipeline(
    dataset=dataset,
    config=_active_config,
    visualize=LIVE_VISUALIZE,
)
pipeline.run()

poses      = pipeline.poses   # list of N 4x4 numpy arrays (SE3)
timestamps = pipeline.times   # list of N timestamps (seconds)

# =============================================================================
# SAVE RESULTS
# =============================================================================

np.save(RESULTS_DIR / "poses.npy",      np.array(poses))
np.save(RESULTS_DIR / "timestamps.npy", np.array(timestamps))
print(f"Saved {len(poses)} poses to {RESULTS_DIR}/")

# =============================================================================
# PLOT TRAJECTORY
# =============================================================================

trajectory = np.array([p[:3, 3] for p in poses])   # (N, 3)  x, y, z
x, y, z    = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
t          = np.array(timestamps)

fig = plt.figure(figsize=(18, 5))

# Top-down x/y
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(x, y, linewidth=0.8)
ax1.set_title("Top-down trajectory (x/y)")
ax1.set_aspect("equal")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

# Z over time — most useful for spotting z drift
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(t, z, linewidth=0.8, color="coral")
ax2.set_title("Z height over time")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("z (m)")
ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

# 3D trajectory
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot(x, y, z, linewidth=0.8)
ax3.set_title("3D trajectory")
ax3.set_xlabel("x (m)")
ax3.set_ylabel("y (m)")
ax3.set_zlabel("z (m)")

plt.tight_layout()
plt.show()
