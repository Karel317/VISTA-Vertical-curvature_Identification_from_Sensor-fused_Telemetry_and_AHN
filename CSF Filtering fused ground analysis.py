    # =============================================================================
# GROUND PROFILE ANALYSIS
# Computes and plots ground height, slope, and vertical curvature
# along the forward (X) axis of the bike using the fused CSF ground cloud.
# =============================================================================

import numpy as np
import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# =============================================================================
# SETTINGS
# =============================================================================

INPUT_FILE   = r'C:\ROSBAGS VERWIJDER NA BEP\rosbag_0.mcap'
FRAME_INDEX  = 1776

# Slice width: only use ground points within ±Y_SLICE meters of centerline
Y_SLICE = 1.0

# Bin size along X axis in meters (smaller = more detail, noisier)
BIN_SIZE = 0.2

# Smoothing window size (number of bins) for slope and curvature
# Increase if plots look too noisy
SMOOTH_WINDOW = 5

# Output plot file
OUTPUT_PLOT = r'C:\ROSBAGS VERWIJDER NA BEP\ground_profile.png'

# LiDAR topics and transforms (same as fusion script)
LIDAR_TOPICS = [
    "/rslidar/M1P_deskewed",
    "/rslidar/helios_R",
    "/rslidar/helios_L",
]

SENSOR_TRANSFORMS = {
    "/rslidar/M1P_deskewed": {
        "translation": [0.800,  0.000, 0.876],
        "rotation_rpy_deg": [0.1, -1.7,  1.7],
    },
    "/rslidar/helios_R": {
        "translation": [-0.743, -0.243, 0.857],
        "rotation_rpy_deg": [2.4, -2.6, -179.9],
    },
    "/rslidar/helios_L": {
        "translation": [-0.745,  0.191, 0.876],
        "rotation_rpy_deg": [3.5, -2.1,  179.9],
    },
}

# CSF parameters
CLOTH_RESOLUTION = 0.5
SLOPE_SMOOTH     = True
THRESHOLD        = 0.3


# =============================================================================
# TRANSFORM HELPERS
# =============================================================================

def rpy_deg_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0         ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,          1]])
    return Rz @ Ry @ Rx


def transform_points(points, translation, rotation_rpy_deg):
    R = rpy_deg_to_rotation_matrix(*rotation_rpy_deg)
    t = np.array(translation)
    return (R @ points.T).T + t


# =============================================================================
# MCAP LOADER
# =============================================================================

def _load_mcap_topic(filepath, topic, frame_index):
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        print("  ERROR: mcap libraries not installed.")
        sys.exit(1)

    frames = []
    with open(filepath, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=[topic]):
            pts = _ros_pc2_to_numpy(ros_msg)
            if pts is not None and len(pts) > 0:
                frames.append(pts)

    if len(frames) == 0:
        print(f"  WARNING: No messages on topic '{topic}' — skipping.")
        return None

    idx = min(int(frame_index), len(frames) - 1)
    print(f"      [{topic}]  frame {idx}/{len(frames)-1}  ({len(frames[idx]):,} points)")
    return frames[idx]


def _ros_pc2_to_numpy(msg):
    import struct
    field_map = {f.name: f for f in msg.fields}
    if not all(k in field_map for k in ("x", "y", "z")):
        return None
    x_off = field_map["x"].offset
    y_off = field_map["y"].offset
    z_off = field_map["z"].offset
    step  = msg.point_step
    data  = msg.data
    n     = msg.width * msg.height
    xyz   = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        b = i * step
        xyz[i, 0] = struct.unpack_from("f", data, b + x_off)[0]
        xyz[i, 1] = struct.unpack_from("f", data, b + y_off)[0]
        xyz[i, 2] = struct.unpack_from("f", data, b + z_off)[0]
    return xyz[np.isfinite(xyz).all(axis=1)]


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "="*60)
print("  GROUND PROFILE ANALYSIS")
print("="*60)

# STEP 1 — Load and transform all sensors
print(f"\n[1/5] Loading frame {FRAME_INDEX} from {len(LIDAR_TOPICS)} sensors...")
clouds = []
for topic in LIDAR_TOPICS:
    pts = _load_mcap_topic(INPUT_FILE, topic, FRAME_INDEX)
    if pts is None:
        continue
    tf  = SENSOR_TRANSFORMS[topic]
    pts = transform_points(pts, tf["translation"], tf["rotation_rpy_deg"])
    clouds.append(pts)

merged = np.vstack(clouds)
print(f"      Total points: {len(merged):,}")

# STEP 2 — Run CSF ground filter
print("\n[2/5] Running CSF ground filter...")
try:
    import CSF
except ImportError:
    print("  ERROR: CSF not installed.  Run:  pip install CSF")
    sys.exit(1)

csf = CSF.CSF()
csf.params.bSloopSmooth     = SLOPE_SMOOTH
csf.params.cloth_resolution = CLOTH_RESOLUTION
csf.params.threshold        = THRESHOLD
csf.setPointCloud(merged)

ground_idx     = CSF.VecInt()
non_ground_idx = CSF.VecInt()
csf.do_filtering(ground_idx, non_ground_idx)
ground = merged[np.array(list(ground_idx))]
print(f"      Ground points: {len(ground):,}  ({100*len(ground)/len(merged):.1f}%)")

# STEP 3 — Slice along centerline
print(f"\n[3/5] Slicing ground points within Y = ±{Y_SLICE}m of centerline...")
mask   = np.abs(ground[:, 1]) <= Y_SLICE
slice_pts = ground[mask]
print(f"      Points in slice: {len(slice_pts):,}")

if len(slice_pts) < 10:
    print("  ERROR: Too few points in slice. Try increasing Y_SLICE.")
    sys.exit(1)

# STEP 4 — Bin by X and compute median Z per bin
print(f"\n[4/5] Binning along X axis (bin size = {BIN_SIZE}m)...")
x = slice_pts[:, 0]
z = slice_pts[:, 2]

x_min = np.floor(x.min() / BIN_SIZE) * BIN_SIZE
x_max = np.ceil(x.max()  / BIN_SIZE) * BIN_SIZE
bins  = np.arange(x_min, x_max + BIN_SIZE, BIN_SIZE)

bin_centers = []
bin_heights = []

for i in range(len(bins) - 1):
    in_bin = (x >= bins[i]) & (x < bins[i+1])
    if in_bin.sum() >= 3:  # need at least 3 points per bin
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        bin_heights.append(np.median(z[in_bin]))

bin_centers = np.array(bin_centers)
bin_heights = np.array(bin_heights)
print(f"      Bins with data: {len(bin_centers)}")

# STEP 5 — Compute slope and curvature
print("\n[5/5] Computing slope and curvature...")

# Smooth height first to reduce noise
z_smooth = uniform_filter1d(bin_heights, size=SMOOTH_WINDOW)

# Slope = dZ/dX in degrees
dz = np.gradient(z_smooth, bin_centers)
slope_deg = np.degrees(np.arctan(dz))

# Curvature = d²Z/dX² (rate of change of slope)
d2z = np.gradient(dz, bin_centers)
curvature = d2z  # m^-1

# Smooth slope and curvature for cleaner plots
slope_smooth     = uniform_filter1d(slope_deg, size=SMOOTH_WINDOW)
curvature_smooth = uniform_filter1d(curvature, size=SMOOTH_WINDOW)

# Plot
print("      Generating plots...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(f"Ground Profile — Frame {FRAME_INDEX}  |  Y slice ±{Y_SLICE}m",
             fontsize=14, fontweight='bold')

# --- Plot 1: Ground height ---
ax1 = axes[0]
ax1.plot(bin_centers, z_smooth, color='royalblue', linewidth=1.5, label='Height (smoothed)')
ax1.scatter(bin_centers, bin_heights, color='lightsteelblue', s=8, alpha=0.5, label='Raw median Z')
ax1.set_ylabel("Height Z (m)", fontsize=11)
ax1.set_title("Ground Height", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')

# --- Plot 2: Slope ---
ax2 = axes[1]
ax2.plot(bin_centers, slope_smooth, color='darkorange', linewidth=1.5, label='Slope (smoothed)')
ax2.plot(bin_centers, slope_deg, color='moccasin', linewidth=0.8, alpha=0.6, label='Raw slope')
ax2.set_ylabel("Slope (degrees)", fontsize=11)
ax2.set_title("Ground Slope along Forward Direction", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')

# Shade positive/negative slope regions
ax2.fill_between(bin_centers, slope_smooth, 0,
                 where=(slope_smooth >= 0), alpha=0.15, color='red',    label='Uphill')
ax2.fill_between(bin_centers, slope_smooth, 0,
                 where=(slope_smooth <  0), alpha=0.15, color='blue',   label='Downhill')

# --- Plot 3: Curvature ---
ax3 = axes[2]
ax3.plot(bin_centers, curvature_smooth, color='green', linewidth=1.5, label='Curvature (smoothed)')
ax3.plot(bin_centers, curvature,        color='lightgreen', linewidth=0.8, alpha=0.6, label='Raw curvature')
ax3.set_ylabel("Curvature (m⁻¹)", fontsize=11)
ax3.set_xlabel("Forward distance X (m)", fontsize=11)
ax3.set_title("Vertical Curvature along Forward Direction", fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='gray', linewidth=0.5, linestyle='--')

# Shade convex/concave regions
ax3.fill_between(bin_centers, curvature_smooth, 0,
                 where=(curvature_smooth >= 0), alpha=0.15, color='orange', label='Convex (crest)')
ax3.fill_between(bin_centers, curvature_smooth, 0,
                 where=(curvature_smooth <  0), alpha=0.15, color='purple', label='Concave (dip)')

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"\n      Saved to: {OUTPUT_PLOT}")
plt.show()

print("\n" + "="*60)
print("  Ground profile analysis complete!")
print("="*60 + "\n")