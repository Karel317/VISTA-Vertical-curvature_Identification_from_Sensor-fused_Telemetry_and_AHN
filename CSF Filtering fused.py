# =============================================================================
# METHOD 1: Ground Filtering using CSF (Cloth Simulation Filter)
# With multi-sensor fusion: M1P (front) + helios_L + helios_R (rear)
# All sensors transformed into base_link frame before merging.
# =============================================================================

import numpy as np
import open3d as o3d
import os
import sys

# =============================================================================
# SETTINGS — edit these before running
# =============================================================================

INPUT_FILE = r'C:\ROSBAGS VERWIJDER NA BEP\rosbag_0.mcap'

# Topics to merge
LIDAR_TOPICS = [
    "/rslidar/M1P_deskewed",
    "/rslidar/helios_R",
    "/rslidar/helios_L",
]

# Which frame to use (0 = first, or any index up to ~1725)
FRAME_INDEX = 1776

# CSF tuning parameters
CLOTH_RESOLUTION = 0.5
SLOPE_SMOOTH     = True
THRESHOLD        = 0.3


# =============================================================================
# SENSOR TRANSFORMS (from Foxglove TF tree, in base_link frame)
# Translation in meters, Rotation in degrees (roll, pitch, yaw)
# =============================================================================

SENSOR_TRANSFORMS = {
    "/rslidar/M1P_deskewed": {
        "translation": [0.800,  0.000, 0.848],
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


# =============================================================================
# TRANSFORM HELPERS
# =============================================================================

def rpy_deg_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Convert roll/pitch/yaw (degrees) to a 3x3 rotation matrix."""
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
    """
    Apply a rigid transform to an Nx3 point array.
    translation: [x, y, z] in meters
    rotation_rpy_deg: [roll, pitch, yaw] in degrees
    """
    R = rpy_deg_to_rotation_matrix(*rotation_rpy_deg)
    t = np.array(translation)
    return (R @ points.T).T + t


# =============================================================================
# MCAP LOADER
# =============================================================================

def _load_mcap_topic(filepath, topic, frame_index):
    """Load a single topic from an mcap file, return one frame as Nx3 numpy."""
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        print("  ERROR: mcap libraries not installed.")
        print("  Run:  pip install mcap mcap-ros2-support")
        sys.exit(1)

    frames = []
    with open(filepath, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=[topic]):
            pts = _ros_pc2_to_numpy(ros_msg)
            if pts is not None and len(pts) > 0:
                frames.append(pts)

    if len(frames) == 0:
        print(f"  WARNING: No messages found on topic '{topic}' — skipping.")
        return None

    # Pick closest available frame index
    idx = min(int(frame_index), len(frames) - 1)
    print(f"      [{topic}]  frame {idx}/{len(frames)-1}  "
          f"({len(frames[idx]):,} points)")
    return frames[idx]


def _ros_pc2_to_numpy(msg):
    """Parse a ROS PointCloud2 message into an Nx3 float32 array."""
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

    xyz = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        b = i * step
        xyz[i, 0] = struct.unpack_from("f", data, b + x_off)[0]
        xyz[i, 1] = struct.unpack_from("f", data, b + y_off)[0]
        xyz[i, 2] = struct.unpack_from("f", data, b + z_off)[0]

    return xyz[np.isfinite(xyz).all(axis=1)]


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def color_by_height(points):
    """Color points from blue (low) to red (high)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    z      = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    colors = np.zeros((len(z_norm), 3))
    colors[:, 0] = z_norm
    colors[:, 2] = 1.0 - z_norm
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def color_by_sensor(clouds_dict):
    """Color each sensor's points a distinct color for verification."""
    sensor_colors = {
        "/rslidar/M1P_deskewed": [1.0, 0.4, 0.0],   # orange
        "/rslidar/helios_R":     [0.0, 0.6, 1.0],   # cyan
        "/rslidar/helios_L":     [0.2, 0.9, 0.2],   # green
    }
    all_points = []
    all_colors = []
    for topic, pts in clouds_dict.items():
        color = sensor_colors.get(topic, [1, 1, 1])
        all_points.append(pts)
        all_colors.append(np.tile(color, (len(pts), 1)))

    combined_pts = np.vstack(all_points)
    combined_col = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pts)
    pcd.colors = o3d.utility.Vector3dVector(combined_col)
    return pcd


def show_pointcloud(pcd, title="Point Cloud"):
    print(f"\n  [Viewer] {title}")
    print("  Controls: left-drag=rotate | scroll=zoom | right-drag=pan | Q=quit")
    o3d.visualization.draw_geometries([pcd], window_name=title,
                                       width=1200, height=800)


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "="*60)
print("  METHOD 1: CSF Ground Filtering (Multi-sensor fusion)")
print("="*60)

# STEP 1 — Load and transform each sensor
print(f"\n[1/6] Loading frame {FRAME_INDEX} from {len(LIDAR_TOPICS)} sensors...")

transformed_clouds = {}
for topic in LIDAR_TOPICS:
    pts = _load_mcap_topic(INPUT_FILE, topic, FRAME_INDEX)
    if pts is None:
        continue
    tf = SENSOR_TRANSFORMS[topic]
    pts_transformed = transform_points(pts, tf["translation"], tf["rotation_rpy_deg"])
    transformed_clouds[topic] = pts_transformed

if len(transformed_clouds) == 0:
    print("  ERROR: No point clouds loaded. Check topics and frame index.")
    sys.exit(1)

# STEP 2 — Show sensor-colored cloud to verify alignment
print("\n[2/6] Showing sensor-colored cloud to verify alignment...")
print("      Orange = M1P (front) | Cyan = helios_R | Green = helios_L")
show_pointcloud(color_by_sensor(transformed_clouds),
    title="Sensor fusion — colored by sensor (close to continue)")

# STEP 3 — Merge all clouds
print("\n[3/6] Merging all sensors into one cloud...")
merged = np.vstack(list(transformed_clouds.values()))
print(f"      Total points: {len(merged):,}")
print(f"      X  {merged[:,0].min():.2f} → {merged[:,0].max():.2f} m")
print(f"      Y  {merged[:,1].min():.2f} → {merged[:,1].max():.2f} m")
print(f"      Z  {merged[:,2].min():.2f} → {merged[:,2].max():.2f} m")

# STEP 4 — Show merged cloud colored by height
print("\n[4/6] Showing merged cloud colored by height...")
show_pointcloud(color_by_height(merged),
    title="Merged cloud — colored by height (close to continue)")

# STEP 5 — Run CSF
print("\n[5/6] Running CSF ground filter...")
print(f"      cloth_resolution={CLOTH_RESOLUTION}m | "
      f"slope_smooth={SLOPE_SMOOTH} | threshold={THRESHOLD}m")

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

ground_points     = merged[np.array(list(ground_idx))]
non_ground_points = merged[np.array(list(non_ground_idx))]

pct_g  = 100 * len(ground_points) / len(merged)
pct_ng = 100 * len(non_ground_points) / len(merged)
print(f"\n      Ground points:     {len(ground_points):,}  ({pct_g:.1f}%)")
print(f"      Non-ground points: {len(non_ground_points):,}  ({pct_ng:.1f}%)")

# STEP 6 — Show and save ground only
print("\n[6/6] Showing ground-only point cloud...")
show_pointcloud(color_by_height(ground_points),
    title="CSF — Ground Points Only (close to continue)")

base        = os.path.splitext(INPUT_FILE)[0]
output_file = base + f"_frame{FRAME_INDEX}_fused_CSF_ground.ply"

out_pcd = o3d.geometry.PointCloud()
out_pcd.points = o3d.utility.Vector3dVector(ground_points)
o3d.io.write_point_cloud(output_file, out_pcd)

print(f"\n      Saved to: {output_file}")
print("\n" + "="*60)
print("  CSF fusion filtering complete!")
print("="*60 + "\n")