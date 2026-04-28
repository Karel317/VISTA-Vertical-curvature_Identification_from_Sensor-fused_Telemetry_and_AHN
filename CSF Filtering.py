# =============================================================================
# METHOD 1: Ground Filtering using CSF (Cloth Simulation Filter)
# =============================================================================
# Supported input formats:
#   .las / .laz   — standard LiDAR exchange format
#   .ply          — polygon file format (common in research)
#   .pcd          — Point Cloud Data format (common in ROS)
#   .bin          — Velodyne/KITTI binary format (x, y, z, intensity as float32)
#   .mcap         — ROS2 bag format
#   .bag          — ROS1 bag format
#
# HOW TO INSTALL (run these in your terminal one by one):
#
#   Always needed:
#     pip install numpy open3d CSF
#
#   For .las / .laz files:
#     pip install laspy[lazrs]
#
#   For .mcap files (ROS2):
#     pip install mcap mcap-ros2-support
#
#   For .bag files (ROS1):
#     pip install rosbags
#
#   Note: .ply  .pcd  .bin are handled by open3d/numpy — nothing extra needed.
# =============================================================================

import numpy as np
import open3d as o3d
import os
import sys

# =============================================================================
# SETTINGS — edit these before running
# =============================================================================

INPUT_FILE = "your_file.mcap"    # <-- change to your actual file path

# For .mcap / .bag files only:
#   Set this to the ROS topic that contains LiDAR data.
#   If you don't know the topic name, set it to None and run the script —
#   it will print all available topics so you can pick the right one.
#   Common names: "/lidar/points"  "/velodyne_points"  "/os_cloud_node/points"
LIDAR_TOPIC = None

# For .mcap / .bag files only:
#   Which scan frame to use. 0 = first frame, 1 = second frame, etc.
#   Set to "all" to merge every frame into one large point cloud.
FRAME_INDEX = 0

# CSF tuning parameters:
#   cloth_resolution — grid cell size in meters
#     0.1–0.5 = fine detail, better for hilly/complex terrain (slower)
#     1.0–2.0 = coarse, good for flat terrain (faster)
#   slope_smooth
#     True  = rolling/gentle hills
#     False = steep or very rugged terrain
#   threshold — max distance (m) from the cloth to still count as ground
#     0.1 = strict (fewer ground points)
#     0.5 = permissive (more ground points)
CLOTH_RESOLUTION = 0.5
SLOPE_SMOOTH     = True
THRESHOLD        = 0.3


# =============================================================================
# UNIVERSAL LOADER — detects format automatically
# =============================================================================

def load_pointcloud(filepath, lidar_topic=None, frame_index=0):
    """
    Loads a point cloud from any supported format.
    Returns an Nx3 numpy array of (x, y, z) coordinates.
    """
    if not os.path.exists(filepath):
        print(f"\n  ERROR: File not found: '{filepath}'")
        print("  Check that INPUT_FILE is set to the correct path.")
        sys.exit(1)

    ext = os.path.splitext(filepath)[1].lower()
    print(f"      Format detected: {ext}")

    # --- .las / .laz ---
    if ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError:
            print("  ERROR: laspy is not installed.")
            print("  Run:  pip install laspy[lazrs]")
            sys.exit(1)
        las = laspy.read(filepath)
        return np.vstack([las.x, las.y, las.z]).T

    # --- .ply ---
    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            print("  ERROR: .ply loaded 0 points — check the file is valid.")
            sys.exit(1)
        return pts

    # --- .pcd ---
    elif ext == ".pcd":
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            print("  ERROR: .pcd loaded 0 points — check the file is valid.")
            sys.exit(1)
        return pts

    # --- .bin (Velodyne / KITTI) ---
    elif ext == ".bin":
        # Binary layout: repeated blocks of float32 (x, y, z, intensity)
        data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        return data[:, :3]   # drop intensity, keep x y z

    # --- .mcap (ROS2) ---
    elif ext == ".mcap":
        return _load_mcap(filepath, lidar_topic, frame_index)

    # --- .bag (ROS1) ---
    elif ext == ".bag":
        return _load_ros1bag(filepath, lidar_topic, frame_index)

    else:
        print(f"  ERROR: Format '{ext}' is not supported.")
        print("  Supported formats: .las  .laz  .ply  .pcd  .bin  .mcap  .bag")
        sys.exit(1)


# --- ROS2 .mcap loader ---

def _load_mcap(filepath, topic, frame_index):
    try:
        from mcap_ros2.reader import read_bag_messages
        from mcap.reader import make_reader
    except ImportError:
        print("  ERROR: mcap libraries not installed.")
        print("  Run:  pip install mcap mcap-ros2-support")
        sys.exit(1)

    # If no topic given, print all topics and exit so the user can choose
    if topic is None:
        print("\n  LIDAR_TOPIC is not set. Available topics in this file:")
        with open(filepath, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            for ch in summary.channels.values():
                n = summary.statistics.channel_message_counts.get(ch.id, "?")
                print(f"    {ch.topic}  —  {n} messages  [{ch.schema.name}]")
        print("\n  Set LIDAR_TOPIC at the top of this script and run again.")
        sys.exit(0)

    print(f"      Extracting topic '{topic}' ...")
    frames = []
    with open(filepath, "rb") as f:
        for msg in read_bag_messages(f, topics=[topic]):
            pts = _ros_pc2_to_numpy(msg.ros_msg)
            if pts is not None and len(pts) > 0:
                frames.append(pts)

    if len(frames) == 0:
        print(f"  ERROR: No messages found on topic '{topic}'.")
        print("  Set LIDAR_TOPIC = None to see all available topics.")
        sys.exit(1)

    return _select_frames(frames, frame_index)


# --- ROS1 .bag loader ---

def _load_ros1bag(filepath, topic, frame_index):
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.serde import deserialize_cdr
    except ImportError:
        print("  ERROR: rosbags is not installed.")
        print("  Run:  pip install rosbags")
        sys.exit(1)

    if topic is None:
        print("\n  LIDAR_TOPIC is not set. Available topics in this file:")
        with Reader(filepath) as reader:
            for conn in reader.connections:
                print(f"    {conn.topic}  [{conn.msgtype}]")
        print("\n  Set LIDAR_TOPIC at the top of this script and run again.")
        sys.exit(0)

    print(f"      Extracting topic '{topic}' ...")
    frames = []
    with Reader(filepath) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"  ERROR: Topic '{topic}' not found.")
            print("  Set LIDAR_TOPIC = None to see all available topics.")
            sys.exit(1)
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            pts = _ros_pc2_to_numpy(msg)
            if pts is not None and len(pts) > 0:
                frames.append(pts)

    return _select_frames(frames, frame_index)


# --- Convert ROS PointCloud2 message to numpy ---

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

    # Remove NaN / Inf values (common in spinning LiDAR scans)
    return xyz[np.isfinite(xyz).all(axis=1)]


# --- Frame selection helper ---

def _select_frames(frames, frame_index):
    print(f"      Found {len(frames)} scan frames.")
    if frame_index == "all":
        print(f"      Merging all frames into one cloud...")
        merged = np.vstack(frames)
        print(f"      Total points after merge: {len(merged):,}")
        return merged
    else:
        idx = min(int(frame_index), len(frames) - 1)
        print(f"      Using frame {idx}  ({len(frames[idx]):,} points)")
        return frames[idx]


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def color_by_height(points):
    """Color points from blue (low) to red (high) for easy terrain reading."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    z      = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    colors = np.zeros((len(z_norm), 3))
    colors[:, 0] = z_norm          # red   = high
    colors[:, 2] = 1.0 - z_norm   # blue  = low
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def show_pointcloud(pcd, title="Point Cloud"):
    """Open an interactive 3D viewer. Close the window to continue the script."""
    print(f"\n  [Viewer] {title}")
    print("  Controls: left-drag=rotate | scroll=zoom | right-drag=pan | Q=quit")
    o3d.visualization.draw_geometries([pcd], window_name=title,
                                       width=1200, height=800)


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "="*60)
print("  METHOD 1: CSF Ground Filtering")
print("="*60)

# STEP 1 — Load
print(f"\n[1/5] Loading: {INPUT_FILE}")
points = load_pointcloud(INPUT_FILE, LIDAR_TOPIC, FRAME_INDEX)
print(f"      Points loaded:  {len(points):,}")
print(f"      X  {points[:,0].min():.2f} → {points[:,0].max():.2f} m")
print(f"      Y  {points[:,1].min():.2f} → {points[:,1].max():.2f} m")
print(f"      Z  {points[:,2].min():.2f} → {points[:,2].max():.2f} m")

# STEP 2 — Visualize raw
print("\n[2/5] Showing RAW point cloud — close the window to continue...")
show_pointcloud(color_by_height(points),
    title="RAW Point Cloud — colored by height (close to continue)")

# STEP 3 — Run CSF
print("\n[3/5] Running CSF ground filter...")
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
csf.setPointCloud(points)

ground_idx     = CSF.VecInt()
non_ground_idx = CSF.VecInt()
csf.do_filtering(ground_idx, non_ground_idx)

ground_points     = points[np.array(list(ground_idx))]
non_ground_points = points[np.array(list(non_ground_idx))]

pct_g  = 100 * len(ground_points) / len(points)
pct_ng = 100 * len(non_ground_points) / len(points)
print(f"\n      Ground points:     {len(ground_points):,}  ({pct_g:.1f}%)")
print(f"      Non-ground points: {len(non_ground_points):,}  ({pct_ng:.1f}%)")

# STEP 4 — Visualize ground only
print("\n[4/5] Showing GROUND-ONLY point cloud — close the window to continue...")
show_pointcloud(color_by_height(ground_points),
    title="CSF — Ground Points Only (close to continue)")

# STEP 5 — Save as .ply (works regardless of input format)
print("\n[5/5] Saving ground-only point cloud...")
base        = os.path.splitext(INPUT_FILE)[0]
output_file = base + "_CSF_ground.ply"

out_pcd = o3d.geometry.PointCloud()
out_pcd.points = o3d.utility.Vector3dVector(ground_points)
o3d.io.write_point_cloud(output_file, out_pcd)

print(f"      Saved to: {output_file}")
print("\n" + "="*60)
print("  CSF filtering complete!")
print(f"  Output: {output_file}")
print("  Run method2_pointnet_filtering.py next to compare results.")
print("="*60 + "\n")
