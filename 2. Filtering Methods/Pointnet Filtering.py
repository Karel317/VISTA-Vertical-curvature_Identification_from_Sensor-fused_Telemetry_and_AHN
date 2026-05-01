# =============================================================================
# METHOD 2: Ground Filtering using RandLA-Net / PointNet++ (pre-trained)
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
#   Step 1 — PyTorch (pick ONE based on your hardware):
#     CPU only (works on any machine, no GPU needed):
#       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#     NVIDIA GPU (much faster if you have one):
#       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#
#   Step 2 — Open3D and Open3D-ML:
#     pip install open3d
#     pip install open3d-ml
#
#   Step 3 — Other dependencies:
#     pip install numpy
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
# The pre-trained model is trained on the Semantic3D outdoor LiDAR dataset.
# It labels every point with one of these classes:
#   0 = unlabelled        5 = buildings
#   1 = man-made terrain  6 = hard scape (pavements, etc.)
#   2 = natural terrain   7 = scan artefacts
#   3 = high vegetation   8 = cars
#   4 = low vegetation
# Classes 1 and 2 are what we keep as "ground".
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
#   If you don't know the topic name, set it to None — the script will
#   print all available topics so you can pick the right one.
LIDAR_TOPIC = None

# Which scan frame to use. 0 = first, 1 = second, etc.
# Set to "all" to merge all frames into one cloud.
FRAME_INDEX = 0

# Ground class IDs from the Semantic3D dataset:
#   1 = man-made terrain (roads, gravel, paths)
#   2 = natural terrain  (grass, dirt, bare earth)
GROUND_LABELS = [1, 2]

# Path to the pre-trained model weights.
# The script will try to download them automatically on first run (~20MB).
CHECKPOINT_PATH = "randlanet_semantic3d.pth"
CHECKPOINT_URL  = ("https://storage.googleapis.com/open3d-releases/"
                   "model-zoo/randlanet_semantic3d_202201071330utc.pth")

# Path to the CSF ground output from method1 for comparison.
# Set to None if you haven't run method1 yet — comparison will be skipped.
CSF_OUTPUT_FILE = None   # e.g. "your_file_CSF_ground.ply"


# =============================================================================
# UNIVERSAL LOADER — identical to method1, supports all formats
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

    if ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError:
            print("  ERROR: laspy not installed.  Run:  pip install laspy[lazrs]")
            sys.exit(1)
        las = laspy.read(filepath)
        return np.vstack([las.x, las.y, las.z]).T

    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            print("  ERROR: .ply loaded 0 points.")
            sys.exit(1)
        return pts

    elif ext == ".pcd":
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            print("  ERROR: .pcd loaded 0 points.")
            sys.exit(1)
        return pts

    elif ext == ".bin":
        data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        return data[:, :3]

    elif ext == ".mcap":
        return _load_mcap(filepath, lidar_topic, frame_index)

    elif ext == ".bag":
        return _load_ros1bag(filepath, lidar_topic, frame_index)

    else:
        print(f"  ERROR: Format '{ext}' not supported.")
        print("  Supported: .las  .laz  .ply  .pcd  .bin  .mcap  .bag")
        sys.exit(1)


def _load_mcap(filepath, topic, frame_index):
    try:
        from mcap_ros2.reader import read_bag_messages
        from mcap.reader import make_reader
    except ImportError:
        print("  ERROR: mcap libraries not installed.")
        print("  Run:  pip install mcap mcap-ros2-support")
        sys.exit(1)

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
        print(f"  ERROR: No messages on topic '{topic}'.")
        print("  Set LIDAR_TOPIC = None to list available topics.")
        sys.exit(1)

    return _select_frames(frames, frame_index)


def _load_ros1bag(filepath, topic, frame_index):
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.serde import deserialize_cdr
    except ImportError:
        print("  ERROR: rosbags not installed.  Run:  pip install rosbags")
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
            print("  Set LIDAR_TOPIC = None to list available topics.")
            sys.exit(1)
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            pts = _ros_pc2_to_numpy(msg)
            if pts is not None and len(pts) > 0:
                frames.append(pts)

    return _select_frames(frames, frame_index)


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


def _select_frames(frames, frame_index):
    print(f"      Found {len(frames)} scan frames.")
    if frame_index == "all":
        merged = np.vstack(frames)
        print(f"      Merged all frames: {len(merged):,} points total")
        return merged
    else:
        idx = min(int(frame_index), len(frames) - 1)
        print(f"      Using frame {idx}  ({len(frames[idx]):,} points)")
        return frames[idx]


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


def color_by_label(points, labels):
    """
    Color each point by its predicted semantic class so you can
    visually inspect what the model detected.
    """
    class_colors = {
        0: [0.5, 0.5, 0.5],   # unlabelled       → gray
        1: [0.6, 0.4, 0.1],   # man-made terrain  → brown   ← GROUND
        2: [0.2, 0.7, 0.2],   # natural terrain   → green   ← GROUND
        3: [0.0, 0.4, 0.0],   # high vegetation   → dark green
        4: [0.6, 0.9, 0.4],   # low vegetation    → light green
        5: [0.8, 0.2, 0.2],   # buildings         → red
        6: [0.5, 0.5, 0.8],   # hard scape        → blue-gray
        7: [0.9, 0.9, 0.0],   # scan artefacts    → yellow
        8: [0.9, 0.5, 0.0],   # cars              → orange
    }
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.array([class_colors.get(int(l), [0.5, 0.5, 0.5]) for l in labels])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def show_pointcloud(pcd, title="Point Cloud"):
    """Open an interactive 3D viewer. Close the window to continue."""
    print(f"\n  [Viewer] {title}")
    print("  Controls: left-drag=rotate | scroll=zoom | right-drag=pan | Q=quit")
    o3d.visualization.draw_geometries([pcd], window_name=title,
                                       width=1200, height=800)


def show_comparison(pcd_csf, pcd_pn):
    """
    Show CSF ground and PointNet++ ground side by side in one viewer.
    CSF result is shifted to the left, PointNet++ to the right.
    """
    csf_pts = np.asarray(pcd_csf.points)
    x_range = csf_pts[:, 0].max() - csf_pts[:, 0].min()
    offset  = x_range * 1.2    # gap between the two clouds

    left  = o3d.geometry.PointCloud(pcd_csf)
    right = o3d.geometry.PointCloud(pcd_pn)
    left.translate([-offset / 2, 0, 0])
    right.translate([ offset / 2, 0, 0])

    print("\n  [Viewer] Side-by-side comparison")
    print("           LEFT  = CSF result")
    print("           RIGHT = PointNet++ result")
    print("  Controls: left-drag=rotate | scroll=zoom | right-drag=pan | Q=quit")
    o3d.visualization.draw_geometries(
        [left, right],
        window_name="Comparison — CSF (left)  vs  PointNet++ (right)",
        width=1400, height=800
    )


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "="*60)
print("  METHOD 2: PointNet++ / RandLA-Net Ground Filtering")
print("="*60)

# STEP 1 — Load
print(f"\n[1/6] Loading: {INPUT_FILE}")
points = load_pointcloud(INPUT_FILE, LIDAR_TOPIC, FRAME_INDEX)
print(f"      Points loaded:  {len(points):,}")
print(f"      X  {points[:,0].min():.2f} → {points[:,0].max():.2f} m")
print(f"      Y  {points[:,1].min():.2f} → {points[:,1].max():.2f} m")
print(f"      Z  {points[:,2].min():.2f} → {points[:,2].max():.2f} m")

# STEP 2 — Visualize raw
print("\n[2/6] Showing RAW point cloud — close the window to continue...")
show_pointcloud(color_by_height(points),
    title="RAW Point Cloud — colored by height (close to continue)")

# STEP 3 — Load pre-trained model
print("\n[3/6] Loading pre-trained RandLA-Net model (Semantic3D weights)...")

try:
    import open3d.ml.torch as ml3d
except ImportError:
    print("  ERROR: open3d-ml is not installed.")
    print("  Run:  pip install open3d-ml")
    sys.exit(1)

# Download weights automatically if not already present
if not os.path.exists(CHECKPOINT_PATH):
    print(f"      Downloading model weights (~20MB)...")
    import urllib.request
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
        print(f"      Downloaded: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"      Auto-download failed: {e}")
        print("      Download manually from:")
        print("      https://github.com/isl-org/Open3D-ML#pretrained-models")
        print(f"      Save the file as: {CHECKPOINT_PATH}")
        sys.exit(1)

model    = ml3d.models.RandLANet(num_classes=8, num_neighbors=16, num_layers=4)
pipeline = ml3d.pipelines.SemanticSegmentation(model=model, device="cpu")
# Tip: change device="cpu" to device="cuda" if you have an NVIDIA GPU

pipeline.load_ckpt(CHECKPOINT_PATH)
print("      Model loaded successfully.")

# STEP 4 — Run inference (label every point)
print("\n[4/6] Running inference on your point cloud...")
print("      Every point will be labelled: terrain / vegetation / building / car / etc.")
print("      This can take 1–5 minutes depending on point count and hardware.")

data   = {"point": points, "feat": None}
result = pipeline.run_inference(data)
labels = np.array(result["predict_labels"])

# Print class breakdown
class_names = {
    0: "unlabelled",
    1: "man-made terrain  ← GROUND",
    2: "natural terrain   ← GROUND",
    3: "high vegetation",
    4: "low vegetation",
    5: "buildings",
    6: "hard scape",
    7: "scan artefacts",
    8: "cars",
}
print("\n      Detected classes:")
for cid, name in class_names.items():
    count = int(np.sum(labels == cid))
    if count > 0:
        print(f"        Class {cid} — {name}: {count:,} pts "
              f"({100*count/len(labels):.1f}%)")

# Show all points colored by class
print("\n      Showing all points colored by class — close the window to continue...")
print("      Legend: brown/green = ground | red = buildings | dark green = trees")
show_pointcloud(color_by_label(points, labels),
    title="PointNet++ — all points by class (close to continue)")

# STEP 5 — Extract and show ground only
print("\n[5/6] Extracting ground points...")
ground_mask   = np.isin(labels, GROUND_LABELS)
ground_points = points[ground_mask]

pct_g  = 100 * len(ground_points) / len(points)
pct_ng = 100 * (1 - len(ground_points) / len(points))
print(f"      Ground points:     {len(ground_points):,}  ({pct_g:.1f}%)")
print(f"      Non-ground points: {len(points)-len(ground_points):,}  ({pct_ng:.1f}%)")

print("\n      Showing GROUND-ONLY point cloud — close the window to continue...")
show_pointcloud(color_by_height(ground_points),
    title="PointNet++ — Ground Points Only (close to continue)")

# STEP 6 — Side-by-side comparison with CSF result
print("\n[6/6] Comparison with CSF result...")
pn_pcd = color_by_height(ground_points)

if CSF_OUTPUT_FILE and os.path.exists(CSF_OUTPUT_FILE):
    print(f"      Loading CSF result: {CSF_OUTPUT_FILE}")
    csf_pcd_raw = o3d.io.read_point_cloud(CSF_OUTPUT_FILE)
    csf_pts     = np.asarray(csf_pcd_raw.points)
    csf_pcd     = color_by_height(csf_pts)
    print(f"      CSF ground points:     {len(csf_pts):,}")
    print(f"      PointNet++ ground pts: {len(ground_points):,}")
    print("\n      Showing side-by-side — close the window when done.")
    show_comparison(csf_pcd, pn_pcd)
elif CSF_OUTPUT_FILE:
    print(f"      File not found: '{CSF_OUTPUT_FILE}' — skipping comparison.")
else:
    print("      CSF_OUTPUT_FILE is not set — skipping comparison.")
    print("      To compare: run method1 first, then set CSF_OUTPUT_FILE at")
    print("      the top of this script to the .ply file it saved.")

# Save output
print("\nSaving PointNet++ ground-only point cloud...")
base        = os.path.splitext(INPUT_FILE)[0]
output_file = base + "_PointNet_ground.ply"

out_pcd = o3d.geometry.PointCloud()
out_pcd.points = o3d.utility.Vector3dVector(ground_points)
o3d.io.write_point_cloud(output_file, out_pcd)

print(f"      Saved to: {output_file}")
print("\n" + "="*60)
print("  PointNet++ filtering complete!")
print(f"  Output: {output_file}")
print("="*60 + "\n")
