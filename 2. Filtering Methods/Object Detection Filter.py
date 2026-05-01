"""
==============================================================================
  POINT CLOUD GROUND FILTERING + OBJECT DETECTION
==============================================================================

  Pipeline (in order):
    1. Load point cloud (.bin, .mcap, .ply, .pcd, .las/.laz, .bag)
    2. Visualize the raw cloud (colored by height)
    3. Run RandLA-Net semantic segmentation -> split ground / non-ground
    4. Cluster non-ground points (DBSCAN) -> draw a 3D box per object
    5. Visualize: removed points = RED, kept (ground) points = GREEN
    6. Visualize: kept (ground) points only
    7. Save kept points to .ply

  WHY THIS APPROACH
  -----------------
  RandLA-Net gives a class label for every point (ground, vegetation, car,
  building...). That's perfect for a clean ground/non-ground split.
  But it does NOT separate individual objects — every car gets the same
  label, so you can't draw one box per car from labels alone.

  So we add DBSCAN clustering on the non-ground points: nearby points get
  grouped into "objects", and we draw an oriented 3D bounding box around
  each cluster. This is the standard LiDAR detection pipeline when you
  don't have a fully-trained 3D object detector handy.

  INSTALL
  -------
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install open3d open3d-ml numpy
    pip install mcap mcap-ros2-support     # for .mcap files
    pip install laspy[lazrs]               # for .las/.laz files
    pip install rosbags                    # for .bag files (ROS1)
==============================================================================
"""

import os
import sys
import numpy as np
import open3d as o3d


# =============================================================================
#  SETTINGS  --  edit these before running
# =============================================================================

INPUT_FILE = "your_file.bin"      # path to your .bin / .mcap / .ply / .pcd file

LIDAR_TOPIC = None                # for .mcap/.bag: ROS topic name, or None to list
FRAME_INDEX = 0                   # which frame to use, or "all" to merge

GROUND_LABELS = [1, 2]            # Semantic3D class IDs counted as "ground"

# DBSCAN clustering parameters for object detection on non-ground points
DBSCAN_EPS         = 0.5          # neighborhood radius in meters
DBSCAN_MIN_POINTS  = 20           # min points to form a cluster
MIN_CLUSTER_POINTS = 30           # ignore clusters smaller than this

# Pre-trained model
CHECKPOINT_PATH = "randlanet_semantic3d.pth"
CHECKPOINT_URL  = ("https://storage.googleapis.com/open3d-releases/"
                   "model-zoo/randlanet_semantic3d_202201071330utc.pth")

OUTPUT_SUFFIX = "_ground_only.ply"


# =============================================================================
#  LOADERS
# =============================================================================

def load_pointcloud(filepath, lidar_topic=None, frame_index=0):
    """Load any supported format. Returns Nx3 float32 array of (x,y,z)."""
    if not os.path.exists(filepath):
        sys.exit(f"\n  ERROR: file not found: {filepath!r}\n"
                 f"  Edit INPUT_FILE at the top of the script.")

    ext = os.path.splitext(filepath)[1].lower()
    print(f"      Format: {ext}")

    if ext in (".las", ".laz"):
        import laspy
        las = laspy.read(filepath)
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    if ext in (".ply", ".pcd"):
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0:
            sys.exit(f"  ERROR: {ext} file loaded 0 points.")
        return pts

    if ext == ".bin":
        # KITTI/Velodyne: float32 records of 4 (x,y,z,intensity) or 3 (x,y,z)
        raw = np.fromfile(filepath, dtype=np.float32)
        if raw.size % 4 == 0:
            return raw.reshape(-1, 4)[:, :3]
        if raw.size % 3 == 0:
            return raw.reshape(-1, 3)
        sys.exit(f"  ERROR: .bin size ({raw.size} floats) is not divisible by 3 or 4.")

    if ext == ".mcap":
        return _load_mcap(filepath, lidar_topic, frame_index)

    if ext == ".bag":
        return _load_ros1bag(filepath, lidar_topic, frame_index)

    sys.exit(f"  ERROR: unsupported format {ext!r}.\n"
             f"  Supported: .bin .mcap .ply .pcd .las .laz .bag")


def _load_mcap(filepath, topic, frame_index):
    try:
        from mcap_ros2.decoder import DecoderFactory
        from mcap.reader import make_reader
    except ImportError:
        sys.exit("  ERROR: mcap libs missing.  pip install mcap mcap-ros2-support")

    # If no topic given, list available ones and exit
    if topic is None:
        print("\n  LIDAR_TOPIC not set. Available topics in this file:")
        with open(filepath, "rb") as f:
            reader = make_reader(f)
            for ch in reader.get_summary().channels.values():
                print(f"    {ch.topic}   [{ch.schema.name}]")
        print("\n  Set LIDAR_TOPIC to one of the above and run again.")
        sys.exit(0)

    print(f"      Reading topic '{topic}' ...")
    frames = []
    with open(filepath, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, ros_msg in reader.iter_decoded_messages(topics=[topic]):
            pts = _ros_pc2_to_numpy(ros_msg)
            if pts is not None and len(pts) > 0:
                frames.append(pts)

    if not frames:
        sys.exit(f"  ERROR: no PointCloud2 messages on topic {topic!r}.")
    return _select_frames(frames, frame_index)


def _load_ros1bag(filepath, topic, frame_index):
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.serde import deserialize_cdr
    except ImportError:
        sys.exit("  ERROR: rosbags missing.  pip install rosbags")

    if topic is None:
        print("\n  LIDAR_TOPIC not set. Available topics:")
        with Reader(filepath) as reader:
            for conn in reader.connections:
                print(f"    {conn.topic}   [{conn.msgtype}]")
        sys.exit(0)

    frames = []
    with Reader(filepath) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            sys.exit(f"  ERROR: topic {topic!r} not found.")
        for conn, _, raw in reader.messages(connections=conns):
            msg = deserialize_cdr(raw, conn.msgtype)
            pts = _ros_pc2_to_numpy(msg)
            if pts is not None and len(pts) > 0:
                frames.append(pts)
    return _select_frames(frames, frame_index)


def _ros_pc2_to_numpy(msg):
    """Vectorized PointCloud2 -> Nx3 float32 array."""
    field_map = {f.name: f for f in msg.fields}
    if not all(k in field_map for k in ("x", "y", "z")):
        return None
    n    = msg.width * msg.height
    step = msg.point_step
    buf  = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, step)
    def grab(name):
        off = field_map[name].offset
        return buf[:, off:off+4].copy().view(np.float32).ravel()
    xyz = np.stack([grab("x"), grab("y"), grab("z")], axis=1)
    return xyz[np.isfinite(xyz).all(axis=1)]


def _select_frames(frames, frame_index):
    print(f"      Found {len(frames)} scan frames.")
    if frame_index == "all":
        merged = np.vstack(frames)
        print(f"      Merged all -> {len(merged):,} points")
        return merged
    idx = min(int(frame_index), len(frames) - 1)
    print(f"      Using frame {idx} ({len(frames[idx]):,} points)")
    return frames[idx]


# =============================================================================
#  VISUALIZATION
# =============================================================================

def _make_pcd(points, colors=None):
    """Build an Open3D point cloud from a numpy array (+ optional colors)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def color_by_height(points):
    """Blue (low) -> red (high) gradient -- nice for inspecting raw clouds."""
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = z_norm           # red increases with height
    colors[:, 2] = 1.0 - z_norm     # blue decreases with height
    return _make_pcd(points, colors)


def color_solid(points, rgb):
    """All points one color."""
    colors = np.tile(np.asarray(rgb, dtype=np.float64), (len(points), 1))
    return _make_pcd(points, colors)


def show(geometries, title):
    """Open the interactive viewer. Window must be closed (Q) to continue."""
    if not isinstance(geometries, list):
        geometries = [geometries]
    print(f"\n  [Viewer] {title}")
    print("  Controls: drag=rotate | scroll=zoom | shift+drag=pan | Q=quit")
    o3d.visualization.draw_geometries(
        geometries, window_name=title, width=1280, height=800
    )


# =============================================================================
#  GROUND / NON-GROUND SEGMENTATION  (RandLA-Net pre-trained)
# =============================================================================

def segment_ground(points):
    """Return per-point class labels using the pre-trained RandLA-Net model."""
    try:
        import open3d.ml.torch as ml3d
    except ImportError:
        sys.exit("  ERROR: open3d-ml missing.  pip install open3d-ml")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"      Downloading model weights (~20MB) ...")
        import urllib.request
        try:
            urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
        except Exception as e:
            sys.exit(f"      Download failed: {e}\n"
                     f"      Get it manually from "
                     f"https://github.com/isl-org/Open3D-ML#pretrained-models\n"
                     f"      and save as {CHECKPOINT_PATH}")

    model    = ml3d.models.RandLANet(num_classes=8, num_neighbors=16, num_layers=4)
    pipeline = ml3d.pipelines.SemanticSegmentation(model=model, device="cpu")
    pipeline.load_ckpt(CHECKPOINT_PATH)

    print("      Running inference (this can take 1-5 min on CPU) ...")
    result = pipeline.run_inference({"point": points, "feat": None})
    return np.asarray(result["predict_labels"])


def report_class_breakdown(labels):
    names = {0: "unlabelled", 1: "man-made terrain (GROUND)",
             2: "natural terrain (GROUND)", 3: "high vegetation",
             4: "low vegetation", 5: "buildings", 6: "hard scape",
             7: "scan artefacts", 8: "cars"}
    print("\n      Class breakdown:")
    for cid, name in names.items():
        count = int(np.sum(labels == cid))
        if count > 0:
            print(f"        [{cid}] {name:<30}  {count:>10,} pts "
                  f"({100*count/len(labels):.1f}%)")


# =============================================================================
#  OBJECT DETECTION  (DBSCAN clustering + bounding boxes)
# =============================================================================

def detect_objects(non_ground_points,
                   eps=DBSCAN_EPS,
                   min_pts=DBSCAN_MIN_POINTS,
                   min_cluster=MIN_CLUSTER_POINTS):
    """
    Cluster non-ground points into discrete objects using DBSCAN, then
    fit an oriented 3D bounding box to each cluster.

    Returns
    -------
    cluster_ids : (N,) int array  -- cluster id per input point, -1 = noise
    bboxes      : list of o3d.geometry.OrientedBoundingBox
    """
    if len(non_ground_points) == 0:
        return np.array([], dtype=int), []

    pcd = _make_pcd(non_ground_points)

    # Open3D ships an efficient DBSCAN implementation
    print(f"      DBSCAN  eps={eps}m  min_points={min_pts} ...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        cluster_ids = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_pts))

    n_clusters = cluster_ids.max() + 1 if cluster_ids.size and cluster_ids.max() >= 0 else 0
    print(f"      Found {n_clusters} raw clusters "
          f"({np.sum(cluster_ids == -1):,} noise points)")

    bboxes = []
    kept = 0
    for cid in range(n_clusters):
        mask = cluster_ids == cid
        if mask.sum() < min_cluster:
            continue
        cluster_pts = non_ground_points[mask]
        try:
            box = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(cluster_pts)
            )
            box.color = (1.0, 0.6, 0.0)   # orange boxes
            bboxes.append(box)
            kept += 1
        except RuntimeError:
            pass    # too few unique points for a 3D box

    print(f"      Kept {kept} object boxes (>= {min_cluster} points each)")
    return cluster_ids, bboxes


def color_by_cluster(points, cluster_ids):
    """Distinct color per cluster, dark gray for noise."""
    colors = np.full((len(points), 3), 0.25)   # noise = dark gray
    n_clusters = cluster_ids.max() + 1 if cluster_ids.size and cluster_ids.max() >= 0 else 0
    if n_clusters > 0:
        rng = np.random.default_rng(42)
        palette = rng.uniform(0.3, 1.0, size=(n_clusters, 3))
        valid = cluster_ids >= 0
        colors[valid] = palette[cluster_ids[valid]]
    return _make_pcd(points, colors)


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 64)
    print("  POINT CLOUD GROUND FILTERING + OBJECT DETECTION")
    print("=" * 64)

    # --- 1. LOAD --------------------------------------------------------------
    print(f"\n[1/6] Loading {INPUT_FILE}")
    points = load_pointcloud(INPUT_FILE, LIDAR_TOPIC, FRAME_INDEX)
    print(f"      Points: {len(points):,}")
    print(f"      X {points[:,0].min():7.2f} -> {points[:,0].max():7.2f} m")
    print(f"      Y {points[:,1].min():7.2f} -> {points[:,1].max():7.2f} m")
    print(f"      Z {points[:,2].min():7.2f} -> {points[:,2].max():7.2f} m")

    # --- 2. VISUALIZE RAW -----------------------------------------------------
    print("\n[2/6] Showing RAW point cloud (colored by height)")
    show(color_by_height(points), "1. Raw point cloud (height = blue->red)")

    # --- 3. SEGMENT GROUND vs NON-GROUND -------------------------------------
    print("\n[3/6] Running RandLA-Net semantic segmentation")
    labels        = segment_ground(points)
    report_class_breakdown(labels)

    ground_mask   = np.isin(labels, GROUND_LABELS)
    ground_pts    = points[ground_mask]
    nonground_pts = points[~ground_mask]
    print(f"\n      Ground:     {len(ground_pts):,} pts "
          f"({100*len(ground_pts)/len(points):.1f}%)")
    print(f"      Non-ground: {len(nonground_pts):,} pts "
          f"({100*len(nonground_pts)/len(points):.1f}%)")

    # --- 4. DETECT OBJECTS + DRAW BOXES --------------------------------------
    print("\n[4/6] Detecting objects in non-ground points (DBSCAN)")
    cluster_ids, bboxes = detect_objects(nonground_pts)

    print(f"\n      Showing {len(bboxes)} detected objects with bounding boxes")
    cluster_pcd = color_by_cluster(nonground_pts, cluster_ids)
    ground_dim  = color_solid(ground_pts, [0.3, 0.3, 0.3])   # ground in gray
    show([ground_dim, cluster_pcd, *bboxes],
         f"2. Detected objects ({len(bboxes)} boxes) + ground in gray")

    # --- 5. RED (removed) vs GREEN (kept) ------------------------------------
    print("\n[5/6] Showing removed (RED) vs kept (GREEN)")
    removed_red  = color_solid(nonground_pts, [1.0, 0.15, 0.15])
    kept_green   = color_solid(ground_pts,    [0.15, 0.85, 0.15])
    show([removed_red, kept_green],
         "3. RED = removed (non-ground)   GREEN = kept (ground)")

    # --- 6. KEPT POINTS ONLY -------------------------------------------------
    print("\n      Showing KEPT points only")
    show(color_by_height(ground_pts),
         "4. Kept points only (ground, colored by height)")

    # --- 7. SAVE -------------------------------------------------------------
    base   = os.path.splitext(INPUT_FILE)[0]
    output = base + OUTPUT_SUFFIX
    o3d.io.write_point_cloud(output, _make_pcd(ground_pts))
    print(f"\n[6/6] Saved: {output}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
