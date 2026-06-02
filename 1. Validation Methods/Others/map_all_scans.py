from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# =============================================================================
# SETTINGS
# =============================================================================

DATASET_PATH           = Path(r"C:\Users\Leons\OneDrive - Delft University of Technology\BEP\Users\kuitbrug_Users_OneDrive - Delft University of Technology.mcap")
RESULTS_DIR            = Path(__file__).parent / "KISS ICP results" / "Users chnology"
LIDAR_TOPICS           = ["/rslidar/M1P_deskewed"]
VOXEL_SIZE_DISPLAY     = 0.15   # metres — voxel size for display downsampling
EVERY_N_SCANS          = 5      # 1 = all scans; set to e.g. 5 for a quick preview
BATCH_DOWNSAMPLE_EVERY = 200    # flush + downsample every N processed scans
CORRIDOR_RADIUS        = 10.0   # metres — only keep points within this distance of the driven path

SAVE_MAP     = True             # set to False to skip saving
#SAVE_PATH    = Path(r"D:\ONLY non-flat surfaces\Pointclouds") / f"{DATASET_PATH.stem}_map.bin"
SAVE_PATH    = Path(r"C:\Users\Leons\OneDrive - Delft University of Technology\BEP\Users\kuitbrug_Users_OneDrive - Delft University of Technology") / f"{DATASET_PATH.stem}_map.bin"


# =============================================================================
# SENSOR EXTRINSICS  (sensor → base_link, from Foxglove TF tree)
# =============================================================================

SENSOR_TRANSFORMS = {
    "/rslidar/M1P_deskewed": {
        "translation":      [0.800,  0.000, 0.848],
        "rotation_rpy_deg": [0.1, -1.7,  1.7],
    },
    # "/rslidar/helios_R": {
    #     "translation":      [-0.743, -0.243, 0.857],
    #     "rotation_rpy_deg": [2.4, -2.6, -179.9],
    # },
    # "/rslidar/helios_L": {
    #     "translation":      [-0.745,  0.191, 0.876],
    #     "rotation_rpy_deg": [3.5, -2.1,  179.9],
    # },
}

# =============================================================================
# HELPERS
# =============================================================================

def _rpy_to_matrix4(roll_deg, pitch_deg, yaw_deg, translation):
    """Build a 4×4 rigid transform from RPY (degrees) + translation."""
    R = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = translation
    return T


def _ros_pc2_to_numpy(msg):
    """Parse a ROS PointCloud2 message into an Nx3 float32 array (vectorised)."""
    field_map = {f.name: f for f in msg.fields}
    if not all(k in field_map for k in ("x", "y", "z")):
        return None
    n    = msg.width * msg.height
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
    xyz  = np.empty((n, 3), dtype=np.float32)
    for i, name in enumerate(("x", "y", "z")):
        f = field_map[name]
        xyz[:, i] = data[:, f.offset:f.offset + 4].view(np.float32).flatten()
    return xyz[np.isfinite(xyz).all(axis=1)]


# =============================================================================
# STEP 1 — Load KISS-ICP poses and fix timestamps
# =============================================================================

print(f"\nLoading KISS-ICP poses from {RESULTS_DIR}/")
poses_raw       = np.load(RESULTS_DIR / "poses.npy")        # (N, 4, 4)
pose_timestamps = np.load(RESULTS_DIR / "timestamps.npy")   # (N,)

# Sort and deduplicate — Slerp requires strictly increasing times
_sort = np.argsort(pose_timestamps, kind="stable")
pose_timestamps = pose_timestamps[_sort]
poses_raw       = poses_raw[_sort]
_unique = np.concatenate([[True], np.diff(pose_timestamps) > 0])
pose_timestamps = pose_timestamps[_unique]
poses_raw       = poses_raw[_unique]
print(f"  {len(poses_raw)} unique poses  |  "
      f"kiss-icp t=[{pose_timestamps[0]:.2f}, {pose_timestamps[-1]:.2f}]")

# =============================================================================
# STEP 2 — Auto-calibrate: map MCAP Unix seconds → KISS-ICP timestamp space
# =============================================================================

_KISS_ICP_TOPIC = "/rslidar/M1P_deskewed"
_calib_mcap = []
print(f"\nCalibrating timestamps using first frames of {_KISS_ICP_TOPIC} ...")
with open(DATASET_PATH, "rb") as f:
    _reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for _, channel, message, _ in _reader.iter_decoded_messages(topics=[_KISS_ICP_TOPIC]):
        _calib_mcap.append(message.log_time / 1e9)
        if len(_calib_mcap) >= 20:
            break

_n          = min(len(_calib_mcap), len(pose_timestamps))
_mcap_arr   = np.array(_calib_mcap[:_n])
_kiss_arr   = pose_timestamps[:_n]
_scale      = (_kiss_arr[-1] - _kiss_arr[0]) / (_mcap_arr[-1] - _mcap_arr[0])
_t_offset   = _kiss_arr[0] - _scale * _mcap_arr[0]
print(f"  scale={_scale:.6f}  offset={_t_offset:.3f}")

def _to_pose_t(t_unix_s):
    return _scale * t_unix_s + _t_offset

# =============================================================================
# STEP 3 — Build pose interpolator (SLERP rotation + linear translation)
# =============================================================================

_rotations    = Rotation.from_matrix(poses_raw[:, :3, :3])
_translations = poses_raw[:, :3, 3]
_slerp        = Slerp(pose_timestamps, _rotations)

def interpolate_pose(t_unix_s):
    t = float(np.clip(_to_pose_t(t_unix_s), pose_timestamps[0], pose_timestamps[-1]))
    i = int(np.searchsorted(pose_timestamps, t))
    i = np.clip(i, 1, len(pose_timestamps) - 1)
    alpha = (t - pose_timestamps[i - 1]) / (pose_timestamps[i] - pose_timestamps[i - 1])
    trans = (1.0 - alpha) * _translations[i - 1] + alpha * _translations[i]
    T = np.eye(4)
    T[:3, :3] = _slerp(t).as_matrix()
    T[:3,  3] = trans
    return T

# =============================================================================
# STEP 4 — Pre-build sensor extrinsic matrices (4×4)
# =============================================================================

extrinsics = {
    topic: _rpy_to_matrix4(*tf["rotation_rpy_deg"], tf["translation"])
    for topic, tf in SENSOR_TRANSFORMS.items()
}

# =============================================================================
# STEP 5 — Stream all MCAP scans, transform to world frame, batch-downsample
# =============================================================================

accumulated_raw      = []
partial_clouds       = []
scan_count_total     = 0
scan_count_processed = 0

print(f"\nStreaming all scans from {DATASET_PATH.name} ...")
print(f"  EVERY_N_SCANS={EVERY_N_SCANS}  |  BATCH_DOWNSAMPLE_EVERY={BATCH_DOWNSAMPLE_EVERY}  |  "
      f"VOXEL_SIZE_DISPLAY={VOXEL_SIZE_DISPLAY} m\n")

with open(DATASET_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=LIDAR_TOPICS):
        scan_count_total += 1

        if scan_count_total % EVERY_N_SCANS != 0:
            continue

        t   = message.log_time / 1e9
        pts = _ros_pc2_to_numpy(ros_msg)
        if pts is None or len(pts) == 0:
            continue

        T_ext   = extrinsics[channel.topic]
        T_pose  = interpolate_pose(t)
        T_total = T_pose @ T_ext

        pts_h     = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        pts_world = (T_total @ pts_h.T).T[:, :3]

        accumulated_raw.append(pts_world)
        scan_count_processed += 1

        if scan_count_total % 100 == 0:
            print(f"  scans seen: {scan_count_total:>5}  |  processed: {scan_count_processed:>5}  |  "
                  f"partial clouds: {len(partial_clouds)}")

        if scan_count_processed % BATCH_DOWNSAMPLE_EVERY == 0:
            batch = np.vstack(accumulated_raw).astype(np.float64)
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(batch)
            pcd_ds = pcd_tmp.voxel_down_sample(VOXEL_SIZE_DISPLAY)
            ds_pts = np.asarray(pcd_ds.points)
            partial_clouds.append(ds_pts)
            accumulated_raw.clear()
            print(f"  [batch flush]  {len(batch):>9,} pts  →  {len(ds_pts):>7,} after downsample")

# Flush final remainder
if accumulated_raw:
    batch = np.vstack(accumulated_raw).astype(np.float64)
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(batch)
    pcd_ds = pcd_tmp.voxel_down_sample(VOXEL_SIZE_DISPLAY)
    ds_pts = np.asarray(pcd_ds.points)
    partial_clouds.append(ds_pts)
    accumulated_raw.clear()
    print(f"  [final flush]  {len(batch):>9,} pts  →  {len(ds_pts):>7,} after downsample")

print(f"\nTotal scans seen: {scan_count_total}  |  processed: {scan_count_processed}")

if not partial_clouds:
    print("\nNo points accumulated — check DATASET_PATH and LIDAR_TOPICS.")
    raise SystemExit(1)

# =============================================================================
# STEP 6 — Build trajectory (line + sphere markers at each pose)
# =============================================================================

TRAJ_SPHERE_RADIUS = 0.3   # metres — size of the markers along the path

traj_positions = poses_raw[:, :3, 3]   # (N, 3) — already sorted from Step 1
traj_lines     = [[i, i + 1] for i in range(len(traj_positions) - 1)]

traj_lineset = o3d.geometry.LineSet()
traj_lineset.points = o3d.utility.Vector3dVector(traj_positions)
traj_lineset.lines  = o3d.utility.Vector2iVector(np.array(traj_lines))
traj_lineset.paint_uniform_color([1.0, 0.0, 0.0])   # red

# Sphere markers make the path visible from any zoom level (LineSet is only 1 px wide)
traj_mesh = o3d.geometry.TriangleMesh()
for pos in traj_positions:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=TRAJ_SPHERE_RADIUS, resolution=8)
    sphere.translate(pos)
    traj_mesh += sphere
traj_mesh.paint_uniform_color([1.0, 0.0, 0.0])
traj_mesh.compute_vertex_normals()

# =============================================================================
# STEP 7 — Merge, final downsample, Z-height colouring, visualise
# =============================================================================

all_pts = np.vstack(partial_clouds)

# One final global downsample to remove duplicate voxels at seam boundaries
pcd_final = o3d.geometry.PointCloud()
pcd_final.points = o3d.utility.Vector3dVector(all_pts)
pcd_final = pcd_final.voxel_down_sample(VOXEL_SIZE_DISPLAY)
all_pts = np.asarray(pcd_final.points)

print(f"Final map: {len(all_pts):,} points  (voxel={VOXEL_SIZE_DISPLAY} m)")

# Keep only points within CORRIDOR_RADIUS metres of the driven path
traj_tree = cKDTree(traj_positions)
dists, _  = traj_tree.query(all_pts, workers=-1)
mask      = dists <= CORRIDOR_RADIUS
all_pts   = all_pts[mask]
pcd_final.points = o3d.utility.Vector3dVector(all_pts)
print(f"After {CORRIDOR_RADIUS} m corridor filter: {len(all_pts):,} points")

# Z-height gradient — clip at 2nd/98th percentile to suppress outliers
z      = all_pts[:, 2]
z_lo   = np.percentile(z, 2)
z_hi   = np.percentile(z, 98)
z_norm = np.clip((z - z_lo) / (z_hi - z_lo + 1e-9), 0.0, 1.0)
colors = plt.get_cmap("viridis")(z_norm)[:, :3]   # (N, 3) RGB, drop alpha

pcd_final.colors = o3d.utility.Vector3dVector(colors)

window_title = (f"Full Map — {DATASET_PATH.stem}  "
                f"[{scan_count_processed} scans | {len(all_pts):,} pts | "
                f"voxel={VOXEL_SIZE_DISPLAY} m]")

if SAVE_MAP:
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # KITTI-style binary: flat float32 array of shape (N, 4) — x, y, z, intensity (0)
    pts_xyz = np.asarray(pcd_final.points, dtype=np.float32)
    intensity = np.zeros((len(pts_xyz), 1), dtype=np.float32)
    np.hstack([pts_xyz, intensity]).tofile(str(SAVE_PATH))
    print(f"Map saved to {SAVE_PATH}  ({len(pts_xyz):,} points)")

print(f"\nOpening Open3D viewer …")
print(f"  {window_title}")

o3d.visualization.draw_geometries(
    [pcd_final, traj_lineset, traj_mesh],
    window_name=window_title,
    width=1280,
    height=720,
    point_show_normal=False,
)
