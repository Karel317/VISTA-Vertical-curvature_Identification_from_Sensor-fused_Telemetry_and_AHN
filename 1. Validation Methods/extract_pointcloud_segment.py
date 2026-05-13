# TODO
# Not finished, it does work but its so many points and it saves a 1.5gb file in the repo
# I need to downsample the point cloud, maybe compress it too
#
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


# =============================================================================
# SETTINGS
# =============================================================================

#DATASET_PATH = Path(r"C:\Users\aitha\OneDrive\Bureaublad\Systeem en regeltechnieken\lidar-ground-segmentation\1. Validation Methods\KISS ICP\KISS ICP results\2026_04_29 15_20_00\poses.npy")
#OUTPUT_PATH = Path(r"C:\ROSBAGS VERWIJDER NA BEP\output")

DATASET_PATH = Path(r"C:\ROSBAGS VERWIJDER NA BEP\29 april\2026_04_29\15_20_00\rosbag\rosbag_0.mcap")
RESULTS_DIR  = Path(r"C:\Users\aitha\OneDrive\Bureaublad\Systeem en regeltechnieken\lidar-ground-segmentation\1. Validation Methods\KISS ICP\KISS ICP results\2026_04_29 15_20_00")
OUTPUT_DIR   = Path(r"C:\ROSBAGS VERWIJDER NA BEP\output")

T_START     = 1777468956.122513909   # paste begin timestamp from Foxglove
T_END       = 1777468976.117642921   # paste end timestamp from Foxglove
TIME_MARGIN = 10.0                   # seconds before/after to catch approach/departure points

LIDAR_TOPICS = [
    "/rslidar/M1P_deskewed",
    "/rslidar/helios_R",
    "/rslidar/helios_L",
]

# =============================================================================
# SENSOR EXTRINSICS  (sensor → base_link, from Foxglove TF tree)
# =============================================================================

SENSOR_TRANSFORMS = {
    "/rslidar/M1P_deskewed": {
        "translation":      [0.800,  0.000, 0.848],
        "rotation_rpy_deg": [0.1, -1.7,  1.7],
    },
    "/rslidar/helios_R": {
        "translation":      [-0.743, -0.243, 0.857],
        "rotation_rpy_deg": [2.4, -2.6, -179.9],
    },
    "/rslidar/helios_L": {
        "translation":      [-0.745,  0.191, 0.876],
        "rotation_rpy_deg": [3.5, -2.1,  179.9],
    },
}

# =============================================================================
# RESULTS DIRECTORY  (same naming as run_kiss_icp.py)
# =============================================================================

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
#
# KISS-ICP timestamps may use a different epoch or unit than Unix seconds.
# We read the first 20 M1P frames from the MCAP, compare their log_time values
# (Unix nanoseconds → seconds) against the matching KISS-ICP timestamps, and
# fit a linear mapping:  kiss_t = scale * mcap_unix_s + offset
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
print(f"  (T_START={T_START:.3f} maps to kiss_t={_scale*T_START + _t_offset:.3f})")

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
# STEP 5 — Stream MCAP, filter by time window, transform points to global frame
# =============================================================================

t_lo = T_START - TIME_MARGIN
t_hi = T_END   + TIME_MARGIN

print(f"\nExtracting scans in [{t_lo:.2f}, {t_hi:.2f}] s  "
      f"(window {T_END - T_START:.1f} s + {TIME_MARGIN:.0f} s margin each side)")

all_points  = []
scan_counts = {topic: 0 for topic in LIDAR_TOPICS}

with open(DATASET_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=LIDAR_TOPICS):
        t = message.log_time / 1e9   # nanoseconds → seconds
        if not (t_lo <= t <= t_hi):
            continue

        pts = _ros_pc2_to_numpy(ros_msg)
        if pts is None or len(pts) == 0:
            continue

        T_ext  = extrinsics[channel.topic]   # sensor → base_link
        T_pose = interpolate_pose(t)          # base_link → world

        T_total = T_pose @ T_ext
        pts_h   = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        pts_world = (T_total @ pts_h.T).T[:, :3]

        all_points.append(pts_world)
        scan_counts[channel.topic] += 1

print("\nScans used per sensor:")
for topic, count in scan_counts.items():
    print(f"  {topic:<35}  {count} scans")

if not all_points:
    print("\nNo points found — check T_START / T_END against the bag timestamps.")
else:
    cloud = np.vstack(all_points)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))
    o3d.visualization.draw_geometries([pcd], window_name="Merged Point Cloud")
    print(f"\nTotal points: {len(cloud):,}")
    # ==========================================================================
    # STEP 6 — Save mcap
    # ==========================================================================
from mcap.writer import Writer
from mcap_ros2.writer import Writer as Ros2Writer
import struct
print([m for m in dir(Ros2Writer) if not m.startswith("_")])
# ... (keep everything above Step 6 the same)

if not all_points:
    print("\nNo points found — check T_START / T_END against the bag timestamps.")
else:
    from mcap_ros2.writer import Writer as Ros2Writer
    import struct

    cloud = np.vstack(all_points)
    print(f"\nTotal points: {len(cloud):,}")

    mcap_out = OUTPUT_DIR / f"segment_{T_START:.0f}_{T_END:.0f}.mcap"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def make_pointcloud2_bytes(points_xyz, stamp_sec, stamp_nsec):
        n          = len(points_xyz)
        point_step = 12
        row_step   = n * point_step
        data       = points_xyz.astype(np.float32).tobytes()

        header       = struct.pack("<I", 0x00000001)
        ros_header   = struct.pack("<II", stamp_sec, stamp_nsec)
        frame_id     = b"map\x00"
        frame_id_len = struct.pack("<I", len(frame_id))

        def pack_field(name, offset, datatype, count):
            name_b = name.encode() + b"\x00"
            return struct.pack("<I", len(name_b)) + name_b + struct.pack("<IBI", offset, datatype, count)

        fields       = pack_field("x", 0, 7, 1) + pack_field("y", 4, 7, 1) + pack_field("z", 8, 7, 1)
        fields_count = struct.pack("<I", 3)
        meta         = struct.pack("<??IIIi", False, False, 1, n, point_step, row_step)
        data_len     = struct.pack("<I", len(data))

        return header + ros_header + frame_id_len + frame_id + fields_count + fields + meta + data_len + data

    stamp_sec  = int(T_START)
    stamp_nsec = int((T_START % 1) * 1e9)
    msg_bytes  = make_pointcloud2_bytes(cloud, stamp_sec, stamp_nsec)

    with open(mcap_out, "wb") as f:
        with Ros2Writer(output=f) as writer:
            schema = writer.register_msgdef("sensor_msgs/msg/PointCloud2", "")
            writer.write_message(
                topic="/merged_cloud",
                schema=schema,
                message=msg_bytes,
                log_time=int(T_START * 1e9),
                publish_time=int(T_START * 1e9),
            )

    print(f"MCAP saved to {mcap_out}")
