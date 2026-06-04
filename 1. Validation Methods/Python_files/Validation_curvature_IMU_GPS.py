"""IMU + GPS curvature validation (EKF speed fusion).

Modular port of the `Validation_curvature_IMU_GPS` notebook, written in the same
style as `AHN5_validation.py` (a SETTINGS block, `_`-prefixed helper functions, a
`run()` orchestrator and an argparse `__main__`) so it can be imported and driven
from a larger pipeline.

Computes, in a window around a reference timestamp:
    kappa_terrain_spline : Frenet curvature on a spline fit of the elevation profile
    kappa_terrain_pitch  : dalpha/ds, pitch rate along the path
    kappa_path           : omega_z / v, horizontal (turning) path curvature
Speed v is fused from ZED pose differentiation and GPS differentiation through a
1-D Kalman filter.

The IMU input may be either an MCAP file (read directly) or a ZED `.svo`/`.svo2`
recording. If it is an SVO, it is first converted to a temporary MCAP (only a
slice around TIME is converted) which is deleted once the run finishes. The GPS
input is always an MCAP (rosbag).

Usage (CLI):
    python Validation_curvature_IMU_GPS.py \
        --imu-file recording.svo2 --gps-file rosbag_0.mcap --plot True
"""
from __future__ import annotations

import os
import sys
import argparse
import math
import struct
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.interpolate as sc
from pyproj import Transformer
from filterpy.kalman import KalmanFilter
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap.writer import Writer as McapWriter

# Shared cutting/loading helpers. Add this file's folder to sys.path so the import
# resolves regardless of how the module is loaded (run_validations adds it already;
# pipeline_validation loads modules by explicit path and does not).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcap_utils import cut_mcap, load_gps

# ZED SDK (pyzed) is imported lazily inside _convert_svo_to_mcap, because it is
# only needed when the IMU input is an SVO file. Running on an MCAP requires no
# ZED SDK installation.
sl = None

# ============================================================
# SETTINGS — edit these before running
# ============================================================

IMU_INPUT_PATH = r"D:\Data_gathered\2026_05_22\Camera\10_50_00\back_22_05_2026-10_50_00.mcap"  # .mcap, .svo or .svo2
GPS_FILE_PATH  = r"D:\Data_gathered\2026_05_22\Rosbag\10_50_00\rosbag\rosbag_0.mcap"
OUTPUT_PATH    = r"D:\Validation_results"

TIME           = 1779439917.972930459          # reference Unix timestamp (same as the other validation files)

# Topics
POSE_TOPIC = "/zed/pose"                        # position (x, y, z) + orientation quaternion -> pitch
IMU_TOPIC  = "/zed/imu/data"                    # angular velocity z (yaw rate omega_z)
GPS_TOPIC  = "/navsat_topic"

# Profile window
WIN_BEFORE = 5.0                                # m before reference point to include
WIN_AFTER  = 20.0                               # m after reference point to include
DELTA_M    = 0.1                                # currently unused (reserved for fixed-interval resampling)
MIN_SPEED  = 0.5                                # m/s — below this kappa_path is set to NaN
SMOOTH_W   = 21                                 # Savitzky-Golay window for pitch smoothing (0 disables)
GPS_WINDOW = 30.0                               # s either side of TIME to read from the GPS MCAP

# Spline fit for the elevation profile
SPLINE_SMOOTH = 0.0                             # spline smoothing factor (0 = interpolating fit)

# EKF speed-fusion tuning (1-D Kalman filter; state = [v, a])
EKF_P0_V       = 4.0                            # initial variance on speed (m/s)^2
EKF_P0_A       = 1.0                            # initial variance on acceleration (m/s^2)^2
EKF_R_ZED      = 0.25                           # ZED speed measurement noise variance (m/s)^2
EKF_R_GPS      = 1.0                            # GPS speed measurement noise variance (m/s)^2
EKF_SIGMA_JERK = 0.5                            # process (jerk) noise (m/s^3)
GPS_MATCH_DT   = 0.12                           # s — accept a GPS speed update within this dt of a ZED sample

# SVO -> MCAP conversion (only used when the IMU input is an .svo/.svo2)
SVO_WINDOW_S = 30.0                             # +/- s around TIME to convert (keep >= GPS_WINDOW and wide enough for WIN_AFTER m)

# MCAP pre-cut: before reading, an MCAP input is sliced to a small window around
# TIME and cached to a temp file, so the first run pays the scan cost once and
# later runs read only the slice. Keep >= GPS_WINDOW and wide enough for WIN_AFTER m.
CUT_WINDOW_S = 30.0                             # +/- s around TIME to keep in the cut MCAP


# ── ROS2 message definitions (used when converting an SVO to MCAP) ───────────

IMU_MSGDEF = """\
std_msgs/Header header
geometry_msgs/Quaternion orientation
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
uint32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Quaternion
float64 x 0
float64 y 0
float64 z 0
float64 w 1
================================================================================
MSG: geometry_msgs/Vector3
float64 x 0
float64 y 0
float64 z 0
"""

POSE_MSGDEF = """\
std_msgs/Header header
geometry_msgs/Pose pose
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
uint32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x 0
float64 y 0
float64 z 0
================================================================================
MSG: geometry_msgs/Quaternion
float64 x 0
float64 y 0
float64 z 0
float64 w 1
"""


# ── CDR encoder (SVO -> MCAP) ─────────────────────────────────────────────────

class CDREncoder:
    """Minimal CDR (little-endian) encoder for ROS2 messages.

    Tracks the current buffer position and inserts alignment padding so that
    each field starts at its natural alignment boundary.
    """

    def __init__(self):
        self._buf = bytearray()

    def _align(self, n: int) -> None:
        pad = (-len(self._buf)) % n
        self._buf += b'\x00' * pad

    def uint32(self, v: int) -> None:
        self._align(4)
        self._buf += struct.pack('<I', v)

    def float64(self, v: float) -> None:
        self._align(8)
        self._buf += struct.pack('<d', v)

    def float64_array(self, values) -> None:
        for v in values:
            self.float64(v)

    def string(self, s: str) -> None:
        b = s.encode('utf-8') + b'\x00'
        self._align(4)
        self._buf += struct.pack('<I', len(b))
        self._buf += b

    def header(self, sec: int, nsec: int, frame_id: str) -> None:
        self.uint32(sec)
        self.uint32(nsec)
        self.string(frame_id)

    def to_bytes(self) -> bytes:
        # CDR encapsulation identifier: little-endian
        return b'\x00\x01\x00\x00' + bytes(self._buf)


def _encode_imu(
    sec: int, nsec: int, frame_id: str,
    qx: float, qy: float, qz: float, qw: float,
    gx: float, gy: float, gz: float,   # rad/s
    ax: float, ay: float, az: float,   # m/s^2
) -> bytes:
    e = CDREncoder()
    e.header(sec, nsec, frame_id)
    e.float64(qx); e.float64(qy); e.float64(qz); e.float64(qw)
    e.float64_array([0.0] * 9)   # orientation_covariance (unknown -> zeros)
    e.float64(gx); e.float64(gy); e.float64(gz)
    e.float64_array([0.0] * 9)   # angular_velocity_covariance
    e.float64(ax); e.float64(ay); e.float64(az)
    e.float64_array([0.0] * 9)   # linear_acceleration_covariance
    return e.to_bytes()


def _encode_pose_stamped(
    sec: int, nsec: int, frame_id: str,
    tx: float, ty: float, tz: float,
    qx: float, qy: float, qz: float, qw: float,
) -> bytes:
    e = CDREncoder()
    e.header(sec, nsec, frame_id)
    e.float64(tx); e.float64(ty); e.float64(tz)
    e.float64(qx); e.float64(qy); e.float64(qz); e.float64(qw)
    return e.to_bytes()


def _svo_frame_at_ts(zed: "sl.Camera", sensors_data: "sl.SensorsData",
                     target_ns: int, total_frames: int) -> int:
    """Binary-search the SVO for the first frame whose IMU timestamp >= target_ns."""
    lo, hi = 0, total_frames - 1
    while lo < hi:
        mid = (lo + hi) // 2
        zed.set_svo_position(mid)
        zed.grab()
        zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
        ts = sensors_data.get_imu_data().timestamp.get_nanoseconds()
        if ts < target_ns:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ── Helper functions ─────────────────────────────────────────────────────────

def _is_svo(path) -> bool:
    """True if the path looks like a ZED SVO recording (.svo / .svo2)."""
    return str(path).lower().endswith((".svo", ".svo2"))


def _convert_svo_to_mcap(svo_path, out_mcap_path, timestamp):
    """Convert a ZED SVO recording to MCAP (positional tracking + IMU).

    Only the [timestamp - SVO_WINDOW_S, timestamp + SVO_WINDOW_S] slice is
    written, as sensor_msgs/Imu on /zed/imu/data and geometry_msgs/PoseStamped
    on /zed/pose — exactly the topics the validation reads.
    """
    global sl
    try:
        import pyzed.sl as sl  # noqa: F401  (populates the module-level `sl`)
    except ImportError as exc:
        raise RuntimeError(
            "The IMU input is an SVO file, which requires the ZED SDK Python API "
            "(pyzed) to convert to MCAP. Install the ZED SDK and `pyzed`, or pass "
            "an already-converted .mcap file as the IMU input."
        ) from exc

    if not os.path.exists(svo_path):
        raise FileNotFoundError(f"SVO file not found: {svo_path}")

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP   # ROS2 REP-103
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS                        # GEN 3
    init_params.svo_real_time_mode = False                                   # process as fast as possible

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    pose = sl.Pose()
    sensors_data = sl.SensorsData()
    translation = sl.Translation()
    orientation = sl.Orientation()

    total_frames = zed.get_svo_number_of_frames()

    start_ns = int((timestamp - SVO_WINDOW_S) * 1_000_000_000)
    end_ns   = int((timestamp + SVO_WINDOW_S) * 1_000_000_000)

    # Seek BEFORE enabling tracking — set_svo_position is unreliable while tracking is active
    start_frame = _svo_frame_at_ts(zed, sensors_data, start_ns, total_frames)
    zed.set_svo_position(start_frame)

    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    frame = 0
    try:
        with open(out_mcap_path, "wb") as f:
            writer = McapWriter(f)
            writer.start(profile="ros2", library="zed_svo_to_mcap")

            imu_schema_id = writer.register_schema(
                name="sensor_msgs/Imu", encoding="ros2msg", data=IMU_MSGDEF.encode())
            pose_schema_id = writer.register_schema(
                name="geometry_msgs/PoseStamped", encoding="ros2msg", data=POSE_MSGDEF.encode())
            imu_ch = writer.register_channel(
                topic=IMU_TOPIC, message_encoding="cdr", schema_id=imu_schema_id)
            pose_ch = writer.register_channel(
                topic=POSE_TOPIC, message_encoding="cdr", schema_id=pose_schema_id)

            while True:
                err = zed.grab()
                if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                if err != sl.ERROR_CODE.SUCCESS:
                    continue

                tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                imu_data = sensors_data.get_imu_data()

                ts_ns = imu_data.timestamp.get_nanoseconds()
                if ts_ns > end_ns:
                    break
                sec = ts_ns // 1_000_000_000
                nsec = ts_ns % 1_000_000_000

                # Orientation: use camera pose result; fall back to identity when lost
                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    pose.get_orientation(orientation)
                    q = orientation.get()
                    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                else:
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

                # Angular velocity: ZED returns deg/s -> convert to rad/s for ROS2
                av = imu_data.get_angular_velocity()
                la = imu_data.get_linear_acceleration()

                imu_bytes = _encode_imu(
                    sec, nsec, "imu",
                    qx, qy, qz, qw,
                    math.radians(float(av[0])),
                    math.radians(float(av[1])),
                    math.radians(float(av[2])),
                    float(la[0]), float(la[1]), float(la[2]),
                )
                writer.add_message(channel_id=imu_ch, log_time=ts_ns,
                                   data=imu_bytes, publish_time=ts_ns)

                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    pose.get_translation(translation)
                    t = translation.get()
                    pose_bytes = _encode_pose_stamped(
                        sec, nsec, "world",
                        float(t[0]), float(t[1]), float(t[2]),
                        qx, qy, qz, qw,
                    )
                    writer.add_message(channel_id=pose_ch, log_time=ts_ns,
                                       data=pose_bytes, publish_time=ts_ns)

                frame += 1

            writer.finish()
    finally:
        zed.disable_positional_tracking()
        zed.close()

    return out_mcap_path


def _svo_to_mcap_cached(svo_path, timestamp, window_s):
    """Convert an SVO to a windowed MCAP once, caching the result on disk.

    Same caching idea as `_cut_mcap`: the ZED-SDK conversion (the slow part) runs
    only on the first call; later calls with the same SVO/timestamp/window reuse
    the cached MCAP in the system temp dir. The converted MCAP already contains
    only the pose/IMU topics over [TIME ± window_s], so no further cut is needed.

    Returns the path to the cached MCAP.
    """
    svo_path = str(svo_path)
    if not os.path.exists(svo_path):
        raise FileNotFoundError(f"SVO file not found: {svo_path}")

    cache_dir = os.path.join(tempfile.gettempdir(), "mcap_cut_cache")
    os.makedirs(cache_dir, exist_ok=True)

    st = os.stat(svo_path)
    stem = os.path.splitext(os.path.basename(svo_path))[0]
    out_path = os.path.join(
        cache_dir,
        f"{stem}_svo_{int(timestamp)}_{int(window_s)}s_{st.st_size}_{int(st.st_mtime)}.mcap",
    )

    if os.path.exists(out_path):
        print(f"Using cached SVO->MCAP: {out_path}")
        return out_path

    print(f"Converting SVO to MCAP (+/-{window_s:.0f}s around TIME; first run, then cached)...")
    tmp_path = out_path + ".tmp"
    _convert_svo_to_mcap(svo_path, tmp_path, timestamp)
    os.replace(tmp_path, out_path)   # atomic: a partial conversion never looks cached
    print(f"SVO->MCAP cached: {out_path}")
    return out_path


def _quaternion_to_pitch(qx, qy, qz, qw):
    qx, qy, qz, qw = np.array(qx), np.array(qy), np.array(qz), np.array(qw)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1, 1)
    return np.degrees(np.arcsin(sinp))


def _read_imu(imu_mcap, data):
    """Read /zed/pose and /zed/imu/data from the IMU MCAP."""
    pose_data = {"t": [], "x": [], "y": [], "z": [], "qx": [], "qy": [], "qz": [], "qw": []}
    imu_raw   = {"t": [], "omega_z": []}

    with open(imu_mcap, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=[POSE_TOPIC, IMU_TOPIC]
        ):
            if channel.topic == POSE_TOPIC:
                t = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec / 1e9
                pose_data["t"].append(t)
                pose_data["x"].append(ros_msg.pose.position.x)
                pose_data["y"].append(ros_msg.pose.position.y)
                pose_data["z"].append(ros_msg.pose.position.z)
                pose_data["qx"].append(ros_msg.pose.orientation.x)
                pose_data["qy"].append(ros_msg.pose.orientation.y)
                pose_data["qz"].append(ros_msg.pose.orientation.z)
                pose_data["qw"].append(ros_msg.pose.orientation.w)
            elif channel.topic == IMU_TOPIC:
                t = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec / 1e9
                imu_raw["t"].append(t)
                imu_raw["omega_z"].append(ros_msg.angular_velocity.z)

    if len(pose_data["t"]) == 0:
        raise RuntimeError(
            f"No messages on pose topic '{POSE_TOPIC}' in {imu_mcap}. "
            "Check POSE_TOPIC / IMU_TOPIC.")
    if len(imu_raw["t"]) == 0:
        raise RuntimeError(
            f"No messages on IMU topic '{IMU_TOPIC}' in {imu_mcap}. "
            "Check IMU_TOPIC.")

    data["pose_data"] = pose_data
    data["imu_raw"]   = imu_raw
    return data


def _read_gps(gps_file, data, timestamp, gps_topic):
    """Load the GPS slice (shared cut + decode) and project it to RD New."""
    # Shared with AHN5_validation via mcap_utils.load_gps: the bag is cut + decoded
    # once per run and reused. GPS_WINDOW is the slice half-width.
    gps_raw = load_gps(gps_file, gps_topic, timestamp, GPS_WINDOW)

    # Convert to Dutch RD New (EPSG:28992)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    rd_xs, rd_ys = transformer.transform(gps_raw["lon"], gps_raw["lat"])
    rd_xs = np.array(rd_xs)
    rd_ys = np.array(rd_ys)
    gps_t = np.array(gps_raw["t"])

    # Cumulative arc-length along the GPS track
    gps_s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rd_xs), np.diff(rd_ys)))])
    print(f"GPS track length: {gps_s[-1]:.1f} m")

    data["gps_raw"] = gps_raw
    data["rd_xs"] = rd_xs
    data["rd_ys"] = rd_ys
    data["gps_t"] = gps_t
    data["gps_s"] = gps_s
    return data


def _fuse_speed(data):
    """1-D Kalman filter fusing ZED pose speed and GPS speed (state = [v, a])."""
    pose_data = data["pose_data"]
    pose_t = np.array(pose_data["t"])
    pose_x = np.array(pose_data["x"])
    pose_y = np.array(pose_data["y"])
    if len(pose_t) < 2:
        raise RuntimeError("Need at least 2 pose samples for speed fusion.")

    # ZED scalar speed: |dpos| / dt
    pose_dt   = np.diff(pose_t)
    pose_dist = np.hypot(np.diff(pose_x), np.diff(pose_y))
    v_zed_raw = np.where(pose_dt > 1e-6, pose_dist / pose_dt, 0.0)
    v_zed_t   = 0.5 * (pose_t[:-1] + pose_t[1:])          # midpoint timestamps

    # GPS scalar speed: |ddist| / dt
    gps_t = data["gps_t"]
    gps_s = data["gps_s"]
    gps_dt   = np.diff(gps_t)
    gps_dist = np.diff(gps_s)
    v_gps_raw = np.where(gps_dt > 1e-6, gps_dist / gps_dt, 0.0)
    v_gps_t   = 0.5 * (gps_t[:-1] + gps_t[1:])

    # Run on the ZED time axis (higher rate); GPS is folded in as a second measurement.
    kf = KalmanFilter(dim_x=2, dim_z=1)
    dt0 = float(np.median(pose_dt))

    kf.x = np.array([[float(v_zed_raw[0])], [0.0]])   # [v, a]
    kf.P = np.diag([EKF_P0_V, EKF_P0_A])              # initial covariance
    kf.H = np.array([[1.0, 0.0]])                     # observe speed only
    kf.Q = np.array([
        [dt0 ** 4 / 4 * EKF_SIGMA_JERK ** 2, dt0 ** 3 / 2 * EKF_SIGMA_JERK ** 2],
        [dt0 ** 3 / 2 * EKF_SIGMA_JERK ** 2, dt0 ** 2      * EKF_SIGMA_JERK ** 2],
    ])

    R_zed = np.array([[EKF_R_ZED]])
    R_gps = np.array([[EKF_R_GPS]])
    v_fused = np.zeros(len(v_zed_t))

    for i in range(len(v_zed_t)):
        dt = float(pose_dt[i]) if pose_dt[i] > 1e-6 else dt0
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        kf.predict()

        # Primary update: ZED speed
        kf.R = R_zed
        kf.update(np.array([[v_zed_raw[i]]]))

        # Secondary update: GPS speed when a GPS sample is close in time.
        # searchsorted gives an insertion index; the nearest sample may be at
        # that index or the one before it, so check both neighbours.
        ins = np.searchsorted(v_gps_t, v_zed_t[i])
        candidates = [j for j in (ins - 1, ins) if 0 <= j < len(v_gps_t)]
        if candidates:
            idx_gps = min(candidates, key=lambda j: abs(v_gps_t[j] - v_zed_t[i]))
            if abs(v_gps_t[idx_gps] - v_zed_t[i]) < GPS_MATCH_DT:
                kf.R = R_gps
                kf.update(np.array([[v_gps_raw[idx_gps]]]))

        v_fused[i] = max(kf.x[0, 0], 0.0)   # speed is non-negative

    # Cumulative arc-length on the ZED pose (distance axis)
    pose_s = np.concatenate([[0.0], np.cumsum(pose_dist)])
    v_fused_at_pose = np.interp(pose_t, v_zed_t, v_fused)

    data.update(dict(
        pose_t=pose_t, pose_x=pose_x, pose_y=pose_y,
        pose_dist=pose_dist, pose_s=pose_s,
        v_zed_t=v_zed_t, v_zed_raw=v_zed_raw,
        v_gps_t=v_gps_t, v_gps_raw=v_gps_raw,
        v_fused=v_fused, v_fused_at_pose=v_fused_at_pose,
    ))
    return data


def _elevation_profile(data, timestamp):
    """Pitch from the pose quaternion, integrated over distance to an elevation profile."""
    pose_data = data["pose_data"]
    pose_t    = data["pose_t"]
    pose_s    = data["pose_s"]
    pose_dist = data["pose_dist"]

    pitch_raw  = _quaternion_to_pitch(pose_data["qx"], pose_data["qy"],
                                      pose_data["qz"], pose_data["qw"])
    pitch_bias = float(np.mean(pitch_raw))
    pitch = pitch_raw - pitch_bias

    if SMOOTH_W > 0:
        pitch = savgol_filter(pitch, window_length=SMOOTH_W, polyorder=3)

    # ds along the ZED pose path, elevation by integrating pitch over distance
    ds = np.concatenate([[0.0], pose_dist])
    imu_elevation = np.cumsum(ds * np.sin(np.radians(pitch)))

    # Window around the reference time
    ref_idx = int(np.argmin(np.abs(pose_t - timestamp)))
    dt_err  = abs(pose_t[ref_idx] - timestamp)

    s_rel = pose_s - pose_s[ref_idx]
    mask  = (s_rel >= -WIN_BEFORE) & (s_rel <= WIN_AFTER)
    if not mask.any():
        raise RuntimeError("Profile window is empty — check TIME / WIN_BEFORE / WIN_AFTER.")

    s_win         = s_rel[mask]
    z_win         = imu_elevation[mask] - imu_elevation[mask][0]
    pitch_win     = pitch[mask]
    pitch_rad_win = np.radians(pitch_win)

    data.update(dict(
        pitch=pitch, imu_elevation=imu_elevation,
        ref_idx=ref_idx, dt_err=dt_err, s_rel=s_rel, mask=mask,
        s_win=s_win, z_win=z_win, pitch_win=pitch_win, pitch_rad_win=pitch_rad_win,
    ))
    return data


def _terrain_curvature(data):
    """Vertical terrain curvature: spline-Frenet (method A) and pitch differentiation (method B)."""
    s_win         = data["s_win"]
    z_win         = data["z_win"]
    pitch_rad_win = data["pitch_rad_win"]

    # Method A: Frenet curvature on a spline fit
    sp_z   = sc.make_splrep(s_win, z_win, s=SPLINE_SMOOTH)
    dzds   = sp_z.derivative(nu=1)(s_win)
    d2zds2 = sp_z.derivative(nu=2)(s_win)
    kappa_terrain_spline = np.abs(d2zds2) / (1.0 + dzds ** 2) ** 1.5

    # Method B: direct pitch differentiation (valid for small pitch angles)
    kappa_terrain_pitch = np.gradient(pitch_rad_win, s_win)

    data.update(dict(
        spline_z=sp_z,
        kappa_terrain_spline=kappa_terrain_spline,
        kappa_terrain_pitch=kappa_terrain_pitch,
    ))
    return data


def _path_curvature(data):
    """Horizontal path curvature kappa_path = omega_z / v (NaN below MIN_SPEED)."""
    pose_t          = data["pose_t"]
    imu_raw         = data["imu_raw"]
    mask            = data["mask"]
    v_fused_at_pose = data["v_fused_at_pose"]

    omega_z_all = np.interp(pose_t, np.array(imu_raw["t"]), np.array(imu_raw["omega_z"]))
    omega_z_win = omega_z_all[mask]
    speed_win   = v_fused_at_pose[mask]

    # Divide only where speed is above MIN_SPEED; elsewhere leave NaN. Using
    # np.divide with `where` avoids evaluating omega_z / 0 (np.where would still
    # compute the division for every element and raise a divide-by-zero warning).
    fast = speed_win > MIN_SPEED
    kappa_path = np.full_like(speed_win, np.nan, dtype=float)
    np.divide(omega_z_win, speed_win, out=kappa_path, where=fast)

    data.update(dict(omega_z_win=omega_z_win, speed_win=speed_win, kappa_path=kappa_path))
    return data


def _positional_tracking_profile(data, timestamp):
    """Elevation straight from the ZED pose z, on the same window as the pitch profile."""
    pose_data = data["pose_data"]
    pose_z    = np.array(pose_data["z"])
    mask      = data["mask"]
    ref_idx   = data["ref_idx"]

    z_pos_win = pose_z[mask] - pose_z[ref_idx]
    data["z_pos_win"] = z_pos_win
    return data


def _plot(data):
    v_zed_t = data["v_zed_t"]; v_zed_raw = data["v_zed_raw"]
    v_gps_t = data["v_gps_t"]; v_gps_raw = data["v_gps_raw"]
    v_fused = data["v_fused"]
    s_win = data["s_win"]; z_win = data["z_win"]; z_pos_win = data["z_pos_win"]
    kappa_terrain_spline = data["kappa_terrain_spline"]
    kappa_terrain_pitch  = data["kappa_terrain_pitch"]
    kappa_path           = data["kappa_path"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    ax_v, ax_z, ax_kv = axes

    # --- Speed ---
    if len(v_zed_t):
        ax_v.plot(v_zed_t - v_zed_t[0], v_zed_raw, label="ZED (raw)",
                  color="steelblue", alpha=0.5, linewidth=1)
    if len(v_gps_t):
        ax_v.plot(v_gps_t - v_gps_t[0], v_gps_raw, label="GPS (raw)",
                  color="orange", alpha=0.5, linewidth=1)
    if len(v_zed_t):
        ax_v.plot(v_zed_t - v_zed_t[0], v_fused, label="EKF fused",
                  color="black", linewidth=2)
    ax_v.axhline(MIN_SPEED, color="red", linestyle="--", linewidth=1,
                 label=f"MIN_SPEED={MIN_SPEED} m/s")
    ax_v.set_xlabel("Time (s)"); ax_v.set_ylabel("Speed (m/s)")
    ax_v.set_title("EKF Speed Fusion (ZED + GPS)")
    ax_v.legend(); ax_v.grid(True)

    # --- Elevation profile ---
    ax_z.plot(s_win, z_win, label="IMU elevation (pitch integration)", color="blue")
    ax_z.plot(s_win, z_pos_win, label="z from positional tracking", color="green")
    ax_z.set_xlabel("Distance from reference (m)"); ax_z.set_ylabel("Elevation (m, relative)")
    ax_z.set_title("Elevation Profile")
    ax_z.legend(); ax_z.grid(True)

    # --- Vertical terrain curvature (+ path curvature for comparison) ---
    ax_kv.plot(s_win, kappa_terrain_spline, label="Spline method", color="orange")
    ax_kv.plot(s_win, kappa_path, label="omega_z / v  (EKF speed)", color="red")
    ax_kv.plot(s_win, np.abs(kappa_terrain_pitch), label="Pitch diff. method",
               color="purple", linestyle="--")
    ax_kv.set_xlabel("Distance (m)"); ax_kv.set_ylabel("kappa (1/m)")
    ax_kv.set_title("Vertical Terrain Curvature")
    ax_kv.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_kv.legend(); ax_kv.grid(True)

    plt.tight_layout()
    plt.show()


def _save_output(gps_file, data, timestamp, output_location):
    """Save .npz profiles compatible with the rest of the validation pipeline.

    The output folder is derived from the timestamp (datetime.fromtimestamp), the
    same convention as lidar_pipeline_unified and the other validation methods, so
    everything lands in output_location/<date>/<time>/<int(timestamp)>.
    (gps_file is kept for signature compatibility but no longer sets the path.)
    """
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    save_dir = os.path.join(output_location, dt.strftime("%Y_%m_%d"),
                            dt.strftime("%H_%M_%S"), str(int(timestamp)))
    os.makedirs(save_dir, exist_ok=True)

    t_int = int(timestamp)

    # EKF curvature profile
    method = "EKF_curvature_validation"
    fpath = os.path.join(save_dir, f"{method}_{t_int}.npz")
    np.savez_compressed(
        fpath,
        x=np.array([]),
        y=np.array([]),
        z=data["z_win"],
        s=data["s_win"],
        t=np.array([timestamp]),
        method=np.array([method]),
        kappa_terrain_spline=data["kappa_terrain_spline"],
        kappa_terrain_pitch=data["kappa_terrain_pitch"],
        kappa_path=data["kappa_path"],
        gps_lon=np.array(data["gps_raw"]["lon"]),
        gps_lat=np.array(data["gps_raw"]["lat"]),
    )
    print(f"Saved {method} -> {fpath}")

    # Positional-tracking elevation profile
    method = "Z_positional_tracking"
    fpath = os.path.join(save_dir, f"{method}_{t_int}.npz")
    np.savez_compressed(
        fpath,
        x=np.array([]),
        y=np.array([]),
        z=data["z_pos_win"],
        s=data["s_win"],
        t=np.array([timestamp]),
        method=np.array([method]),
    )
    print(f"Saved {method} -> {fpath}")


# ── Main functions ─────────────────────────────────────────────────────────

def run(imu_input, gps_file, timestamp, gps_topic, plot, output_location):
    """Run the full IMU + GPS curvature validation.

    Inputs are cut/converted to small cached MCAP slices FIRST (before any
    reading or computation): an SVO IMU input is converted to a windowed MCAP,
    an MCAP IMU input is cut to its pose/IMU topics, and the GPS MCAP is cut to
    its GPS topic. All three are cached in the system temp dir, so the first run
    pays the cost once and later runs reuse the slices.
    """
    timestamp = float(timestamp)

    # ── Step 1: cut/convert every input to a cached MCAP slice (done first) ────
    if _is_svo(imu_input):
        # SVO: convert (and window) to a cached MCAP via the ZED SDK.
        imu_mcap = _svo_to_mcap_cached(imu_input, timestamp, SVO_WINDOW_S)
    else:
        # MCAP: cut to a window around TIME, keeping only the pose/IMU topics.
        imu_mcap = cut_mcap(imu_input, timestamp, CUT_WINDOW_S,
                            topics=[POSE_TOPIC, IMU_TOPIC])

    # ── Step 2: read the slices and compute ───────────────────────────────────
    # GPS is loaded through the shared mcap_utils.load_gps (cut + decoded once and
    # reused by AHN5_validation in the same run); pass the ORIGINAL gps_file so the
    # cache key matches AHN5's.
    data = {}
    _read_imu(imu_mcap, data)
    _read_gps(gps_file, data, timestamp, gps_topic)
    _fuse_speed(data)
    _elevation_profile(data, timestamp)
    _terrain_curvature(data)
    _path_curvature(data)
    _positional_tracking_profile(data, timestamp)
    _save_output(gps_file, data, timestamp, output_location)
    if plot:
        _plot(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IMU + GPS curvature validation (SVO inputs are auto-converted to MCAP).")
    parser.add_argument("--imu-file", default=IMU_INPUT_PATH,
                        help="IMU input: .mcap, .svo or .svo2 (default: %(default)s)")
    parser.add_argument("--gps-file", default=GPS_FILE_PATH,
                        help="MCAP path with GPS data (default: %(default)s)")
    parser.add_argument("--output-location", default=OUTPUT_PATH,
                        help="General output folder; a date/time/timestamp tree is created inside it "
                             "(default: %(default)s)")
    parser.add_argument("--timestamp", default=TIME, type=float,
                        help="Reference Unix timestamp (default: %(default)s)")
    parser.add_argument("--gps_topic", default=GPS_TOPIC,
                        help="GPS topic name (default: %(default)s)")
    parser.add_argument("--plot", action="store_true",
                        help="Show plots")
    args = parser.parse_args()
    run(args.imu_file, args.gps_file, args.timestamp, args.gps_topic, args.plot, args.output_location)
