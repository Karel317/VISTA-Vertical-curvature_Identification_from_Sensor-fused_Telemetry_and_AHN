"""
validation_curvature.py
Importable + standalone version of Validation_curvature_IMU_GPS.ipynb.

    run(input_file: str, timestamp: str) -> list[str]   # paths of saved .npz

input_file = ZED pose/IMU MCAP (the file that changes per session)
timestamp  = reference Unix time (string); selects the window + names the output
"""

# ── Module-level setup: runs ONCE when imported (input-independent only) ──────
import os
import argparse
import numpy as np
import scipy.interpolate as sc
from scipy.signal import savgol_filter
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from pyproj import Transformer
from filterpy.kalman import KalmanFilter

# CONFIG — tweak here; reused by every run() call
POSE_TOPIC = "/zed/pose"
IMU_TOPIC  = "/zed/imu/data"
GPS_TOPIC  = "/navsat_topic"

INPUT_FILE = r"D:\Data_gathered\2026_05_22\Rosbag\10_50_00\rosbag\rosbag_0.mcap"
TIMESTAMP  = "1779439917.972930459"
SAVE_ROOT  = r"D:\Validation_results"

DELTA_M    = 0.1
WIN_BEFORE = 5.0
WIN_AFTER  = 20.0
MIN_SPEED  = 0.5
SMOOTH_W   = 21
GPS_WINDOW = 30.0

# Reusable, somewhat-expensive resource: build the CRS transformer ONCE and let
# every run() reuse it. This is exactly the "slow setup at module scope" idea.
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


# ── Internal helpers ─────────────────────────────────────────────────────────
def _quaternion_to_pitch(qx, qy, qz, qw):
    qx, qy, qz, qw = map(np.array, (qx, qy, qz, qw))
    sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    return np.degrees(np.arcsin(sinp))


def _load_zed(input_file):
    pose = {k: [] for k in ("t", "x", "y", "z", "qx", "qy", "qz", "qw")}
    imu  = {"t": [], "omega_z": []}
    with open(input_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=[POSE_TOPIC, IMU_TOPIC]):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            if channel.topic == POSE_TOPIC:
                pose["t"].append(t)
                pose["x"].append(msg.pose.position.x)
                pose["y"].append(msg.pose.position.y)
                pose["z"].append(msg.pose.position.z)
                pose["qx"].append(msg.pose.orientation.x)
                pose["qy"].append(msg.pose.orientation.y)
                pose["qz"].append(msg.pose.orientation.z)
                pose["qw"].append(msg.pose.orientation.w)
            else:
                imu["t"].append(t)
                imu["omega_z"].append(msg.angular_velocity.z)
    return pose, imu


def _load_gps(ts):
    raw = {"t": [], "lat": [], "lon": [], "alt": []}
    start_ns = int((ts - GPS_WINDOW) * 1e9)
    end_ns   = int((ts + GPS_WINDOW) * 1e9)
    with open(GPS_FILE_PATH, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, message, msg in reader.iter_decoded_messages(
                topics=[GPS_TOPIC], start_time=start_ns, end_time=end_ns):
            raw["t"].append(message.publish_time / 1e9)
            raw["lat"].append(msg.latitude)
            raw["lon"].append(msg.longitude)
            raw["alt"].append(msg.altitude)
    rd_xs, rd_ys = TRANSFORMER.transform(raw["lon"], raw["lat"])   # reuse module resource
    gps_t = np.array(raw["t"])
    gps_s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rd_xs), np.diff(rd_ys)))])
    return raw, gps_t, gps_s


def _fuse_speed(pose_t, pose_dist, gps_t, gps_s):
    pose_dt   = np.diff(pose_t)
    v_zed_raw = np.where(pose_dt > 1e-6, pose_dist / pose_dt, 0.0)
    v_zed_t   = 0.5 * (pose_t[:-1] + pose_t[1:])
    gps_dt    = np.diff(gps_t)
    v_gps_raw = np.where(gps_dt > 1e-6, np.diff(gps_s) / gps_dt, 0.0)
    v_gps_t   = 0.5 * (gps_t[:-1] + gps_t[1:])

    kf  = KalmanFilter(dim_x=2, dim_z=1)
    dt0 = float(np.median(pose_dt))
    kf.x = np.array([[float(v_zed_raw[0])], [0.0]])
    kf.P = np.diag([4.0, 1.0])
    kf.H = np.array([[1.0, 0.0]])
    sigma_j = 0.5
    kf.Q = np.array([[dt0**4 / 4 * sigma_j**2, dt0**3 / 2 * sigma_j**2],
                     [dt0**3 / 2 * sigma_j**2, dt0**2     * sigma_j**2]])
    R_zed, R_gps = np.array([[0.25]]), np.array([[1.0]])

    v_fused = np.zeros(len(v_zed_t))
    for i in range(len(v_zed_t)):
        dt = float(pose_dt[i]) if pose_dt[i] > 1e-6 else dt0
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        kf.predict()
        kf.R = R_zed
        kf.update(np.array([[v_zed_raw[i]]]))
        j = np.searchsorted(v_gps_t, v_zed_t[i])
        if 0 < j < len(v_gps_t) and abs(v_gps_t[j] - v_zed_t[i]) < 0.12:
            kf.R = R_gps
            kf.update(np.array([[v_gps_raw[j]]]))
        v_fused[i] = max(kf.x[0, 0], 0.0)
    return v_zed_t, v_fused


# ── Public interface ─────────────────────────────────────────────────────────
def run(input_file: str, timestamp: str) -> list[str]:
    ts = float(timestamp)
    print(f"[validation_curvature] {input_file}  @ t={ts}")

    pose, imu = _load_zed(input_file)
    raw, gps_t, gps_s = _load_gps(ts)

    pose_t = np.array(pose["t"])
    pose_x = np.array(pose["x"])
    pose_y = np.array(pose["y"])
    pose_dist = np.hypot(np.diff(pose_x), np.diff(pose_y))
    pose_s    = np.concatenate([[0.0], np.cumsum(pose_dist)])

    v_zed_t, v_fused = _fuse_speed(pose_t, pose_dist, gps_t, gps_s)
    v_fused_at_pose  = np.interp(pose_t, v_zed_t, v_fused)

    # Elevation from integrated pitch
    pitch = _quaternion_to_pitch(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    pitch -= np.mean(pitch)
    if SMOOTH_W > 0:
        pitch = savgol_filter(pitch, window_length=SMOOTH_W, polyorder=3)
    ds = np.concatenate([[0.0], pose_dist])
    elevation = np.cumsum(ds * np.sin(np.radians(pitch)))

    ref_idx = int(np.argmin(np.abs(pose_t - ts)))
    s_rel = pose_s - pose_s[ref_idx]
    mask  = (s_rel >= -WIN_BEFORE) & (s_rel <= WIN_AFTER)

    s_win = s_rel[mask]
    z_win = elevation[mask] - elevation[mask][0]
    pitch_rad_win = np.radians(pitch[mask])

    # Curvature: spline (Frenet), pitch-derivative, and path ω_z / v
    sp_z   = sc.make_splrep(s_win, z_win, s=0)
    dzds   = sp_z.derivative(nu=1)(s_win)
    d2zds2 = sp_z.derivative(nu=2)(s_win)
    kappa_terrain_spline = np.abs(d2zds2) / (1.0 + dzds**2) ** 1.5
    kappa_terrain_pitch  = np.gradient(pitch_rad_win, s_win)

    omega_z_win = np.interp(pose_t, np.array(imu["t"]), np.array(imu["omega_z"]))[mask]
    speed_win   = v_fused_at_pose[mask]
    kappa_path  = np.where(speed_win > MIN_SPEED, omega_z_win / speed_win, np.nan)

    # ── Save (mirror the notebook: a results dir + {method}_{timestamp}.npz) ──
    save_dir = os.path.join(SAVE_ROOT, "pipeline_out")   # or derive date/time from input_file
    os.makedirs(save_dir, exist_ok=True)
    saved = []

    method = "EKF_curvature_validation"
    p = os.path.join(save_dir, f"{method}_{timestamp}.npz")
    np.savez_compressed(p, x=None, y=None, z=z_win, s=s_win, t=[ts],
                        method=np.array([method]),
                        kappa_terrain_spline=kappa_terrain_spline,
                        kappa_terrain_pitch=kappa_terrain_pitch,
                        kappa_path=kappa_path,
                        gps_lon=raw["lon"], gps_lat=raw["lat"])
    saved.append(p)

    # Z from positional tracking
    method = "Z_positional_tracking"
    z_all = np.array(pose["z"])
    cum   = pose_s - pose_s[ref_idx]
    m2    = (cum >= -WIN_BEFORE) & (cum <= WIN_AFTER)
    p = os.path.join(save_dir, f"{method}_{timestamp}.npz")
    np.savez_compressed(p, x=None, y=None, z=z_all[m2] - z_all[ref_idx],
                        s=cum[m2], t=[ts], method=np.array([method]))
    saved.append(p)

    print(f"[validation_curvature] saved: {saved}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU+GPS curvature validation.")
    parser.add_argument("--input-file", default=INPUT_FILE,
                        help="MCAP path with GPS (default: %(default)s)")
    parser.add_argument("--timestamp", default=TIMESTAMP,
                        help="Reference Unix timestamp (default: %(default)s)")
    args = parser.parse_args()
    run(args.input_file, args.timestamp)
