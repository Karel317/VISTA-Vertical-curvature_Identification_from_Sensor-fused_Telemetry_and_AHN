#!/usr/bin/env python3
"""
ZED SVO → MCAP converter.
Reads an SVO file, runs positional tracking, and writes
sensor_msgs/Imu and geometry_msgs/PoseStamped messages to MCAP.

Usage:
    python "IMU SVO to MCAP.py" recording.svo [--output recording.mcap]
"""
import argparse
import math
import struct
import sys
from pathlib import Path

import pyzed.sl as sl
from mcap.writer import Writer as McapWriter

# ── Hardcoded paths (set these to run directly from IDE) ─────────────────────
# Leave SVO_PATH as an empty string to use command-line arguments instead.

SVO_PATH = r"D:\Data_gathered\2026_05_22\Camera\10_50_00\front_22_05_2026-10_50_00.svo2"     # e.g. r"C:\recordings\my_recording.svo"
OUTPUT_PATH = r""    # leave empty to auto-generate alongside the SVO file

# ── Time-range filter ─────────────────────────────────────────────────────────
# Set both to Unix timestamps (seconds, float) to process only that slice.
# Leave as 0.0 to process the entire file.
START_UNIX_S = 1779439914.817100912   # e.g. 1716372600.0
END_UNIX_S   = 1779439925.810906095   # e.g. 1716372660.0


# ── ROS2 message definitions (ros2msg schema encoding) ───────────────────────

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


# ── CDR encoder ───────────────────────────────────────────────────────────────

class CDREncoder:
    """Minimal CDR (little-endian) encoder for ROS2 messages.

    Tracks the current buffer position and inserts alignment padding
    so that each field starts at its natural alignment boundary.
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


def encode_imu(
    sec: int, nsec: int, frame_id: str,
    qx: float, qy: float, qz: float, qw: float,
    gx: float, gy: float, gz: float,   # rad/s
    ax: float, ay: float, az: float,   # m/s²
) -> bytes:
    e = CDREncoder()
    e.header(sec, nsec, frame_id)
    e.float64(qx); e.float64(qy); e.float64(qz); e.float64(qw)
    e.float64_array([0.0] * 9)   # orientation_covariance (unknown → zeros)
    e.float64(gx); e.float64(gy); e.float64(gz)
    e.float64_array([0.0] * 9)   # angular_velocity_covariance
    e.float64(ax); e.float64(ay); e.float64(az)
    e.float64_array([0.0] * 9)   # linear_acceleration_covariance
    return e.to_bytes()


def encode_pose_stamped(
    sec: int, nsec: int, frame_id: str,
    tx: float, ty: float, tz: float,
    qx: float, qy: float, qz: float, qw: float,
) -> bytes:
    e = CDREncoder()
    e.header(sec, nsec, frame_id)
    e.float64(tx); e.float64(ty); e.float64(tz)
    e.float64(qx); e.float64(qy); e.float64(qz); e.float64(qw)
    return e.to_bytes()

# ── SVO timestamp seek ────────────────────────────────────────────────────────

def _svo_frame_at_ts(zed: sl.Camera, sensors_data: sl.SensorsData,
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if SVO_PATH:
        svo_path = Path(SVO_PATH)
        output_path = Path(OUTPUT_PATH) if OUTPUT_PATH else svo_path.with_suffix(".mcap")
    else:
        parser = argparse.ArgumentParser(
            description="Convert a ZED SVO recording to MCAP (positional tracking + IMU)."
        )
        parser.add_argument("svo_path", help="Path to input .svo or .svo2 file")
        parser.add_argument(
            "--output", "-o",
            help="Output MCAP path (default: same directory and stem as svo_path, .mcap extension)",
        )
        args = parser.parse_args()
        svo_path = Path(args.svo_path)
        output_path = Path(args.output) if args.output else svo_path.with_suffix(".mcap")

    if not svo_path.exists():
        sys.exit(f"Error: SVO file not found: {svo_path}")

    # ── ZED ──────────────────────────────────────────────────────────────────
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP  # ROS2 REP-103
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS  # GEN 3
    init_params.svo_real_time_mode = False  # process every frame as fast as possible

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"ZED open failed: {status}")

    pose = sl.Pose()
    sensors_data = sl.SensorsData()
    translation = sl.Translation()
    orientation = sl.Orientation()

    total_frames = zed.get_svo_number_of_frames()
    print(f"SVO: {svo_path.name}  ({total_frames} frames)")
    print(f"Out: {output_path}")

    start_ns = int(START_UNIX_S * 1_000_000_000) if START_UNIX_S else 0
    end_ns   = int(END_UNIX_S   * 1_000_000_000) if END_UNIX_S   else 0

    # Seek BEFORE enabling tracking — set_svo_position is unreliable while tracking is active
    if start_ns:
        start_frame = _svo_frame_at_ts(zed, sensors_data, start_ns, total_frames)
        print(f"Seeking to frame {start_frame} (ts >= {START_UNIX_S}s)")
        zed.set_svo_position(start_frame)
    if end_ns:
        print(f"Will stop at ts > {END_UNIX_S}s")

    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    # ── MCAP writer ───────────────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        writer = McapWriter(f)
        writer.start(profile="ros2", library="zed_svo_to_mcap")

        imu_schema_id = writer.register_schema(
            name="sensor_msgs/Imu",
            encoding="ros2msg",
            data=IMU_MSGDEF.encode(),
        )
        pose_schema_id = writer.register_schema(
            name="geometry_msgs/PoseStamped",
            encoding="ros2msg",
            data=POSE_MSGDEF.encode(),
        )
        imu_ch = writer.register_channel(
            topic="/zed/imu/data",
            message_encoding="cdr",
            schema_id=imu_schema_id,
        )
        pose_ch = writer.register_channel(
            topic="/zed/pose",
            message_encoding="cdr",
            schema_id=pose_schema_id,
        )

        frame = 0
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
            if end_ns and ts_ns > end_ns:
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

            # Angular velocity: ZED returns deg/s → convert to rad/s for ROS2
            av = imu_data.get_angular_velocity()
            la = imu_data.get_linear_acceleration()

            imu_bytes = encode_imu(
                sec, nsec, "imu",
                qx, qy, qz, qw,
                math.radians(float(av[0])),
                math.radians(float(av[1])),
                math.radians(float(av[2])),
                float(la[0]), float(la[1]), float(la[2]),
            )
            writer.add_message(
                channel_id=imu_ch,
                log_time=ts_ns,
                data=imu_bytes,
                publish_time=ts_ns,
            )

            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                pose.get_translation(translation)
                t = translation.get()
                pose_bytes = encode_pose_stamped(
                    sec, nsec, "world",
                    float(t[0]), float(t[1]), float(t[2]),
                    qx, qy, qz, qw,
                )
                writer.add_message(
                    channel_id=pose_ch,
                    log_time=ts_ns,
                    data=pose_bytes,
                    publish_time=ts_ns,
                )

            frame += 1
            if frame % 100 == 0:
                pct = 100 * frame // total_frames if total_frames > 0 else 0
                print(f"\r  {frame}/{total_frames} ({pct}%)", end="", flush=True)

        writer.finish()

    print(f"\nDone — wrote {frame} frames to {output_path}")

    zed.disable_positional_tracking()
    zed.close()


if __name__ == "__main__":
    main()
