from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

FILE_PATH = r"C:\Users\Leons\Downloads\back_front_left_right_0.mcap"  # 👈 only change this

MODE = 1  # 👈 change to 1 or 2

IMU_TOPICS = [
    "/zed_left/zed_node/imu/data",
    "/zed_right/zed_node/imu/data",
    "/zed_front/zed_node/imu/data",
    "/zed_back/zed_node/imu/data",
]

# Storage for each camera
data = {topic: {"t": [], "gx": [], "gy": [], "gz": [], "qx": [], "qy": [], "qz": [], "qw": []} for topic in IMU_TOPICS}

print("Reading file...")
with open(FILE_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic in IMU_TOPICS:
            d = data[channel.topic]
            d["t"].append(message.publish_time / 1e9)
            d["gx"].append(ros_msg.angular_velocity.x)
            d["gy"].append(ros_msg.angular_velocity.y)
            d["gz"].append(ros_msg.angular_velocity.z)
            d["qx"].append(ros_msg.orientation.x)
            d["qy"].append(ros_msg.orientation.y)
            d["qz"].append(ros_msg.orientation.z)
            d["qw"].append(ros_msg.orientation.w)
print("Done reading!")

def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion arrays to roll, pitch, yaw in degrees."""
    qx, qy, qz, qw = np.array(qx), np.array(qy), np.array(qz), np.array(qw)

    # Roll (x-axis)
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr, cosr)

    # Pitch (y-axis)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny, cosy)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# ── MODE 1: per-camera side-by-side plots ────────────────────────────────────
if MODE == 1:
    for topic in IMU_TOPICS:
        d = data[topic]
        if not d["t"]:
            print(f"No data found for {topic}")
            continue

        name = topic.split("/")[1]
        t = np.array(d["t"])
        t = t - t[0]

        roll_rate  = np.array(d["gx"])
        pitch_rate = np.array(d["gy"])
        yaw_rate   = np.array(d["gz"])

        roll, pitch, yaw = quaternion_to_euler(d["qx"], d["qy"], d["qz"], d["qw"])

        # Normalize angles relative to starting position
        roll  = roll  - roll[0]
        pitch = pitch - pitch[0]
        yaw   = yaw   - yaw[0]

        # Smooth
        roll  = savgol_filter(roll,  window_length=201, polyorder=3)
        pitch = savgol_filter(pitch, window_length=201, polyorder=3)
        yaw   = savgol_filter(yaw,   window_length=201, polyorder=3)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle(name, fontsize=14, fontweight="bold")

        # Left — angular velocity
        ax1.plot(t, pitch_rate, label="Pitch rate")
        ax1.set_title("Angular Velocity")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("rad/s")
        ax1.legend()
        ax1.grid(True)

        # Right — orientation from quaternion
        ax2.plot(t, pitch, label="Pitch")
        ax2.set_title("Orientation (from sensor, drift-free)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Degrees (°)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

# ── MODE 2: all 4 pitch angles on one graph ───────────────────────────────────
elif MODE == 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("All Cameras — Pitch", fontsize=14, fontweight="bold")

    pitch_data = {}
    pitch_rate_data = {}

    for topic in IMU_TOPICS:
        d = data[topic]
        if not d["t"]:
            print(f"No data found for {topic}")
            continue

        name = topic.split("/")[1]
        t = np.array(d["t"])
        t = t - t[0]

        pitch_rate = np.array(d["gy"])
        _, pitch, _ = quaternion_to_euler(d["qx"], d["qy"], d["qz"], d["qw"])
        pitch = pitch - pitch[0]
        pitch      = savgol_filter(pitch,      window_length=51, polyorder=3)
        pitch_rate = savgol_filter(pitch_rate, window_length=51, polyorder=3)

        pitch_data[name]      = (t, pitch)
        pitch_rate_data[name] = (t, pitch_rate)

        ax1.plot(t, pitch_rate, label=name, alpha=0.4, linestyle="--")
        ax2.plot(t, pitch,      label=name, alpha=0.4, linestyle="--")

    # Average left and right
    if "zed_left" in pitch_data and "zed_right" in pitch_data:
        t_left,  p_left  = pitch_data["zed_left"]
        t_right, p_right = pitch_data["zed_right"]
        t_shared = np.linspace(max(t_left[0], t_right[0]), min(t_left[-1], t_right[-1]), 1000)

        p_left_interp  = np.interp(t_shared, t_left,  p_left)
        p_right_interp = np.interp(t_shared, t_right, p_right)
        p_avg = (p_left_interp + p_right_interp) / 2

        pr_left_interp  = np.interp(t_shared, pitch_rate_data["zed_left"][0],  pitch_rate_data["zed_left"][1])
        pr_right_interp = np.interp(t_shared, pitch_rate_data["zed_right"][0], pitch_rate_data["zed_right"][1])
        pr_avg = (pr_left_interp + pr_right_interp) / 2

        ax1.plot(t_shared, pr_avg, label="left+right avg", linewidth=2, color="black")
        ax2.plot(t_shared, p_avg,  label="left+right avg", linewidth=2, color="black")

    ax1.set_title("Pitch Angular Velocity")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("rad/s")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Pitch Orientation (drift-free)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Degrees (°)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

plt.show()