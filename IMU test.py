from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ==========================================
# CONFIGURATION
# ==========================================
# File 1: Contains the ZED IMU data
IMU_FILE_PATH = r"C:\Users\Leons\Downloads\back_front_left_right_0.mcap" 

# File 2: Contains the GPS / Odometry data
GPS_FILE_PATH = r"C:\Users\Leons\Downloads\rosbag_0.mcap" # <-- UPDATE THIS

MODE = 1  # 1 is everything side-by-side, 2 is all pitch angles on one graph

IMU_TOPICS = [
    "/zed_left/zed_node/imu/data",
    "/zed_right/zed_node/imu/data",
    "/zed_front/zed_node/imu/data",
    "/zed_back/zed_node/imu/data",
]

GPS_TOPIC = "/odometry/gps" 


MOUNTING_PITCH_OFFSET = 0.0  # Set this if the camera is physically tilted up/down on the car (degrees)
PITCH_MULTIPLIER = 1.0       # Set to -1.0 if your elevation graph goes DOWN when you drive UP a hill

# ==========================================
# DATA STORAGE
# ==========================================
data = {topic: {"t": [],"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": [], "qx": [], "qy": [], "qz": [], "qw": []} for topic in IMU_TOPICS}
gps_data = {"t": [], "v": []}

# ==========================================
# FILE READING
# ==========================================
print("Reading IMU file...")
with open(IMU_FILE_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic in IMU_TOPICS:
            d = data[channel.topic]
            d["t"].append(message.publish_time / 1e9)
            d["ax"].append(ros_msg.linear_acceleration.x)
            d["ay"].append(ros_msg.linear_acceleration.y)
            d["az"].append(ros_msg.linear_acceleration.z)
            d["gx"].append(ros_msg.angular_velocity.x)
            d["gy"].append(ros_msg.angular_velocity.y)
            d["gz"].append(ros_msg.angular_velocity.z)
            d["qx"].append(ros_msg.orientation.x)
            d["qy"].append(ros_msg.orientation.y)
            d["qz"].append(ros_msg.orientation.z)
            d["qw"].append(ros_msg.orientation.w)

print("Reading GPS file...")
try:
    with open(GPS_FILE_PATH, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            if channel.topic == GPS_TOPIC:
                gps_data["t"].append(message.publish_time / 1e9)
                gps_data["v"].append(ros_msg.twist.twist.linear.x)
except FileNotFoundError:
    print(f"Warning: Could not find the GPS file at {GPS_FILE_PATH}")
            
print("Done reading!")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def quaternion_to_euler(qx, qy, qz, qw):
    qx, qy, qz, qw = np.array(qx), np.array(qy), np.array(qz), np.array(qw)
    
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr, cosr)

    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1, 1)
    pitch = np.arcsin(sinp)

    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny, cosy)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def kalman_filter_velocity(ax, t, v_gps, t_gps):
    if len(t_gps) == 0 or len(v_gps) == 0:
        print("WARNING: No GPS data found. Defaulting to pure acceleration integration.")
        return np.cumsum(ax) * np.mean(np.diff(t))

    v_gps_interp = np.interp(t, t_gps, v_gps)
    v_fused = np.zeros_like(ax)
    v = 0.0  
    P = 1.0  
    Q = 0.01 
    R = 0.5  
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        v = v + (ax[i] * dt)
        P = P + Q
        K = P / (P + R)  
        v = v + K * (v_gps_interp[i] - v)
        P = (1 - K) * P
        v_fused[i] = v
        
    return v_fused

def calculate_elevation(speed_array, pitch_deg, t_plot):
    """Calculates strictly increasing distance and true elevation."""
    pitch_rad = np.radians(pitch_deg)
    
    # dt based on plot time
    dt = np.diff(t_plot, prepend=0)
    
    # Distance strictly increases (ds is always positive)
    ds = speed_array * dt
    distance = np.cumsum(ds)
    
    # Elevation = distance * sin(pitch)
    dz = ds * np.sin(pitch_rad)
    elevation = np.cumsum(dz)
    
    return distance, elevation

# ==========================================
# PLOTTING
# ==========================================

# ── MODE 1: per-camera side-by-side plots ────────────────────────────────────
if MODE == 1:
    # Use raw absolute timestamps for GPS
    t_gps_raw = np.array(gps_data["t"])
    v_gps = np.array(gps_data["v"])
    
    for topic in IMU_TOPICS:
        d = data[topic]
        if not d["t"]:
            continue

        name = topic.split("/")[1]
        
        # RAW epoch timestamps from the IMU
        raw_t = np.array(d["t"])
        # Zeroed timestamps ONLY for plotting
        t_plot = raw_t - raw_t[0]

        pitch_rate = np.array(d["gy"])
        roll, pitch, yaw = quaternion_to_euler(d["qx"], d["qy"], d["qz"], d["qw"])

        # FIX: Do not subtract pitch[0]. Only correct for camera mounting angle.
        pitch = (pitch - MOUNTING_PITCH_OFFSET) * PITCH_MULTIPLIER
        pitch = savgol_filter(pitch, window_length=501, polyorder=3)
        
        # FIX: Interpolate using the raw, absolute global timestamps to guarantee alignment
        if len(t_gps_raw) > 0:
            v_gps_interp = np.interp(raw_t, t_gps_raw, v_gps)
        else:
            print("WARNING: No GPS data! Distance will be 0.")
            v_gps_interp = np.zeros_like(raw_t)
            
        # FIX: Force speed to be absolutely positive so distance only goes up
        speed = np.abs(v_gps_interp)
        
        # Prevent division by zero for the curvature math
        v_safe = np.where(speed < 0.1, 0.1, speed) 
        
        # Curvature calculation
        curve = pitch_rate / v_safe  
        curve = savgol_filter(curve, window_length=201, polyorder=3)
        
        # Calculate 2D Elevation Profile
        distance, elevation = calculate_elevation(speed, pitch, t_plot)

        # PLOTTING
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(name, fontsize=14, fontweight="bold")

        ax1.plot(t_plot, curve, label="Curvature", color='blue')
        ax1.set_title("Curvature (1/m)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Kappa (1/m)")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(t_plot, pitch, label="Pitch", color='orange')
        ax2.set_title("Absolute True Pitch")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Degrees (°)")
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(distance, elevation, label="Hill Profile", color='green', linewidth=2)
        ax3.set_title("Approximate Elevation Profile")
        ax3.set_xlabel("Distance Traveled (m)")
        ax3.set_ylabel("Elevation Gain (m)")
        ax3.fill_between(distance, elevation, alpha=0.3, color='green') 
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
elif MODE == 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("All Cameras — Pitch", fontsize=14, fontweight="bold")

    pitch_data = {}
    pitch_rate_data = {}

    for topic in IMU_TOPICS:
        d = data[topic]
        if not d["t"]:
            continue

        name = topic.split("/")[1]
        t = np.array(d["t"])
        t = t - t[0]

        pitch_rate = np.array(d["gy"])
        _, pitch, _ = quaternion_to_euler(d["qx"], d["qy"], d["qz"], d["qw"])
        pitch = pitch - pitch[0]
        
        pitch = savgol_filter(pitch, window_length=51, polyorder=3)
        pitch_rate = savgol_filter(pitch_rate, window_length=51, polyorder=3)

        pitch_data[name] = (t, pitch)
        pitch_rate_data[name] = (t, pitch_rate)

        ax1.plot(t, pitch_rate, label=name, alpha=0.4, linestyle="--")
        ax2.plot(t, pitch, label=name, alpha=0.4, linestyle="--")

    if "zed_left" in pitch_data and "zed_right" in pitch_data:
        t_left, p_left = pitch_data["zed_left"]
        t_right, p_right = pitch_data["zed_right"]
        t_shared = np.linspace(max(t_left[0], t_right[0]), min(t_left[-1], t_right[-1]), 1000)

        p_left_interp = np.interp(t_shared, t_left, p_left)
        p_right_interp = np.interp(t_shared, t_right, p_right)
        p_avg = (p_left_interp + p_right_interp) / 2

        pr_left_interp = np.interp(t_shared, pitch_rate_data["zed_left"][0], pitch_rate_data["zed_left"][1])
        pr_right_interp = np.interp(t_shared, pitch_rate_data["zed_right"][0], pitch_rate_data["zed_right"][1])
        pr_avg = (pr_left_interp + pr_right_interp) / 2

        ax1.plot(t_shared, pr_avg, label="left+right avg", linewidth=2, color="black")
        ax2.plot(t_shared, p_avg, label="left+right avg", linewidth=2, color="black")

    ax1.set_title("Pitch Angular Velocity")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("rad/s")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Pitch Orientation")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Degrees (°)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

plt.show()