from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ----------------- CONFIGURATION --------------------------

# Change these file paths!!
IMU_FILE_PATH = r"C:\ROSBAGS VERWIJDER NA BEP\back_front_left_right_0.mcap" # Contains the ZED IMU data
GPS_FILE_PATH = r"C:\ROSBAGS VERWIJDER NA BEP\rosbag_0.mcap" # Contains the GPS / Odometry data

IMU_TOPICS = [
    "/zed_left/zed_node/imu/data",
    "/zed_right/zed_node/imu/data",
    "/zed_front/zed_node/imu/data",
    "/zed_back/zed_node/imu/data",
]
GPS_TOPIC = "/odometry/gps" 


# Empty dictionaries to hold the data for each topic
data = {topic: {"t": [],"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": [], "qx": [], "qy": [], "qz": [], "qw": []} for topic in IMU_TOPICS}
gps_data = {"t": [], "v": []}

# ------------------- FILE READING --------------------------
print("Reading IMU file...")
try:
    with open(IMU_FILE_PATH, "rb") as f: # Opens the file in reading binary ("rb    ") mode and calls it f
        reader = make_reader(f, decoder_factories=[DecoderFactory()]) # Decodes the binary into python objects
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(): 
            # schema: Describes the structure of the data
            # channel.topic: Tells you which sensor sent the data
            # message.publish_time: When the message was recorded.
            # ros_msg: This is the actual data
            if channel.topic in IMU_TOPICS:
                d = data[channel.topic]
                d["t"].append(message.publish_time / 1e9)
                d["ax"].append(ros_msg.linear_acceleration.x)
                d["ay"].append(ros_msg.linear_acceleration.y)
                #d["az"].append(ros_msg.linear_acceleration.z)
                #d["gx"].append(ros_msg.angular_velocity.x)
                d["gy"].append(ros_msg.angular_velocity.y)
                #d["gz"].append(ros_msg.angular_velocity.z)
                d["qx"].append(ros_msg.orientation.x)
                d["qy"].append(ros_msg.orientation.y)
                d["qz"].append(ros_msg.orientation.z)
                d["qw"].append(ros_msg.orientation.w)
except FileNotFoundError:
    print(f"Warning: Could not find the IMU file at {IMU_FILE_PATH}")

print("Reading GPS file...")
try:
    with open(GPS_FILE_PATH, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            if channel.topic == GPS_TOPIC:
                gps_data["t"].append(message.publish_time / 1e9)
                gps_data["v"].append(ros_msg.twist.twist.linear.x) # Twist is the ROS message for velocity
except FileNotFoundError:
    print(f"Warning: Could not find the GPS file at {GPS_FILE_PATH}")
            
print("Done reading!")

# ---------------HELPER FUNCTIONS-----------------
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
        P = P + Q * dt
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

# ------------------PLOTTING--------------------------

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
    v_safe = np.where(speed < 0.01, 0.01, speed) 
    
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
plt.show()
