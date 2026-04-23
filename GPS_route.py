from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from pyproj import Transformer
import requests
import rasterio
import io
import math
import time

# ----------------- CONFIGURATION --------------------------
IMU_FILE_PATH = r"C:\Users\Leons\Downloads\back_front_left_right_0.mcap"
GPS_FILE_PATH = r"C:\Users\Leons\Downloads\rosbag_0.mcap"

# CHANGE THIS to the topic found in File 1 that contains NavSatFix (Lat/Lon)
GPS_LAT_LON_TOPIC = "/navsat_topic"
GPS_SPEED_TOPIC = "/odometry/gps"

IMU_TOPICS = ["/zed_left/zed_node/imu/data"] # Kept to 1 for clean plotting, add more if needed

# ------------------- DATA CONTAINERS -----------------------
data = {topic: {"t": [], "gy": [], "qx": [], "qy": [], "qz": [], "qw": []} for topic in IMU_TOPICS}
gps_data = {"t": [], "v": []}
gps_coords = {"t": [], "lat": [], "lon": []}

# ------------------- FILE READING --------------------------
print("Reading IMU and GPS files...")
# (IMU Reading)
with open(IMU_FILE_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic in IMU_TOPICS:
            d = data[channel.topic]
            d["t"].append(message.publish_time / 1e9)
            d["gy"].append(ros_msg.angular_velocity.y)
            d["qx"].append(ros_msg.orientation.x)
            d["qy"].append(ros_msg.orientation.y)
            d["qz"].append(ros_msg.orientation.z)
            d["qw"].append(ros_msg.orientation.w)

# (GPS Reading)
with open(GPS_FILE_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        # Get Speed
        if channel.topic == GPS_SPEED_TOPIC:
            gps_data["t"].append(message.publish_time / 1e9)
            gps_data["v"].append(ros_msg.twist.twist.linear.x)
        
        # Get Coordinates
        elif channel.topic == GPS_LAT_LON_TOPIC:
            gps_coords["t"].append(message.publish_time / 1e9)
            gps_coords["lat"].append(ros_msg.latitude)
            gps_coords["lon"].append(ros_msg.longitude)

# --------------- HELPER FUNCTIONS -----------------
def quaternion_to_pitch(qx, qy, qz, qw):
    qx, qy, qz, qw = np.array(qx), np.array(qy), np.array(qz), np.array(qw)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1, 1)
    return np.degrees(np.arcsin(sinp))

def get_dtm_elevation(x, y, session):
    """Fetches AHN4 DTM elevation for a single Dutch RD coordinate."""
    wcs_url = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
    params = {
        "service": "WCS", "version": "1.0.0", "request": "GetCoverage",
        "coverage": "dtm_05m", "crs": "EPSG:28992",
        "bbox": f"{x-0.5},{y-0.5},{x+0.5},{y+0.5}",
        "width": 1, "height": 1, "format": "GEOTIFF"
    }
    try:
        resp = session.get(wcs_url, params=params, timeout=5)
        if resp.status_code == 200:
            with rasterio.open(io.BytesIO(resp.content)) as src:
                return src.read(1)[0][0]
    except Exception:
        pass
    return None

# ------------------ AHN4 DTM PROCESSING --------------------
print("Processing AHN4 True Ground Elevation...")

# Convert all GPS to RD coordinates
transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
rd_xs, rd_ys = transformer.transform(gps_coords["lon"], gps_coords["lat"])

dtm_elevations = []
dtm_distances = []
current_distance = 0.0

# Smart Downsampling: Only query API if vehicle moved > 3 meters to avoid API bans
session = requests.Session()
last_x, last_y = None, None

print(f"Total GPS points: {len(rd_xs)}. Downsampling for API queries...")

for i in range(len(rd_xs)):
    x, y = rd_xs[i], rd_ys[i]
    
    if last_x is None:
        dist_step = 0
    else:
        dist_step = math.hypot(x - last_x, y - last_y)

    # Only query if we moved 3 meters from the last queried point
    if last_x is None or dist_step >= 3.0:
        h = get_dtm_elevation(x, y, session)
        if h is not None and -100 < h < 200:            
            dtm_elevations.append(h)
            dtm_distances.append(current_distance)
        
        last_x, last_y = x, y
        print(f"Queried DTM: {len(dtm_elevations)} points...", end="\r")
        time.sleep(0.05) # Polite delay for PDOK servers
    
    # Keep track of distance based on coordinates
    if i > 0:
        current_distance += math.hypot(x - rd_xs[i-1], y - rd_ys[i-1])

print("\nDone fetching DTM data.")

# ------------------ IMU PROCESSING & PLOTTING -----------------
d = data[IMU_TOPICS[0]]
raw_t = np.array(d["t"])
t_plot = raw_t - raw_t[0]

pitch = quaternion_to_pitch(d["qx"], d["qy"], d["qz"], d["qw"])
pitch = savgol_filter(pitch, window_length=501, polyorder=3)

# Interpolate GPS speed to IMU timestamps
t_gps_raw = np.array(gps_data["t"])
v_gps = np.array(gps_data["v"])
v_gps_interp = np.interp(raw_t, t_gps_raw, v_gps)
speed = np.abs(v_gps_interp)

# IMU Estimated Elevation
dt = np.diff(t_plot, prepend=0)
ds = speed * dt
imu_distance = np.cumsum(ds)
imu_elevation = np.cumsum(ds * np.sin(np.radians(pitch)))

# Align IMU elevation to start at the exact same NAP height as the DTM
if len(dtm_elevations) > 0:
    imu_elevation = imu_elevation + dtm_elevations[0]

# --- PLOTTING ---
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot IMU estimate
ax.plot(imu_distance, imu_elevation, label="IMU Estimated Elevation", color='blue', linewidth=2)

# Plot DTM Ground Truth
if len(dtm_distances) > 0:
    ax.plot(dtm_distances, dtm_elevations, label="AHN4 DTM (True Ground)", color='green', linewidth=2, linestyle="--")

ax.set_title("Ground Profile: IMU Estimate vs AHN4 True DTM", fontsize=14, fontweight="bold")
ax.set_xlabel("Distance Traveled (m)")
ax.set_ylabel("Elevation (m NAP)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()