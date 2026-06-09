# File context and how to use them
Every file's context and usage will be explained briefly.

---

## .mcap _to_.bin.ipynb

This notebook extracts a merged LiDAR point cloud from a single timestamp in an `.mcap` recording and saves it as a `.bin` file. All specified LiDAR topics (M1P, helios_R, helios_L) are merged into one output file in nuScenes format. The notebook also includes utilities for building a GPS-aligned and IMU-aligned map from a time range of the recording, and a polar curvature intensity visualisation.

**Input** \
An `.mcap` recording with one or more `sensor_msgs/PointCloud2` topics

**Adjustable parameters** \
`MCAP_FILE` — path to the `.mcap` recording \
`OUTPUT_FILE` — path and filename for the output `.bin` file \
`TARGET_TIME` — Unix timestamp of the frame to extract \
`TOPICS_TO_MERGE` — list of LiDAR ROS topic names to merge into the output cloud

**Output** \
A nuScenes-format `(N, 5)` `float32` `.bin` file containing the merged point cloud at the requested timestamp

---

## Coordinates_at_time.ipynb

This notebook provides GPS coordinate utilities for working with route recordings. It can compute the distance and bearing between two GPS coordinates, offset a coordinate by a fixed distance in the reverse direction of travel, and look up the GPS coordinate recorded at a specific Unix timestamp by reading from an MCAP file. The GPS sample nearest to a target coordinate can also be found within a configurable search window.

**Adjustable parameters** \
`GPS_FILE_PATH` — path to the `.mcap` recording containing GPS data \
`TIME` — reference Unix timestamp \
`GPS_WINDOW` — seconds either side of `TIME` to read from the bag \
`GPS_TOPIC` — GPS ROS topic name \
`MATCH_THRESHOLD_M` — search radius in metres for finding GPS samples near a target coordinate

**Output** \
Printed coordinate values, distances, and bearing angles in the notebook output

---

## Cutting_MCAP.py

This script cuts a time slice from a `.mcap` recording, keeping only the messages between a specified start and end Unix timestamp. All topics and the TF tree are preserved in the output file. The output filename and folder are derived automatically from the date and time encoded in the input file path.

**Input** \
The original uncut `.mcap` recording file

**Adjustable parameters** \
`INPUT_FILE` — path to the source `.mcap` file (expects the path structure `<date>/Rosbag/<time>/rosbag/rosbag_0.mcap` for auto-naming) \
`OPTIONAL_NAME` — short label prepended to the output filename (e.g. `"kuitbrug"`; set to `""` to omit) \
`START_TIME` — start Unix timestamp in seconds (float) \
`END_TIME` — end Unix timestamp in seconds (float)

**Output** \
A trimmed `.mcap` file saved to `<OneDrive BEP folder>/<date>/` with the name `<label>_<date>_<time>.mcap`

---

## IMU SVO to MCAP.py

This script converts a ZED camera `.svo` or `.svo2` recording into an `.mcap` file containing `sensor_msgs/Imu` messages (angular velocity and linear acceleration) and `geometry_msgs/PoseStamped` messages (position and orientation from ZED positional tracking). An optional time-range filter can be applied to export only a specific window from the SVO file. The output MCAP is ready to use as the IMU input for `EKF.py`.

**Input** \
A ZED `.svo` or `.svo2` recording file

**Adjustable parameters** \
`SVO_PATH` — path to the input `.svo`/`.svo2` file \
`OUTPUT_PATH` — path for the output `.mcap` file (leave empty to auto-generate a path next to the SVO file) \
`START_UNIX_S` / `END_UNIX_S` — Unix timestamps to limit conversion to a time slice (set both to `0.0` to convert the entire file) \
Can also be run from the command line: `python "IMU SVO to MCAP.py" recording.svo [--output recording.mcap]`

**Output** \
An `.mcap` file with topics `/zed/imu/data` (`sensor_msgs/Imu`) and `/zed/pose` (`geometry_msgs/PoseStamped`)

---

## extract_pointcloud_segment.py

This script extracts a world-frame point cloud segment from a defined time window in an `.mcap` recording by combining LiDAR scans with KISS-ICP poses. Each scan is transformed to the global frame using the KISS-ICP trajectory (with SLERP rotation interpolation and linear translation interpolation), applying per-sensor extrinsic calibration from the `SENSOR_TRANSFORMS` table. The resulting merged point cloud is visualised in an Open3D viewer with each sensor coloured differently.

**Input** \
The `.mcap` recording specified by `DATASET_PATH` \
Pre-computed KISS-ICP results (`poses.npy` + `timestamps.npy`) in `RESULTS_DIR`

**Adjustable parameters** \
`DATASET_PATH` — path to the `.mcap` recording \
`RESULTS_DIR` — folder containing the KISS-ICP `poses.npy` and `timestamps.npy` \
`T_START` / `T_END` — Unix timestamps defining the segment to extract \
`TIME_MARGIN` — seconds of padding added before `T_START` and after `T_END` (default: `0.5`) \
`LIDAR_TOPICS` — list of LiDAR ROS topic names to include \
`SENSOR_TRANSFORMS` — per-topic translation and RPY rotation for sensor-to-base_link calibration

**Output** \
An interactive Open3D 3D viewer showing the extracted world-frame point cloud coloured by sensor (orange = M1P, blue = helios_R, green = helios_L)

---

## map_all_scans.py

This script builds a full 3D map of a recording by accumulating all LiDAR scans, transforming each scan to the world frame using KISS-ICP poses, and voxel-downsampling the result in batches to keep memory usage manageable. A corridor filter retains only points within a set radius of the driven path. The final map is coloured by height using the viridis colormap and displayed in an Open3D viewer. It can optionally be saved to a KITTI-format `.bin` file.

**Input** \
The `.mcap` recording specified by `DATASET_PATH` \
Pre-computed KISS-ICP results (`poses.npy` + `timestamps.npy`) in `RESULTS_DIR`

**Adjustable parameters** \
`DATASET_PATH` — path to the `.mcap` recording \
`RESULTS_DIR` — folder containing the KISS-ICP `poses.npy` and `timestamps.npy` \
`LIDAR_TOPICS` — list of LiDAR ROS topic names to accumulate \
`VOXEL_SIZE_DISPLAY` — voxel size for downsampling in metres (default: `0.15`) \
`EVERY_N_SCANS` — process every Nth scan (default: `5`; set to `1` for full detail) \
`BATCH_DOWNSAMPLE_EVERY` — flush and downsample accumulated scans every N processed frames to limit memory use \
`CORRIDOR_RADIUS` — keep only points within this many metres of the driven path (default: `10.0`) \
`SAVE_MAP` — set to `True` to save the final map to disk \
`SAVE_PATH` — path for the output `.bin` file

**Output** \
An interactive Open3D 3D viewer showing the full height-coloured map with the trajectory overlaid in red. Optionally, a KITTI-format `(N, 4)` `float32` `.bin` file saved to `SAVE_PATH`.

---

## merge_csvs.py

This script merges all `.csv` files in a folder into a single output file. Each source file's first row (title row) is skipped and a `source_file` column is added to identify the origin of each row. The script skips `merged_output.csv` itself if it already exists in the folder.

**Adjustable parameters** \
`folder` — path to the folder containing the `.csv` files to merge (hardcoded as `D:\Validation_results\Statistics`; update this to the correct path before running)

**Output** \
A single `merged_output.csv` in `folder` containing the combined rows from all other `.csv` files in that folder
