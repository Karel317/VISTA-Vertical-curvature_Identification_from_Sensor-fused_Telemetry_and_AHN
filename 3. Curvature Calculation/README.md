# File context and how to use them
Every file's context and usage will be explained briefly.

---

## AHN5.py

This script queries the AHN5 (Actueel Hoogtebestand Nederland 5) national height map to obtain reference elevation data along the bike's path. It reads GPS coordinates from an MCAP recording, projects them to the Dutch RD New coordinate system (EPSG:28992), estimates the bike's heading at a reference timestamp, and samples both the DSM (Digital Surface Model) and DTM (Digital Terrain Model) layers of the AHN5 WCS API along a strip in front of and behind the bike. A spline is fitted to each elevation profile and curvature is computed analytically from the spline's derivatives. Results are saved as `.npz` files in the shared format used by `Validation_main.py`.

**Input** \
An `.mcap` recording containing a GPS topic (default: `/navsat_topic`)

**Adjustable parameters** \
`INPUT_FILE` — path to the `.mcap` recording \
`OUTPUT_PATH` — base folder for output; a `<YYYY_MM_DD>/<HH_MM_SS>/<timestamp>` subfolder is created automatically \
`GPS_TOPIC` — ROS topic name of the GPS data (set to `None` to list available topics) \
`TIME` — reference Unix timestamp; the AHN5 strip is centred on the bike's position at this moment \
`X_BACK` / `X_FRONT` — strip extent behind and in front of the bike in metres (default: `−5.0` / `+20.0`) \
`DX` — sample spacing along the strip in metres (default: `0.5`, matching AHN5 native resolution) \
`SMOOTHENING_FACTOR` — spline smoothing factor (`0` = interpolating; higher = smoother) \
Can also be run from the command line: `python AHN5.py --input-file <path> --output-location <path> --timestamp <t> [--gps_topic <topic>] [--plot]`

**Output** \
Two compressed `.npz` files saved to `OUTPUT_PATH/<date>/<time>/<int(TIME)>/`, one per AHN5 layer: \
`AHN5_DSM_<TIME>.npz` and `AHN5_DTM_<TIME>.npz`, each containing: \
`z` — elevation profile relative to the bike position (m) \
`s` — distance along the strip (m) \
`kappa` — curvature profile (1/m) \
`slope_deg` — slope profile (degrees) \
`method` — `"AHN5 DSM"` or `"AHN5 DTM"`

---

## EKF.py

This script computes vertical terrain curvature from IMU and GPS sensor data using an Extended Kalman Filter for speed estimation. It reads pose and angular velocity from a ZED `.svo`/`.svo2` recording or an MCAP file (SVO inputs are converted to a cached MCAP slice automatically), and reads GPS separately from an MCAP. Speed is fused from ZED positional tracking and GPS Doppler speed using a 1-D Kalman filter. The elevation profile is reconstructed by integrating the pitch angle over distance. Curvature is computed using two methods: Frenet curvature on a spline fit of the elevation profile, and direct pitch differentiation. Horizontal path curvature (ω_z / v) is also computed. Results are saved as `.npz` files for `Validation_main.py`.

**Input** \
`IMU_INPUT_PATH` — ZED `.svo`/`.svo2` recording or an `.mcap` file with `/zed/pose` and `/zed/imu/data` topics \
`GPS_FILE_PATH` — an `.mcap` recording containing a GPS topic

**Adjustable parameters** \
`TIME` — reference Unix timestamp (must match the timestamp used in the other validation scripts) \
`OUTPUT_PATH` — base folder for output \
`POSE_TOPIC` / `IMU_TOPIC` / `GPS_TOPIC` — ROS topic names \
`WIN_BEFORE` / `WIN_AFTER` — metres before and after the reference point to include in the profile \
`MIN_SPEED` — speed threshold below which `kappa_path` is set to NaN (default: `0.5` m/s) \
`SMOOTH_W` — Savitzky-Golay smoothing window for pitch in samples (default: `21`; set to `0` to disable) \
`SPLINE_SMOOTH` — spline smoothing factor for the elevation profile (default: `0.0`) \
`EKF_P0_V` / `EKF_P0_A` — initial Kalman filter variance on speed and acceleration \
`EKF_R_ZED` / `EKF_R_GPS` — measurement noise variance for ZED and GPS speed observations \
Can also be run from the command line: `python EKF.py --imu-file <path> --gps-file <path> --output-location <path> --timestamp <t> [--gps_topic <topic>] [--plot]`

**Output** \
Two compressed `.npz` files saved to `OUTPUT_PATH/<date>/<time>/<int(TIME)>/`: \
`EKF_curvature_validation_<TIME>.npz` — curvature profiles: `kappa_terrain_spline`, `kappa_terrain_pitch`, `kappa_path` \
`Z_positional_tracking_<TIME>.npz` — elevation profile from ZED positional tracking

---

## KISS ICP.py

This script extracts a vertical elevation profile along the bike's trajectory using KISS-ICP LiDAR odometry. It either loads pre-computed SE3 poses from a cached `poses.npy` file or runs KISS-ICP on the full MCAP recording to compute them. The elevation profile is extracted for a window around the reference timestamp by querying the trajectory at that position, and the result is saved as a `.npz` file for comparison in `Validation_main.py`. Computed poses are cached to disk so subsequent calls with different timestamps skip the odometry step.

**Input** \
An `.mcap` recording containing the LiDAR topic specified by `TOPIC` \
Or: pre-computed `poses.npy` and `timestamps.npy` in `KISS ICP results/<date> <time>/`

**Adjustable parameters** \
`DATASET_PATH` — path to the `.mcap` recording \
`TOPIC` — LiDAR ROS topic name for KISS-ICP (default: `/rslidar/M1P_deskewed`) \
`OUTPUT_PATH` — base folder for `.npz` output \
`TIME` — reference Unix timestamp \
`X_BACK` / `X_FRONT` — strip extent behind and in front of the bike in metres (default: `−5.0` / `+20.0`) \
`CUSTOM_PARAMETERS` — set to `True` to override individual KISS-ICP config values \
`LIVE_VISUALIZE` — show live odometry visualisation while KISS-ICP runs (slows processing significantly) \
Can also be run from the command line: `python "KISS ICP.py" --input-file <path> --output-location <path> --timestamp <t> [--plot]`

**Output** \
A compressed `.npz` file saved to `OUTPUT_PATH/<date>/<time>/<int(TIME)>/KISS_ICP_<TIME>.npz`, containing: \
`z` — elevation profile relative to the bike position (m) \
`s` — distance along the strip (m) \
`method` — `"KISS_ICP"`

---

## PCA_RANSAC_Height Derivative.ipynb

This notebook (Pipeline Stage 2) reads ground-only point cloud files from the Patchwork++ filter notebook and computes a forward 1-D road elevation and curvature profile using three methods: PCA (covariance-based plane fitting per bin), RANSAC (consensus plane fitting per bin), and height-derivative (median height per bin differentiated twice numerically). All three share the same binning grid, smoothing, and differentiation parameters so results are directly comparable. A 2-D heatmap view is also produced for each method. Results are saved as `.npz` files in the shared format for `Validation_main.py`.

**Input** \
A nuScenes-format `(N, 5)` `float32` `.bin` file containing ground-only points from `Patchwork++ Filter.ipynb`

**Adjustable parameters** \
`INPUT_BIN_PATH` — path to the ground-only `.bin` file \
`ROI_X_RANGE` / `ROI_Y_RANGE` — forward strip region of interest (default: `(−5.0, 15.0)` / `(−1.5, 1.5)`) \
`PROFILE_BIN_SIZE` — forward bin width in metres (default: `0.5`) \
`PROFILE_MIN_PTS` — minimum points per bin required to attempt a plane fit \
`PROFILE_MEDIAN_WINDOW` / `PROFILE_SMOOTH_WINDOW` — two-stage smoothing windows \
`RANSAC_N_ITER` / `RANSAC_DIST_THRESH` / `RANSAC_MIN_INLIERS` — RANSAC tuning parameters \
`HEIGHT_YLIM` / `SLOPE_YLIM` / `KAPPA_YLIM` — fixed y-axis limits for plots (set to `None` for auto-scaling)

**Output** \
Three compressed `.npz` files saved to `D:\Validation_results\<date>\<time>\`, one per method: \
`PCA_<timestamp>.npz`, `RANSAC_<timestamp>.npz`, `height-deriv_<timestamp>.npz` \
Each containing: `z`, `s`, `kappa`, `slope_deg`, `method`, and associated arrays

Example output:

![PCA](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/St.1.PW.PCA.png)

![RANSAC](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/St.1.PW.RANSAC.png)

![HD](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/St.1.PW.HD.png)

---

## IRLS.ipynb
The IRLS method works by fitting a quadratic plane iteratively to a pointcloud. It starts with a random plane in the pointcloud, then the residuals between that plane and the points are calculated before a new plane is fitted. That iterates a certain amount of times until the best plane with the smallest residuals is fitted.

**Useage**
The full pipeline is explained within the jupyter notebook.
Outputs of the quadratic plane fitting looks something like this:
![IRLS Plane Fitting](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/IRLS_quadratic_plane_fitting.png)

After the plane is fitted, further calculations are done. An example of the output:
![Calculation via IRLS](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/Calculation_via_IRLS.png)
