# File context and how to use them
Every file's context and usage will be explained briefly.

---

## LiDAR-based Calculation Pipeline.ipynb

End-to-end LiDAR pipeline that processes a batch of timestamps in a single run. Stage 1 applies two ground filters in parallel on the same MCAP frame per timestamp — Patchwork++ and CSF — and saves the resulting ground-only point clouds as `.bin` files. Stage 2 reads each `.bin` file and computes slope and curvature using four methods: PCA, RANSAC, height-derivative, and IRLS quadric. Each method × filter combination produces one `.npz` result file, giving up to eight files per timestamp. The bag index is built once and reused across all timestamps in the batch.

**Input** \
An `.mcap` recording with LiDAR `sensor_msgs/PointCloud2` topics and `/tf_static` for sensor calibration

**Adjustable parameters** \
`MCAP_PATH` — path to the `.mcap` recording \
`TIMESTAMPS` — list of Unix timestamps to process (generate these with `Random Frame Generator.ipynb`) \
`GROUND_BIN_DIR` — folder where ground-only `.bin` files are saved \
`RESULTS_DIR` — base folder for curvature `.npz` output \
`LIDAR_TOPICS` — list of ROS topic names to load and merge \
`PW_SENSOR_HEIGHT` / `PW_MIN_RANGE` / `PW_MAX_RANGE` — Patchwork++ ground plane fitting parameters \
`CSF_CLOTH_RESOLUTION` / `CSF_SLOPE_SMOOTH` / `CSF_THRESHOLD` / `CSF_X_RANGE` / `CSF_Y_RANGE` — CSF parameters \
`SENSOR_TRANSFORMS` — per-topic translation and RPY rotation for sensor-to-base_link calibration (used by CSF) \
`ROI_X_RANGE` / `ROI_Y_RANGE` — forward strip for Stage 2 curvature calculation \
`PROFILE_BIN_SIZE` / `PROFILE_MIN_PTS` / `PROFILE_MEDIAN_WINDOW` / `PROFILE_SMOOTH_WINDOW` — binning and smoothing \
`RANSAC_N_ITER` / `RANSAC_DIST_THRESH` — RANSAC parameters \
`IRLS_MAX_ITER` / `IRLS_C_TUKEY` — IRLS quadric parameters

**Output** \
Per timestamp, two ground-only `.bin` files in `GROUND_BIN_DIR`: \
`<stem>_patchwork.bin` — Patchwork++ ground points (nuScenes format) \
`<stem>_csf.bin` — CSF ground points (nuScenes format) \
Up to eight `.npz` files per timestamp in `RESULTS_DIR/<date>/<time>/<timestamp>/`: \
`PCA_patchwork_<ms>.npz`, `RANSAC_patchwork_<ms>.npz`, `height-deriv_patchwork_<ms>.npz`, `IRLS_patchwork_<ms>.npz` and the four `_csf` equivalents

---

## Non-LiDAR-based Calculation Pipeline.py

This script orchestrates the three non-LiDAR validation methods: AHN5, IMU/GPS (EKF), and KISS-ICP. It imports each module and calls its `run()` function with a shared configuration block. Modules can be toggled on or off via the `REGISTRY` dict or overridden at runtime with command-line flags. A summary is printed at the end listing which modules succeeded or failed.

**Input** \
`MAIN_MCAP` — path to the main `.mcap` recording (used by AHN5 and KISS-ICP) \
`IMU_INPUT` — ZED `.svo2` or `.mcap` file for the IMU/GPS (EKF) method

**Adjustable parameters** \
`TIME` — reference Unix timestamp (must be the same value across all three methods) \
`OUTPUT_PATH` — base folder for all `.npz` output files \
`GPS_TOPIC` — GPS ROS topic name \
`PLOT` — set to `True` to show plots (each window must be closed before the next module starts) \
`REGISTRY` — dict mapping module names to `(enabled, module_name, config_lambda)`; set `enabled = False` to skip a module \
`--only NAME [NAME ...]` — run only the named modules and skip all others \
`--skip NAME [NAME ...]` — run all enabled modules except the named ones \
`--list` — print the available module names and exit

**Output** \
Delegated to the individual validation scripts (AHN5, EKF, KISS ICP); each saves `.npz` files to `OUTPUT_PATH/<date>/<time>/<int(TIME)>/`

**Warning**
For the IMU_INPUT, specifically to convert .svo files to .mcap files, you need to install ZED SDK and CUDA. This is only possible if you have a NVIDIA GPU. The documentation and downloads can be found with the following link: https://www.stereolabs.com/developers/release. If CUDA isn't already on your machine, the installer at this link offers to download and install it (or you can install it yourself from NVIDIA's CUDA Toolkit archive). 

---

## Validation_main.py

This script loads `.npz` result files from validation methods (physical measurements, AHN5, EKF, KISS-ICP) and LiDAR calculation methods (PCA, RANSAC, height-derivative, IRLS) and compares them quantitatively. All profiles are interpolated to a common distance grid, optionally fine-aligned to the physical reference using a sliding-window search, and trimmed to the evaluation window before comparison. RMSE and MAE are computed for both elevation and curvature for every validation × calculation pair. Results are exported to CSV and visualised as overlaid elevation and curvature plots.

**Input** \
`.npz` files in `VALIDATION_DIR` whose filenames start with one of the entries in `VALIDATION_PREFIXES` \
`.npz` files in `CALCULATION_DIR` whose filenames start with one of the entries in `CALCULATION_PREFIXES`

**Adjustable parameters** \
`VALIDATION_DIR` / `CALCULATION_DIR` — folders containing the `.npz` result files \
`VALIDATION_PREFIXES` — list of filename prefixes for validation-method `.npz` files (e.g. `["Physical_meas_"]`) \
`CALCULATION_PREFIXES` — list of filename prefixes for calculation-method `.npz` files (e.g. `["PCA_", "RANSAC_"]`) \
`THRESHOLD` — height difference for the "fraction within threshold" metric (default: `0.1` m) \
`SMOOTHENING_FACTOR` — spline smoothing applied to all profiles before comparison (default: `0` = no smoothing) \
`START_VALIDATION` — distance in metres where the physical measurement begins to rise (profile alignment anchor) \
`WINDOW_BACK_M` / `WINDOW_FWD_M` — evaluation window size in metres around `START_VALIDATION` \
`ALIGN_TO_REFERENCE` — align all profiles to the physical measurement reference before computing metrics \
`FINE_ALIGN` — enable per-method fine alignment using a sliding-window search \
`FINE_ALIGN_SEARCH_M` / `FINE_ALIGN_STEP_M` — search range and step size for fine alignment

**Output** \
A `.csv` file at `C:\Users\karel\OneDrive\BEP\<timestamp>.csv` with RMSE, MAE, and fraction-in-threshold for every method pair \
Plots: elevation profile overlay, curvature profile overlay (log scale), elevation difference plot

The output .csv files can be merged by using the merge_csvs.py in Useful scripts. The script merges all the files to one .csv file containing all the data. 

---

## Random Frame Generator.ipynb

This utility notebook generates a list of random Unix timestamps distributed uniformly within the first five minutes of a recording. The date and time are parsed automatically from the recording's file path. The output is a ready-to-paste Python list called `TIMESTAMPS` for use in `LiDAR-based Calculation Pipeline.ipynb`.

**Adjustable parameters** \
`file_path` — path to a `.mcap` recording; the date and time are extracted from the path structure `<date>/Rosbag/<time>/rosbag/rosbag_0.mcap` \
`aantal_timestamps` — number of random timestamps to generate

**Output** \
A sorted Python list of Unix timestamp floats printed to the notebook output, ready to paste into the pipeline notebook

**Important Note** \
This script is unfinished. One of the research goals was to detect curvature over full .mcap files, but that was cut short due to time constraints. Here is how we planned to build that script:

From the validation of curvature quantification, CSF with Height derivative turned out to perform the best. To test whether this method is also useful to detect curvature, random timestamps within a .mcap file are generated. The method would then run on these timestamps and classify it as LOW/MEDIUM/HIGH curvature:

Low -> mostly flat (0-|0.05| 1/m) \
Medium -> small hills/minor bridges (|0.051|-|0.1| 1/m) \
High -> bridges, bigger hills (>|0.1| 1/m) \
(NOTE: this is specifically for Delft's surroundings, other areas might have a very different classification)

The model would then output how many times it detects curvature and where. A researcher takes that information and checks it manually in foxglove to determine the successrate of the model.

