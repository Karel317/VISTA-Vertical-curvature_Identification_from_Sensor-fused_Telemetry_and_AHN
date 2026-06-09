# File context and how to use them
Every file's context and usage will be explained briefly.

---

## Object Detection Filter.py

This script applies 3D semantic segmentation using the RandLA-Net neural network to classify every point in a point cloud as ground or non-ground. Non-ground points are then clustered into individual objects using DBSCAN, and an oriented 3D bounding box is drawn around each cluster. The final output is a ground-only point cloud saved as a `.ply` file. The pre-trained Semantic3D model weights are downloaded automatically on the first run if the checkpoint file is not found.

**Input** \
A single point cloud file. Supported formats: `.bin` (KITTI/Velodyne), `.mcap` (ROS bag), `.ply`, `.pcd`, `.las`/`.laz`, `.bag` (ROS1)

**Adjustable parameters** \
`INPUT_FILE` — path to the input point cloud file \
`LIDAR_TOPIC` — for `.mcap`/`.bag`: the ROS topic name to read (set to `None` to list all available topics) \
`FRAME_INDEX` — frame number to use from the bag (`0` = first frame; `"all"` = merge all frames into one cloud) \
`GROUND_LABELS` — Semantic3D class IDs to treat as ground (default: `[1, 2]` = man-made terrain + natural terrain) \
`DBSCAN_EPS` — clustering neighbourhood radius in metres (default: `0.5`) \
`DBSCAN_MIN_POINTS` — minimum points needed to form a DBSCAN cluster (default: `20`) \
`MIN_CLUSTER_POINTS` — minimum cluster size to draw a bounding box for (default: `30`) \
`CHECKPOINT_PATH` — path to the pre-trained RandLA-Net `.pth` weights (downloaded automatically on first run if not found) \
`OUTPUT_SUFFIX` — suffix appended to the input filename for the output (default: `_ground_only.ply`)

**Output** \
A `.ply` file saved alongside the input file containing only the classified ground points. During the run, four sequential interactive 3D viewer windows open: raw cloud coloured by height, detected objects with bounding boxes, red (removed) vs green (kept) comparison overlay, and the final ground-only cloud.

Example of how it filters:

![Object Detectioin](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/Object%20Detection.jpeg)

![Object Detection Points](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/Object%20Detection%20Points.jpeg)

---

## CSF Filtering.ipynb (+cloth_nodes)
This file uses a cloth simulation filter, CSF in short. The pointcloud is inverted and a simulated cloth is layed op top of the surface. All points within a certain threshold of that cloth will be taken as ground.

**Usage** \
This file uses ground segmentation on a single from from a ROS2.mcap recording, using the cloth simulation filter (CSF) algorithm. CSF works by inverting the point cloud upside down and draping a simulated cloth over the surface. All points withn a set distance threshold are classified as ground points.

**Input** \
The notebook needs a .mcap file with the following topics: \
/rslidar/M1P_deskewed \
/rslidar/helios_R \
/rslidar/helios_L 

**Adjustable parameters** \
Input file - path to the .mcap file \
Frame_index - the frame you want to process from the .mcap file \
If you want to process a specific moment in time rather than a frame number, set USE_TIMESTAMP = True in Cell 1b and fill in TARGET_TIMESTAMP with a timestamp copied from Foxglove. The corresponding frame number will be determined automatically.

The CSF parameters in Cell 1 can be adjusted depending on surroundings and quality of the point cloud: \
CSF_CLOTH_RESOLUTION — grid size of the simulated cloth (default: 0.3 m) \
CSF_SLOPE_SMOOTH — smooth slopes (True) or steep terrain (False) \
CSF_THRESHOLD — maximum distance from the cloth to count as ground (default: 0.1 m) \
The width of the strip on which CSF is applied is adjustable in Cell 6 via X_RANGE and Y_RANGE. 

**Output** \
Results are saved to the folder set via OUTPUT_DIR in Cell 1: \
frameXXX_ground.ply — ground points \
frameXXX_merged.ply — full fused point cloud \
frameXXX_ground.bin — ground points in KITTI format \
frameXXX_topview.png / frameXXX_sideview.png / frameXXX_bev_comparison.png — visualisations

Example of how it filters:

![CSF method](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/csf.png)

![CSF Filter](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/CSF%20Filtering.png)

---

## Patchwork++ Filter.ipynb

This notebook (Pipeline Stage 1) applies the Patchwork++ ground segmentation algorithm to one or more LiDAR frames from an MCAP recording. Patchwork++ divides the area around the sensor into concentric zones and angular wedges and fits a local ground plane to each wedge using the lowest points, classifying all points within a set distance threshold as ground. The bag index and sensor calibration tree from `/tf_static` are built once and reused across all timestamps in the batch, so adding more frames is fast. An optional GLE eigenvalue post-filter can be applied after Patchwork++ to discard ground patches that are too rough or too steeply tilted. The ground-only output files are the direct input to `PCA_RANSAC_Height Derivative.ipynb`.

**Input** \
An `.mcap` recording with one or more `sensor_msgs/PointCloud2` topics and `/tf_static` for sensor calibration. Also supports nuScenes `.bin` and KITTI `.bin` files via the `WORKFLOW` setting.

**Adjustable parameters** \
`WORKFLOW` — `"mcap"` (SenseBike default), `"bin"` (nuScenes), or `"kitti"` \
`MCAP_PATH` — path to the `.mcap` recording \
`LIDAR_TOPICS` — list of ROS topic names to load and merge \
`MCAP_FRAME_TIMESTAMPS` — list of Unix timestamps to process in one batch (bag is indexed once) \
`MCAP_FRAME_INDICES` — alternative to timestamps: list of frame indices to process \
`SENSOR_CONFIG` — `sensor_height` (LiDAR height above ground in metres), `min_range`, `max_range` \
`GROUND_BIN_DIR` — output folder for the `.bin` files \
`EXPORT_RAW_CLOUD` / `EXPORT_GROUND` / `EXPORT_FULL_CLOUD` — toggle which of the three output files to write \
`APPLY_EIGEN_FLATNESS_FILTER` — enable the optional GLE post-filter (off by default; can remove genuine curvature signal) \
`POST_CELL_SIZE`, `FLATNESS_THRESHOLD`, `UPRIGHTNESS_THRESHOLD` — GLE post-filter tuning parameters \
`PLOT_EACH_FRAME` — show a before/after visualisation for every processed timestamp

**Output** \
Per timestamp, up to three nuScenes-format `(N, 5)` `float32` `.bin` files saved to `GROUND_BIN_DIR`: \
`*_raw.bin` — full unfiltered scene before Patchwork++ (for before/after comparison in Foxglove) \
`*_pw.bin` — ground points only after Patchwork++ (direct input to `PCA_RANSAC_Height Derivative.ipynb`) \
`*_pw_full.bin` — full scene with intensity `0.0` = ground / `1.0` = obstacle (Foxglove inspection)

Example of Filtering:

![PW++ Filter](https://github.com/Karel317/LiDAR_based_vertical_curvature_calculation/blob/main/Pictures/Filtering_readme.png)
