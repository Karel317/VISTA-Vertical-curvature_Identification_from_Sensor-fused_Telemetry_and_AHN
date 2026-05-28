# File context and how to use them
Every file's context and usage will be explained briefly.

---

## Object Detection Filter.py
This file explores the method of 3D object detection to perform ground segmentation. Using Stereoloab's ZED cameras and object detection model; cars, cyclists and pedestrians can are detected and filtered out of a pointcloud.

**Usage (LARS)**
Hier komt de uitleg over hoe de code werkt. Denk aan: welk file format gaat hierin, zijn er parameters die je zelf kan instellen?, wat gebeurt er doorheen de code, etc.

---




## csf_ground_segmentation.ipynb (+cloth_nodes)
This file uses a cloth simulation filter, CSF in short. The pointcloud is inverted and a simulated cloth is layed op top of the surface. All points within a certain threshold of that cloth will be taken as ground.

**Usage (Rayan)**
Hier komt de uitleg over hoe de code werkt. Denk aan: welk file format gaat hierin, zijn er parameters die je zelf kan instellen?, wat gebeurt er doorheen de code, etc.

usage
This file uses ground segmentation on a single from from a ROS2.mcap recording, using the cloth simulation filter (CSF) algorithm. CSF works by inverting the point cloud upside down and draping a simulated cloth over the surface. All points withn a set distance threshold are classified as ground points.

input
The notebook needs a .mcap file with the following topics:
/rslidar/M1P_deskewed
/rslidar/helios_R
/rslidar/helios_L

adjustable parameters
Input file - path to the .mcap file
Frame_index - the frame you want to process from the .mcap file
If you want to process a specific moment in time rather than a frame number, set USE_TIMESTAMP = True in Cell 1b and fill in TARGET_TIMESTAMP with a timestamp copied from Foxglove. The corresponding frame number will be determined automatically.

The CSF parameters in Cell 1 can be adjusted depending on surroundings and quality of the point cloud:
CSF_CLOTH_RESOLUTION — grid size of the simulated cloth (default: 0.3 m)
CSF_SLOPE_SMOOTH — smooth slopes (True) or steep terrain (False)
CSF_THRESHOLD — maximum distance from the cloth to count as ground (default: 0.1 m)
The width of the strip on which CSF is applied is adjustable in Cell 6 via X_RANGE and Y_RANGE. 

Output
Results are saved to the folder set via OUTPUT_DIR in Cell 1:
frameXXX_ground.ply — ground points
frameXXX_merged.ply — full fused point cloud
frameXXX_ground.bin — ground points in KITTI format
frameXXX_topview.png / frameXXX_sideview.png / frameXXX_bev_comparison.png — visualisations


---



## filter_ground_points_with_patchworks++.ipynb
This file uses a known ground segmentation method known as Patchwork++, https://arxiv.org/abs/2207.11919. 

**Useage (Jonas)**
Hier komt de uitleg over hoe de code werkt. Denk aan: welk file format gaat hierin, zijn er parameters die je zelf kan instellen?, wat gebeurt er doorheen de code, etc.

---







