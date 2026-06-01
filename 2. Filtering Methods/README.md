# File context and how to use them
Every file's context and usage will be explained briefly.

---

## Object Detection Filter.py
This file explores the method of 3D object detection to perform ground segmentation. Using Stereoloab's ZED cameras and object detection model; cars, cyclists and pedestrians can are detected and filtered out of a pointcloud.

**Useage (LARS)**
Hier komt de uitleg over hoe de code werkt. Denk aan: welk file format gaat hierin, zijn er parameters die je zelf kan instellen?, wat gebeurt er doorheen de code, etc.

---




## csf_ground_segmentation.ipynb (+cloth_nodes)
This file uses a cloth simulation filter, CSF in short. The pointcloud is inverted and a simulated cloth is layed op top of the surface. All points within a certain threshold of that cloth will be taken as ground.

**Useage (Rayan)**
Hier komt de uitleg over hoe de code werkt. Denk aan: welk file format gaat hierin, zijn er parameters die je zelf kan instellen?, wat gebeurt er doorheen de code, etc.

---



## filter_ground_points_with_patchworks++.ipynb
This file uses a known ground segmentation method known as Patchwork++, https://arxiv.org/abs/2207.11919. 

**Useage (Jonas)**
Hier komt de uitleg over hoe de code werkt. Denk aan: welk file format gaat hierin, zijn er parameters die je zelf kan instellen?, wat gebeurt er doorheen de code, etc.

---







