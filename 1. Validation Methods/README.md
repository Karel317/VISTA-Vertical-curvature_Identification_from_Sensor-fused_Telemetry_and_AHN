Files explained:

# Main file:
- validation_main.py [NOT DONE]
  Script imports results from .npz files calculates validation metrics

---

# Others:
- **extract_pointcloud_segment.py** [NOT DONE]
  Script to extract a point cloud segment of a full point cloud based
  on begin and end time

- **AHN4_validation_strip.ipynb** [DONE]
  Script to get data from AHN4 height map based on gps coordinates at a 
  single timestamp. The strip is only calculated -5m to 25m in front of the
  gps not the actual route.

  - **Validation_IMU_GPS.ipynb** [NOT DONE]
  Gets data from IMU and GPS to calculate global route and height. It also
  calculates the height at a timestamp to 25m in front of the gps. (The actual
  route not a strip). 

  - **IMU_validation.ipynb** [NOT USED]
  Compares the 4 IMU measurements against each other.

  - **IMU test.py** [NOT USED]
  Was the original script before ipynb, now not used.
