# Dutch Hills Dataset: Identifying vertical curvature using the SenseBike, a sensor equipped bicycle

A LiDAR-based ground segmentation pipeline that builds a dataset with vertical curvature and accurately identifies that curvature in 3D point clouds.

This repository accompanies a Bachelor's End Thesis conducted at **Delft University of Technology**, Department of **Cognitive Robotics**.

---

## 📖 Overview

Traditional ground segmentation methods often assume a flat or near-flat ground plane, which fails on terrain with vertical curvature (hills, slopes, road crests, valleys, etc.). This project addresses that gap by:

- Building a LiDAR dataset that explicitly includes vertical curvature
- Applying filtering methods that identify curved ground accurately
- Providing reproducible scripts and notebooks for analysis and visualization

## 📁 Repository Structure

```
lidar-ground-segmentation/
├── 1. Validation/                  # Scripts that validate the curvature calculation from "3. Curvature Calculation"
├── 2. Filtering Methods/           # Ground segmentation methods
├── 3. Curvature Calculation/       # Calculation methods that identify and quantify vertical curvature
├── .gitattributes
├── .mcap_to_.bin.ipynb             # .mcap to .bin format converter
├── IMU SVO to MCAP.py              # .svo to .mcap format converter
├── README.md
├── environment.yml
└── requirements.txt                # pip requirements
```


## ⚙️ Installation

**Requirements:** Python 3.9+ recommended.

Clone the repository:
```bash
git clone https://github.com/Karel317/lidar-ground-segmentation.git
cd lidar-ground-segmentation
```

Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 📂 Dataset

The dataset is **not included** in this repository due to its size. Download it separately:

> _TODO: Add the dataset source — a link (Google Drive / Zenodo / institutional storage), or instructions for how to obtain it. Mention the format (.pcd, .bin, .las, .ply) and the sensor used (e.g. Velodyne VLP-16, Ouster OS1)._

After downloading, place the data in the `data/` folder so the structure looks like:
```
data/
├── raw/
│   └── <pointcloud files here>
└── labels/
    └── <label files here>
```

---

## 🚀 Usage

### Running the notebooks and python scripts
Launch Visual Studio Code and clone the repository there. 
Now run the desired files via VSCode.

---


## 📊 Results

> _TODO: Briefly describe what the pipeline outputs — segmented point clouds, accuracy/IoU metrics, visualizations. Optionally include a sample figure: `![Example](results/example.png)`._

---

## 🔬 Method
For the full methodology, see the thesis report: `[link or filename of thesis PDF]`.

---

## 👤 Authors

**Karel Peuskens, Jonas Repa, Rayan Ait Hadj Brahim, Lars Wissink, Leon Sinnesael** — Bachelor's End Thesis  
Department of Cognitive Robotics, Delft University of Technology

Supervisor: Dr. H. Caesar

---

