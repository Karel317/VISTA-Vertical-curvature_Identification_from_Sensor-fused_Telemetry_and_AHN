# LiDAR Ground Segmentation with Vertical Curvature (MET CLAUDE GESCHREVEN EN LOET NOG BEKEKEN EN AANGEPAST WORDEN)

A LiDAR-based ground segmentation pipeline that builds a dataset with vertical curvature and accurately identifies that curvature in 3D point clouds.

This repository accompanies a Bachelor's End Thesis conducted at **Delft University of Technology**, Department of **Cognitive Robotics**.

---

## 📖 Overview

Traditional ground segmentation methods often assume a flat or near-flat ground plane, which fails on terrain with vertical curvature (hills, slopes, road crests, valleys, etc.). This project addresses that gap by:

- Building a labeled LiDAR dataset that explicitly includes vertical curvature
- Developing and evaluating a segmentation method that identifies curved ground accurately
- Providing reproducible scripts and notebooks for analysis and visualization

> _TODO: Add a 1–2 sentence summary of your specific contribution / main result._

---

## 📁 Repository Structure

```
lidar-ground-segmentation/
├── notebooks/          # Jupyter notebooks for exploration and visualization
├── src/                # Python source code (segmentation, utils, etc.)
├── data/               # (Not included) place dataset here — see Dataset section
├── results/            # Output figures, segmented clouds, evaluation metrics
├── requirements.txt    # Python dependencies
└── README.md
```

> _TODO: Replace this tree with your actual folder layout. List each main file with a one-line description._

---

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

> _TODO: If you don't have a `requirements.txt` yet, generate one with `pip freeze > requirements.txt` from your working environment. Key libraries are likely: numpy, open3d, matplotlib, scikit-learn, pandas, jupyter._

---

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

### Running the notebooks
Launch Jupyter and open any notebook in the `notebooks/` folder:
```bash
jupyter notebook
```

### Running the scripts
> _TODO: Replace these examples with your real commands._

Example — segment a single point cloud:
```bash
python src/segment.py --input data/raw/cloud_001.pcd --output results/cloud_001_seg.pcd
```

Example — evaluate on the full dataset:
```bash
python src/evaluate.py --data-dir data/ --output results/metrics.csv
```

---

## 📊 Results

> _TODO: Briefly describe what the pipeline outputs — segmented point clouds, accuracy/IoU metrics, visualizations. Optionally include a sample figure: `![Example](results/example.png)`._

---

## 🔬 Method

> _TODO: 2–4 sentences on the approach. E.g. "The pipeline first voxelizes the input cloud, then applies [method X] to estimate ground curvature, and finally classifies each point using [classifier Y]." Reference the thesis for the full description._

For the full methodology, see the thesis report: `[link or filename of thesis PDF]`.

---

## 👤 Author

**Karel** — Bachelor's End Thesis  
Department of Cognitive Robotics, Delft University of Technology

Supervisor: _TODO: supervisor name_

---

## 📄 License

> _TODO: Pick one. Common choices for academic code: MIT (permissive), or "All rights reserved" if you'd rather not allow reuse. If unsure, MIT is a safe default — just add a `LICENSE` file with the MIT text._

---

## 🙏 Acknowledgments

> _TODO: Optional. Thank your supervisor, the Cognitive Robotics department, anyone who provided data or code you built on top of, etc._
