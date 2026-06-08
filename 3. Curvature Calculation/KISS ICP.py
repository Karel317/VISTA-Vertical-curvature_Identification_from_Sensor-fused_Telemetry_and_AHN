import os
import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from kiss_icp.pipeline import OdometryPipeline
from kiss_icp.datasets import dataset_factory

# Disable the "latest" symlink kiss_icp tries to create — it needs extra
# permissions on Windows and we don't use it. (from run_kiss_icp.py)
os.symlink = lambda *_: None

# ============================================================
# SETTINGS — edit these before running
# ============================================================

# Default LiDAR mcap (overridable via --input-file / pipeline input_file)
DATASET_PATH = Path(r"C:\Users\Leons\OneDrive - Delft University of Technology\BEP\Users\kuitbrug_Users_OneDrive - Delft University of Technology.mcap")
TOPIC        = "/rslidar/M1P_deskewed"        # LiDAR topic for KISS-ICP

OUTPUT_PATH  = r"D:\Validation_results"       # where the .npz profile is saved

TIME         = 1779439917.972930459           # reference Unix timestamp

# Strip geometry — extent along the trajectory, relative to the bike at TIME
X_BACK       = -5.0                           # m behind the bike
X_FRONT      = 20.0                           # m in front

METHOD       = "KISS_ICP"                     # .npz `method` field + filename prefix

# KISS-ICP run flags (only used when poses must be (re)computed)
SHORTENED         = True   # how to derive the <date> <time> results folder name
CUSTOM_PARAMETERS = False  # override individual kiss_icp.yaml params below
LIVE_VISUALIZE    = False  # live odometry visualisation (slows processing)

SCRIPT_DIR        = Path(__file__).parent
KISS_RESULTS_ROOT = SCRIPT_DIR.parent / "KISS ICP" / "KISS ICP results"
YAML_PATH         = SCRIPT_DIR.parent / "KISS ICP" / "kiss_icp.yaml"

# ── Helper functions ─────────────────────────────────────────────────────────

def _results_dir_for(dataset_path):
    """Derive the 'KISS ICP results/<date> <time>' folder from the dataset path.

    Mirrors run_kiss_icp.py: SHORTENED controls how date/time are sliced out of
    the path.
    """
    dataset_path = Path(dataset_path)
    if SHORTENED:
        date = dataset_path.parts[-2]      # e.g. "2026_05_22"
        time = dataset_path.stem[-8:]      # last 8 chars e.g. "10_50_00"
    else:
        date = dataset_path.parts[-5]
        time = dataset_path.parts[-3]
    return KISS_RESULTS_ROOT / f"{date} {time}"


def _load_or_run_kiss(dataset_path, topic, results_dir):
    """Load cached poses/timestamps if present, otherwise run KISS-ICP.

    Returns (poses (N,4,4), timestamps (N,)). On a cache miss the pipeline is
    run (a few minutes) and the results are saved for next time.
    """
    poses_path = results_dir / "poses.npy"
    times_path = results_dir / "timestamps.npy"

    if poses_path.exists() and times_path.exists():
        poses      = np.load(poses_path)
        timestamps = np.load(times_path)
        print(f"Loaded {len(poses)} cached poses from {results_dir}/")
        return poses, timestamps

    print(f"No cached poses in {results_dir}/ — running KISS-ICP (this takes a few minutes)…")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(YAML_PATH, "r") as f:
        config = yaml.safe_load(f)

    if CUSTOM_PARAMETERS:
        config["data"]["deskew"]    = False
        config["data"]["max_range"] = 30.0
        config["data"]["min_range"] = 3.0
        config["mapping"]["voxel_size"]           = 0.15
        config["mapping"]["max_points_per_voxel"] = 5
        config["registration"]["max_num_iterations"]    = 500
        config["registration"]["convergence_criterion"] = 0.0001
        config["adaptive_threshold"]["min_motion_th"]   = 0.01

    # Write the (possibly modified) config to a temp file for the pipeline
    active_config = SCRIPT_DIR / "_active_config.yaml"
    with open(active_config, "w") as f:
        yaml.dump(config, f)

    dataset = dataset_factory(
        dataloader="mcap",
        data_dir=Path(dataset_path),
        topic=topic,
    )
    pipeline = OdometryPipeline(
        dataset=dataset,
        config=active_config,
        visualize=LIVE_VISUALIZE,
    )
    pipeline.run()
    active_config.unlink(missing_ok=True)

    poses      = np.array(pipeline.poses)   # (N, 4, 4) SE3
    timestamps = np.array(pipeline.times)   # (N,) seconds

    np.save(poses_path, poses)
    np.save(times_path, timestamps)
    print(f"Saved {len(poses)} poses to {results_dir}/")
    return poses, timestamps


def _extract_profile(poses, timestamps, timestamp):
    """Extract the z-profile along the trajectory from X_BACK to X_FRONT.

    Returns (s_rel, z_rel, ref_idx, trajectory) where s_rel is cumulative
    arc-length re-zeroed at the pose nearest `timestamp`, and z_rel is the
    height relative to the bike over that window.
    """
    trajectory = poses[:, :3, 3]                 # (N, 3)  x, y, z
    x, y, z    = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    t          = np.asarray(timestamps)

    ref_idx = int(np.argmin(np.abs(t - timestamp)))
    dt = abs(t[ref_idx] - timestamp)
    print(f"Using TIME = {timestamp}, closest pose Δt = {dt:.3f}s (index {ref_idx})")

    # Cumulative arc length along the (x, y) ground track, re-zeroed at the bike
    seg   = np.hypot(np.diff(x), np.diff(y))
    cum_s = np.concatenate([[0], np.cumsum(seg)])
    cum_s = cum_s - cum_s[ref_idx]

    # Window [X_BACK, X_FRONT] along the trajectory. argmax/argmin on a boolean
    # array returns 0 when nothing matches, so fall back to the array ends when
    # the requested extent runs past the available trajectory.
    front_hits = cum_s >= X_FRONT
    idx_front  = int(np.argmax(front_hits)) if front_hits.any() else len(cum_s) - 1
    back_hits  = cum_s <= X_BACK
    idx_back   = int(np.where(back_hits)[0][-1]) if back_hits.any() else 0

    s_rel = cum_s[idx_back:idx_front + 1]
    z_rel = z[idx_back:idx_front + 1] - z[ref_idx]

    return s_rel, z_rel, ref_idx, trajectory


def _save_output(timestamp, s_rel, z_rel, results_dir):
    """Save the z-profile as a .npz, matching the shared save convention.

    Layout: OUTPUT_PATH / <date> / <time> / <int(TIME)> / KISS_ICP_<int(TIME)>.npz
    so validation_main.py picks it up. date/time are derived from the timestamp
    (datetime.fromtimestamp), the same as lidar_pipeline_unified and the other
    validation methods — so all results land in the same folder.
    """
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    save_dir = os.path.join(OUTPUT_PATH, dt.strftime("%Y_%m_%d"),
                            dt.strftime("%H_%M_%S"), str(int(timestamp)))
    os.makedirs(save_dir, exist_ok=True)

    fpath = os.path.join(save_dir, f"{METHOD}_{int(timestamp)}.npz")
    np.savez_compressed(
        fpath,
        x      = None,
        y      = None,
        z      = z_rel,
        s      = s_rel,
        t      = np.array([timestamp]),
        method = np.array([METHOD]),
    )
    print(f"Saved {METHOD} → {fpath}")
    return fpath


def _plot(trajectory, timestamps, ref_idx, s_rel, z_rel):
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    t       = np.asarray(timestamps)

    fig = plt.figure(figsize=(20, 10))
    gs  = fig.add_gridspec(2, 3)

    # --- top-left: top-down trajectory ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, y, lw=0.8, color="tab:blue", label="trajectory")
    ax1.plot(x[ref_idx], y[ref_idx], "s", color="red", ms=8, label="bike at TIME")
    ax1.set_title("Top-down trajectory (x/y)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- top-middle: z over time ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, z, lw=0.8, color="coral")
    ax2.axvline(t[ref_idx], color="red", lw=0.7, ls="--", label="TIME")
    ax2.set_title("Z height over time")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("z (m)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- top-right: 3D trajectory ---
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    ax3.plot(x, y, z, lw=0.8)
    ax3.set_title("3D trajectory")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")

    # --- bottom: extracted z-profile along the strip ---
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(s_rel, z_rel, "o-", color="tab:green", ms=3, label=f"{METHOD} profile")
    ax4.axvline(0, color="blue", lw=0.7, ls="--", label="bike at s=0")
    ax4.axhline(0, color="black", lw=0.5)
    ax4.set_title("KISS-ICP ground profile")
    ax4.set_xlabel("s (forward) [m]")
    ax4.set_ylabel("Elevation relative to bike [m]")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("KISS-ICP validation strip", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── Main functions ─────────────────────────────────────────────────────────

def run(input_file, timestamp, plot=False, output_path=OUTPUT_PATH):
    global OUTPUT_PATH
    OUTPUT_PATH = output_path
    timestamp = float(timestamp)

    # Config constant + override: only use input_file if it's a real .mcap on
    # disk (the pipeline may pass a camera/GPS mcap that isn't the LiDAR file);
    # otherwise fall back to the module's DATASET_PATH.
    candidate = Path(input_file) if input_file else None
    if candidate is not None and candidate.suffix.lower() == ".mcap" and candidate.exists():
        dataset_path = candidate
    else:
        dataset_path = DATASET_PATH

    results_dir = _results_dir_for(dataset_path)
    poses, timestamps = _load_or_run_kiss(dataset_path, TOPIC, results_dir)
    s_rel, z_rel, ref_idx, trajectory = _extract_profile(poses, timestamps, timestamp)
    fpath = _save_output(timestamp, s_rel, z_rel, results_dir)
    if plot:
        _plot(trajectory, timestamps, ref_idx, s_rel, z_rel)
    return fpath


def _str2bool(val):
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "t", "yes", "y")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KISS-ICP curvature validation.")
    parser.add_argument("--input-file", default=str(DATASET_PATH),
                        help="LiDAR MCAP path (default: %(default)s)")
    parser.add_argument("--output-location", default=OUTPUT_PATH,
                        help="General output folder; a date/time/TIME subfolder is created (default: %(default)s)")
    parser.add_argument("--timestamp", type=float, default=TIME,
                        help="Reference Unix timestamp (default: %(default)s)")
    parser.add_argument("--plot", type=_str2bool, nargs="?", const=True, default=False,
                        help="Set true for plots (default: %(default)s)")
    args = parser.parse_args()
    run(args.input_file, args.timestamp, args.plot, output_path=args.output_location)
