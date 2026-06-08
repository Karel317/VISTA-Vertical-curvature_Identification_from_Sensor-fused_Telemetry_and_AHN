import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS — edit these before running
# ============================================================
# TODO Delete timestamp


OUTPUT_PATH = r"D:\Validation_results"         # File path where the .npz profiles get saved to

# The CSV lives in the parent "1. Validation Methods" folder; this file is in Python_files/
DEFAULT_CSV_FILE = Path(__file__).parent.parent / "Physical_meas_data.csv"

BACKWARDS = True                              # If we biked backwards on the same slope we need to look at the slope the other way
TIME          = 1779439914.817100912           # reference Unix timestamp (sets the output folder)
STEP_M        = 0.222                          # m between consecutive measurement points
METHOD_PREFIX = "Physical_meas"                # validation_main detects via "Physical_meas" in key

# ── Helper functions ─────────────────────────────────────────────────────────

def _read_csv(csv_path):
    """Read the header labels and the three height columns from the CSV."""
    with open(csv_path, "r") as f:
        labels = [h.strip() for h in f.readline().split(",")]
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=(0, 1, 2))
    return labels, data


def _slugify(label):
    """Make a filesystem/key-safe method name from a bridge label."""
    return f"{METHOD_PREFIX}_" + "_".join(label.split())


def _build_profiles(labels, data, backward):
    """Build one profile per bridge column.

    Ports Physical_meas.py per-column math: subtract first reading, drop NaNs,
    z = -cumsum (cm), s = index * STEP_M (m). Converts z to METERS.
    """
    profiles = {}
    for i in range(data.shape[1]):
        col = data[:, i] - data[0, i]
        col = col[~np.isnan(col)]
        z_cm = -np.cumsum(col)
        z_m  = z_cm / 100.0                     # cm → m
        if backward:                           # biked backwards: reverse so end z becomes first, first z becomes last
            z_m = z_m[::-1]
        s    = np.arange(len(z_m)) * STEP_M
        profiles[_slugify(labels[i])] = {"s": s, "z": z_m}
    return profiles


def _save_output(timestamp, profiles):
    # Output folder is derived from the timestamp (datetime.fromtimestamp), the
    # same convention as lidar_pipeline_unified and the other validation methods,
    # so all results land in OUTPUT_PATH/<date>/<time>/<int(timestamp)>.
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    save_dir = os.path.join(OUTPUT_PATH, dt.strftime("%Y_%m_%d"),
                            dt.strftime("%H_%M_%S"), str(int(timestamp)))
    os.makedirs(save_dir, exist_ok=True)

    paths = []
    for method, p in profiles.items():
        fpath = os.path.join(save_dir, f"{method}_backwards_{int(timestamp)}.npz")
        np.savez_compressed(
            fpath,
            x      = np.array([]),
            y      = np.array([]),
            z      = p["z"],
            s      = p["s"],
            t      = np.array([timestamp]),
            method = np.array([method]),
        )
        print(f"Saved {method} -> {fpath}")
        paths.append(fpath)
    return paths


def _plot(profiles):
    fig, ax = plt.subplots(figsize=(12, 6))
    for method, p in profiles.items():
        ax.plot(p["s"], p["z"], "o-", ms=3, label=method)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Physical measurement ground profiles")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Cumulative height relative to first measurement (m)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Main functions ─────────────────────────────────────────────────────────

def run(input_file, timestamp, backward, plot=False, output_path=OUTPUT_PATH):
    global OUTPUT_PATH
    OUTPUT_PATH = output_path
    timestamp = float(timestamp)

    # The physical data is the fixed CSV. Use input_file only if it's actually a
    # CSV on disk (the pipeline passes an MCAP here, which we ignore); otherwise
    # fall back to DEFAULT_CSV_FILE. The timestamp sets the output folder.
    candidate = Path(input_file) if input_file else None
    if candidate is not None and candidate.suffix.lower() == ".csv" and candidate.exists():
        csv_path = candidate
    else:
        print("INput file not found, using default file")
        csv_path = DEFAULT_CSV_FILE

    labels, data = _read_csv(csv_path)
    profiles = _build_profiles(labels, data, backward)
    paths = _save_output(timestamp, profiles)
    if plot:
        _plot(profiles)
    return paths


def _str2bool(val):
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "t", "yes", "y")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physical measurement curvature validation.")
    parser.add_argument("--input-file", default=str(DEFAULT_CSV_FILE),
                        help="Physical-measurement CSV (default: %(default)s)")
    parser.add_argument("--output-location", default=OUTPUT_PATH,
                        help="General output folder; a date/time/timestamp subfolder is created (default: %(default)s)")
    parser.add_argument("--timestamp", type=float, default=TIME,
                        help="Reference Unix timestamp; sets the output folder (default: %(default)s)")
    parser.add_argument("--plot", type=_str2bool, nargs="?", const=True, default=False,
                        help="Set true for plots (default: %(default)s)")
    parser.add_argument("--backwards", default=BACKWARDS)
    args = parser.parse_args()
    run(args.input_file, args.timestamp,args.backwards, args.plot,  output_path=args.output_location)
