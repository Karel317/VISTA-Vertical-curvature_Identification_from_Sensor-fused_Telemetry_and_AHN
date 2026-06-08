import os
import sys
import argparse
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
import requests
import rasterio
import io
import math
import scipy.interpolate as sc

# Shared cutting/loading helpers. Add this file's folder to sys.path so the import
# resolves regardless of how the module is loaded.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcap_utils import load_gps
# ============================================================
# SETTINGS — edit these before running
# ============================================================

INPUT_FILE      = r"D:\Data_gathered\2026_05_22\Rosbag\10_50_00\rosbag\rosbag_0.mcap"
OUTPUT_PATH =   r"D:\Validation_results"

GPS_TOPIC  = "/navsat_topic"           # set to None to list topics

TIME               = 1779439917.972930459

# GPS slice half-width (s) around TIME. Must match Validation_curvature_IMU_GPS's
# GPS_WINDOW so the shared mcap_utils.load_gps decode is reused across both modules.
GPS_WINDOW         = 30.0

# Strip geometry  
X_BACK             = -5.0                      # m behind the bike
X_FRONT            = 20.0                      # m in front
DX                 = 0.5                       # interval between samples, this is the resolution of the AHN5 DSM heightmap

# Expected height profile, if it is not in this interval it will be seen as invalid and ignored
MIN_HEIGHT         = -50.0                     # m, expected minimum height of the terrain
MAX_HEIGHT         =  100.0                    # m, expected maximum height of the terrain
# Heading estimation
HEADING_WINDOW_S   = 1.0                       # ± seconds for GPS displacement
SMOOTHENING_FACTOR = 0 # Spline smoothing factor (higher = smoother, lower = closer fit to data)

AHN5_LAYERS = {
    "DSM": {
        "wcs_url":  "https://api.ellipsis-drive.com/v3/ogc/wcs/a4a8a27b-e36e-4dd5-a75b-f7b6c18d33ec",
        "coverage": "fc9d369f-94ca-4373-8281-a6854edb67c9",
    },
    "DTM": {
        "wcs_url":  "https://api.ellipsis-drive.com/v3/ogc/wcs/51e11c02-1065-463d-a0b0-263d80293b16",
        "coverage": "41eaf86f-fa84-48f7-a949-355044ed0e4e",
    },
}

# ── Helper functions ─────────────────────────────────────────────────────────

def _read_GPS(INPUT_FILE, GPS_TOPIC):
    # Topic-listing debug path: read the full bag's summary and stop.
    if GPS_TOPIC is None:
        print("\nGPS_TOPIC is None. Choose from the available topics:")
        with open(INPUT_FILE, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            for ch in summary.channels.values():
                n = summary.statistics.channel_message_counts.get(ch.id, "?")
                schema = summary.schemas.get(ch.schema_id, None)
                schema_name = schema.name if schema else "unknown"
                print(f"  {ch.topic}  —  {n} messages  [{schema_name}]")
        raise SystemExit("Set GPS_TOPIC and re-run this cell.")

    # Normal path: cut (disk-cached) + decode via the shared loader, so the bag
    # is read once per run and reused by Validation_curvature_IMU_GPS.
    gps = load_gps(INPUT_FILE, GPS_TOPIC, TIME, GPS_WINDOW)
    return {"t": gps["t"], "lat": gps["lat"], "lon": gps["lon"]}

def _transform_coords(gps_coords): # GPS coordinates are a different system then the coordinates used in the heightmap, hence need for transformation
    lats = gps_coords["lat"]
    lons = gps_coords["lon"]
    TIME = gps_coords["t"]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    rd_xs, rd_ys = transformer.transform(lons, lats)
    rd_xs = np.array(rd_xs)
    rd_ys = np.array(rd_ys)
    return rd_xs, rd_ys

def _calculating_box_for_query(gps_coords, rd_xs, rd_ys):
    times = gps_coords["t"]
    ref_idx = np.argmin(np.abs(times - TIME)) # Find the index of the reference message
    dt = abs(times[ref_idx] - TIME)
    print(f"Using TIME = {TIME}, closest GPS message Δt = {dt:.3f}s "
            f"(index {ref_idx})")

    ref_t    = float(times[ref_idx])
    ref_rd_x = float(rd_xs[ref_idx])
    ref_rd_y = float(rd_ys[ref_idx])

    # Heading angle from GPS displacement over a short window
    mask = (times >= ref_t - HEADING_WINDOW_S) & (times <= ref_t + HEADING_WINDOW_S)
    win_x = rd_xs[mask]
    win_y = rd_ys[mask]
    if len(win_x) < 3: # Means the bike has barely moved or something went wrong with the GPS
        raise RuntimeError(
            f"Only {len(win_x)} GPS samples in ±{HEADING_WINDOW_S}s window. "
            "Increase HEADING_WINDOW_S.")

    hd_x = win_x[-1] - win_x[0]
    hd_y = win_y[-1] - win_y[0]
    hd_norm = math.hypot(hd_x, hd_y) # Hypot is the length of the vector from the origin to (hd_x, hd_y)
    
    if hd_norm < 0.5:
        raise RuntimeError(
            f"Bike moved only {hd_norm*100:.1f} cm in window — "
            "stationary, heading undefined. Pick a different TIME.")

    heading_x   = hd_x / hd_norm
    heading_y   = hd_y / hd_norm
    heading_deg = math.degrees(math.atan2(hd_y, hd_x))

    # Sample points along the forward axis
    s_samples = np.arange(X_BACK, X_FRONT + DX, DX)
    sample_xs_rd = ref_rd_x + s_samples * heading_x
    sample_ys_rd = ref_rd_y + s_samples * heading_y
   
    gps_coords["rd_xs"] = rd_xs
    gps_coords["rd_ys"] = rd_ys
    gps_coords["s_samples"] = s_samples
    gps_coords["sample_xs_rd"] = sample_xs_rd
    gps_coords["sample_ys_rd"] = sample_ys_rd
    gps_coords["ref_rd_x"] = ref_rd_x
    gps_coords["ref_rd_y"] = ref_rd_y
    gps_coords["heading_x"] = heading_x
    gps_coords["heading_y"] = heading_y
    gps_coords["heading_deg"] = heading_deg
    return gps_coords

def _query_ahn5(gps_coords, wcs_url, coverage):
    s_samples =  gps_coords["s_samples"]

    sample_xs_rd = gps_coords["sample_xs_rd"]
    sample_ys_rd = gps_coords["sample_ys_rd"]

    margin = 1.0    # m of padding around the strip
    xmin = sample_xs_rd.min() - margin
    xmax = sample_xs_rd.max() + margin
    ymin = sample_ys_rd.min() - margin
    ymax = sample_ys_rd.max() + margin

    res = 0.5     # AHN5 resolution = 0.5 m
    gw  = int(math.ceil((xmax - xmin) / res)) + 1
    gh  = int(math.ceil((ymax - ymin) / res)) + 1

    params = {
        "service":  "WCS",
        "version":  "1.0.0",
        "request":  "GetCoverage",
        "coverage": coverage,
        "crs":      "EPSG:28992",
        "bbox":     f"{xmin},{ymin},{xmax},{ymax}",
        "width":    gw,
        "height":   gh,
        "format":   "GEOTIFF",
    }

    resp = requests.get(wcs_url, params=params, timeout=30)
    resp.raise_for_status()

    z = np.full(len(s_samples), np.nan)
    with rasterio.open(io.BytesIO(resp.content)) as src:
        no_data = src.nodata
        for i, vals in enumerate(src.sample(zip(sample_xs_rd, sample_ys_rd))):
            v = float(vals[0])
            if no_data is not None and v == no_data:
                continue
            if not np.isfinite(v):
                continue
            if MIN_HEIGHT < v < MAX_HEIGHT:
                z[i] = v
    return z

def _build_profile(name, valid, z_raw, gps_coords):
    s_samples = gps_coords["s_samples"]
    v = valid
    if not v.all():
        z_filled = np.interp(s_samples, s_samples[v], z_raw[v])
    else:
        z_filled = z_raw.copy()

    ref_z_idx = int(np.argmin(np.abs(s_samples)))
    ref_z_nap = float(z_filled[ref_z_idx])
    z_relative = z_filled - ref_z_nap

    spline = sc.make_splrep(s_samples, z_relative, s=SMOOTHENING_FACTOR)
    dzdx   = spline.derivative(nu=1)(s_samples)
    d2zdx2 = spline.derivative(nu=2)(s_samples)
    slope_deg = np.degrees(np.arctan(dzdx))
    kappa     = abs(d2zdx2) / (1.0 + dzdx ** 2) ** 1.5
    with np.errstate(divide="ignore"):
        r = np.where(kappa > 0, 1.0 / kappa, np.inf)

    return {
        "s":           s_samples,
        "x":           s_samples,
        "z":           z_relative,
        "slope_deg":   slope_deg,
        "kappa":       kappa,
        "r":           r,
        "ref_z_nap":   ref_z_nap,
        "ref_time":    TIME,
        "method":      f"AHN5 {name}",
        "spline":      spline,
    }

def _plot(gps_coords, profiles, valid, z_samples):

    rd_xs = gps_coords["rd_xs"]
    rd_ys = gps_coords["rd_ys"]
    s_samples = gps_coords["s_samples"]
    sample_xs_rd = gps_coords["sample_xs_rd"]
    sample_ys_rd = gps_coords["sample_ys_rd"]
    ref_rd_x = gps_coords["ref_rd_x"]
    ref_rd_y = gps_coords["ref_rd_y"]
    heading_x = gps_coords["heading_x"]
    heading_y = gps_coords["heading_y"]
    heading_deg = gps_coords["heading_deg"]

    LAYER_STYLE = {
        "DSM": {"color": "tab:green",  "raw_color": "darkgreen"},
        "DTM": {"color": "tab:orange", "raw_color": "darkorange"},
    }

    fig = plt.figure(figsize=(20, 10))
    gs  = fig.add_gridspec(3, 2)

    # --- top-left: full GPS track + strip ---
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.plot(rd_xs, rd_ys, "-", color="lightgray", lw=1, label="full GPS track")
    ax_map.plot(sample_xs_rd, sample_ys_rd, "o-", color="red", ms=2, label="AHN5 strip")
    ax_map.plot(ref_rd_x, ref_rd_y, "s", color="blue", ms=8, label="bike position")
    ax_map.set_title("Top-down view (bag-wide)")
    ax_map.set_xlabel("RD X [m]")
    ax_map.set_ylabel("RD Y [m]")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="best", fontsize=9)
    ax_map.grid(True, alpha=0.3)

    # --- top-right: zoomed view of the strip ---
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_zoom.plot(sample_xs_rd, sample_ys_rd, "o-", color="red", ms=4, label="sample points")
    ax_zoom.plot(ref_rd_x, ref_rd_y, "s", color="blue", ms=10, label="bike position")
    ax_zoom.annotate("", xy=(ref_rd_x + 4*heading_x, ref_rd_y + 4*heading_y),
                    xytext=(ref_rd_x, ref_rd_y),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax_zoom.text(ref_rd_x + 4.5*heading_x, ref_rd_y + 4.5*heading_y,
                f"{heading_deg:.0f}°", color="blue")
    ax_zoom.set_title("Sample strip (zoomed)")
    ax_zoom.set_xlabel("RD X [m]")
    ax_zoom.set_ylabel("RD Y [m]")
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_zoom.legend(loc="best", fontsize=9)
    ax_zoom.grid(True, alpha=0.3)

    # --- 2nd row: elevation profiles ---
    ax_p = fig.add_subplot(gs[1, :])
    x_smooth = np.linspace(s_samples.min(), s_samples.max(), 1000)

    for name, p in profiles.items():
        style = LAYER_STYLE[name]
        ref_z = p["ref_z_nap"]
        ax_p.plot(s_samples, p["z"], "-", color=style["color"], lw=2, label=f"AHN5 {name} (smoothed)")
        ax_p.plot(s_samples[valid[name]], (z_samples[name] - ref_z)[valid[name]],
                "o", color=style["raw_color"], ms=4, label=f"AHN5 {name} raw")
        ax_p.plot(x_smooth, p["spline"](x_smooth), "--", color=style["color"], lw=1, alpha=0.5)

    ax_p.axvline(0, color="blue", lw=0.7, ls="--", label="bike at x=0")
    ax_p.axhline(0, color="black", lw=0.5)
    ax_p.set_title(f"AHN5 ground profile")
    ax_p.set_xlabel("X (forward) [m]")
    ax_p.set_ylabel("Elevation relative to bike [m]")
    ax_p.legend(loc="best")
    ax_p.grid(True, alpha=0.3)

    # --- 3rd row: radius of curvature ---
    ax_c = fig.add_subplot(gs[2, :])

    for name, p in profiles.items():
        style = LAYER_STYLE[name]
        ax_c.plot(s_samples, p["r"], "-", color=style["color"], lw=2, label=f"AHN5 {name}")

    ax_c.axvline(0, color="blue", lw=0.7, ls="--", label="bike at x=0")
    ax_c.axhline(0, color="black", lw=0.5)
    ax_c.set_title(f"AHN5 radius of curvature")
    ax_c.set_xlabel("Distance (forward) [m]")
    ax_c.set_ylabel("Radius [m]")
    ax_c.legend(loc="best")
    ax_c.grid(True, alpha=0.3)

    plt.suptitle("AHN5 validation strip", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

def _save_output(input_file, profiles):
    # Output folder is derived from the timestamp (datetime.fromtimestamp), the
    # same convention as lidar_pipeline_unified, so every validation method and
    # the calculation pipeline write into the same OUTPUT_PATH/<date>/<time>/<ts>
    # folder. (input_file is no longer used for the path.)
    from datetime import datetime
    dt = datetime.fromtimestamp(TIME)
    SAVE_DIR = os.path.join(OUTPUT_PATH, dt.strftime("%Y_%m_%d"),
                            dt.strftime("%H_%M_%S"), str(int(TIME)))
    os.makedirs(SAVE_DIR, exist_ok=True)

    def __save_profile(p):
        t      = int(p["ref_time"])
        method = p["method"].replace(" ", "_")
        fpath  = os.path.join(SAVE_DIR, f"{method}_{t}.npz")
        np.savez_compressed(
            fpath,
            x          = p["x"],
            y          = np.array([]),
            z          = p["z"],
            s          = p["x"],
            t          = np.array([p["ref_time"]]),
            method     = np.array([p["method"]]),
            kappa      = p["kappa"],
            slope_deg  = p["slope_deg"],

        )
        print(f"Saved {p['method']} → {fpath}")
        return fpath

    for _, p in profiles.items():
        __save_profile(p)


# ── Main functions ─────────────────────────────────────────────────────────

def run(input_file, timestamp, gps_topic, plot, output_path=OUTPUT_PATH):
    global TIME, OUTPUT_PATH
    TIME = float(timestamp)
    OUTPUT_PATH = output_path
    gps_coords = _read_GPS(input_file, gps_topic)
    rd_xs, rd_ys = _transform_coords(gps_coords)
    gps_coords = _calculating_box_for_query(gps_coords, rd_xs, rd_ys)
    z_samples = {}
    valid     = {}
    for name, layer in AHN5_LAYERS.items():
        z_samples[name] = _query_ahn5(gps_coords, layer["wcs_url"], layer["coverage"])
        valid[name]     = ~np.isnan(z_samples[name])
    profiles = {name: _build_profile(name, valid[name], z_samples[name], gps_coords) for name in AHN5_LAYERS}
    _save_output(input_file, profiles)
    if plot:
        _plot(gps_coords, profiles, valid, z_samples)


def _str2bool(val):
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "t", "yes", "y")

if __name__ == "__main__": # I
    parser = argparse.ArgumentParser(description="AHN5 curvature validation.")
    parser.add_argument("--input-file", default=INPUT_FILE,
                        help="MCAP path with GPS data(default: %(default)s)")
    parser.add_argument("--output-location", default=OUTPUT_PATH,
                        help="General output map, will create a date and time folder in this folder (default: %(default)s)")
    parser.add_argument("--timestamp", type=float, default=TIME,
                        help="Reference Unix timestamp (default: %(default)s)")
    parser.add_argument("--gps_topic", default=GPS_TOPIC,
                        help="local path to the correct topic (default: %(default)s)")
    parser.add_argument("--plot", type=_str2bool, nargs="?", const=True, default=False,
                        help="Set true for plots (default: %(default)s)")
    args = parser.parse_args()
    run(args.input_file, args.timestamp, args.gps_topic, args.plot,
        output_path=args.output_location)
