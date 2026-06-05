import os
import sys
import csv
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt

# TODO: 
# Dubbelcheck this code
# Make a pipeline
# Work with time, now everything is saved on 5m instead of seconds
# If both CSF and PW are used either in this code or other code the key should be csf_...
    # Otherwise i think we are overriding data now
# Cut everything after max bridge height (and before?)
# Implement a moving thing so no issues with the gps being inacurate and so methods are a meter off.
    # This can be done in 2 ways, fully letting it change a lot or implementing a max moveable distance based on gps resolution.

PHYSICAL_MEASUREMENT = False
THRESHOLD = 0.1
RESULTS_DIR = r"D:\Validation_results\2026_05_22\10_51_57\1779439917"
DISTANCE_FROM_PHYS_MEAS_POINT = 3.1711242130880257 #Measured in foxglove or inbetween 2 gps points,
BOTTOM_LIMIT_KAPPA = 1e-6 # Minimum curvature to show on the plot (to have a better visualisation)
# ── Prefixes of the result files to load (filenames without .npz extension) ──
VALIDATION_PREFIXES = [
    "Physical_meas_Station_Brug",
]

CALCULATION_PREFIXES = [
    "AHN5_DSM",
    "AHN5_DTM",
    "EKF_curvature_validation",
    "KISS_ICP",
    "Z_positional_tracking",
    "height-deriv_csf",
    "height-deriv_patchwork",
    "PCA_csf",
    "PCA_patchwork",
    "RANSAC_csf",
    "RANSAC_patchwork",
]

SMOOTHENING_FACTOR = 0  # Adjust as needed (0 = no smoothing, higher = more smoothing) # Keep it at 0 i think

# ── Alignment settings ───────────────────────────────────────────────────────
ALIGN_TO_REFERENCE   = True
REFERENCE_SUBSTR     = "Physical_meas"  # Align everything to where the physical measurement rises
ALIGN_ABS_SLOPE      = 0.02   # m/m : minimum sustained rise rate (absolute pass)
ALIGN_ABS_BASELINE   = 0.03   # m   : how far above baseline counts as "risen"
ALIGN_SMOOTH_WIN_M   = 1.0    # m   : smoothing window for z before detection
ALIGN_MIN_RISE_LEN_M = 1.0    # m   : rise must be sustained at least this long

# ── Trim settings ────────────────────────────────────────────────────────────
TRIM_AT_PHYS_PEAK   = True    # Right cutoff at the peak of the physical measurement
PHYS_MEAS_SUBSTR    = "Physical_meas"
TRIM_AT_SLOPE_START = True    # Left cutoff at the detected slope-start
# ─────────────────────────────────────────────────────────────────────────────

def _find_by_prefix(directory, prefixes):
    matches = []
    for fname in os.listdir(directory):
        if fname.endswith(".npz") and any(fname.startswith(p) for p in prefixes):
            matches.append(fname[:-4])
    return matches


# ── Elevation-start detection helpers ─────────────────────────────────────────
def _smooth(z, window_samples):
    z = np.asarray(z, float)
    w = max(1, int(window_samples))
    if w % 2 == 0:
        w += 1
    if w <= 1:
        return z
    pad = w // 2
    zp = np.pad(z, pad, mode="edge")
    return np.convolve(zp, np.ones(w) / w, mode="valid")[:len(z)]


def detect_elevation_start(
    s, z,
    slope_threshold=0.02,
    baseline_tol=0.03,
    smooth_window_m=1.0,
    min_rise_len_m=1.0,
    baseline_frac=0.20,
    baseline_pct=5.0,
    grid_step_m=None,
):
    s = np.asarray(s, float)
    z = np.asarray(z, float)
    m = np.isfinite(s) & np.isfinite(z)
    s, z = s[m], z[m]
    if s.size < 4:
        return None
    order = np.argsort(s)
    s, z = s[order], z[order]
    s, idx = np.unique(s, return_index=True)
    z = z[idx]
    if s.size < 4:
        return None
    if grid_step_m is None:
        grid_step_m = float(np.median(np.diff(s)))
    if not np.isfinite(grid_step_m) or grid_step_m <= 0:
        return None
    sg = np.arange(s[0], s[-1], grid_step_m)
    if sg.size < 4:
        return None
    zg = np.interp(sg, s, z)
    zs = _smooth(zg, round(smooth_window_m / grid_step_m))
    n_base = max(3, int(baseline_frac * zs.size))
    baseline = np.percentile(zs[:n_base], baseline_pct)
    win = max(1, int(round(min_rise_len_m / grid_step_m)))
    win_len_m = win * grid_step_m
    need_rise = slope_threshold * win_len_m
    for i in range(0, zs.size - win):
        rise = zs[i + win] - zs[i]
        if rise >= need_rise and zs[i + win] >= baseline + baseline_tol:
            j0 = max(0, i - win)
            cross = np.argmax(zs[j0:i + win + 1] >= baseline + baseline_tol)
            return float(sg[j0 + cross])
    return None


def detect_elevation_start_adaptive(
    s, z,
    abs_slope=0.02, abs_baseline_tol=0.03,
    char_len_m=10.0, rel_height_frac=0.05,
    **kwargs,
):
    start = detect_elevation_start(
        s, z, slope_threshold=abs_slope, baseline_tol=abs_baseline_tol, **kwargs
    )
    if start is not None:
        return start, "absolute"
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if z.size < 4:
        return None, "failed"
    z_range = np.percentile(z, 95) - np.percentile(z, 5)
    if not np.isfinite(z_range) or z_range <= 0:
        return None, "failed"
    rel_baseline_tol = max(1e-6, rel_height_frac * z_range)
    rel_slope = max(1e-6, (rel_height_frac * z_range) / char_len_m)
    start = detect_elevation_start(
        s, z, slope_threshold=rel_slope, baseline_tol=rel_baseline_tol, **kwargs
    )
    if start is not None:
        return start, f"relative (z_range={z_range:.3f} m)"
    return None, "failed"


def align_profiles_to_reference(store_list, reference_substr, smoothing_factor=0.0, **detect_kwargs):
    ref_start = None
    for label, store in store_list:
        for method, d in store.items():
            if reference_substr in method:
                rs, mode = detect_elevation_start_adaptive(d["s"], d["z"], **detect_kwargs)
                if rs is not None:
                    ref_start = rs
                    print(f"Reference [{label} – {method}] start s={rs:.3f} m ({mode})")
                else:
                    print(f"WARNING: reference [{label} – {method}] start NOT detected.")
                break
    if ref_start is None:
        print("WARNING: no reference start -> no alignment applied.")
        return None
    for label, store in store_list:
        for method, d in store.items():
            start, mode = detect_elevation_start_adaptive(d["s"], d["z"], **detect_kwargs)
            if start is None:
                print(f"WARNING [{label} – {method}]: start not detected -> left unshifted.")
                continue
            shift = ref_start - start
            s_new = d["s"] + shift
            d["s"] = s_new
            d["spline_z"] = sc.make_splrep(s_new, d["z"], s=smoothing_factor)
            d["spline_kappa"] = sc.make_splrep(s_new, d["kappa"], s=smoothing_factor)
            print(f"  [{label} – {method}]: start s={start:.3f} -> shift {shift:+.3f} m ({mode})")
    return ref_start


def find_profile_peak_s(store, method_substr):
    for method, d in store.items():
        if method_substr in method:
            z = np.asarray(d["z"], float)
            s = np.asarray(d["s"], float)
            if z.size == 0 or s.size != z.size or not np.any(np.isfinite(z)):
                print(f"WARNING: peak profile '{method}' has no usable z/s.")
                return None
            idx = int(np.nanargmax(z))
            return float(s[idx])
    print(f"WARNING: no profile matching '{method_substr}' found for peak.")
    return None


def trim_profiles_to_cutoff(store_list, s_cutoff=None, s_min=None, smoothing_factor=0.0):
    lo_txt = f"{s_min:.3f}" if s_min is not None else "-inf"
    hi_txt = f"{s_cutoff:.3f}" if s_cutoff is not None else "+inf"
    for label, store in store_list:
        for method, d in store.items():
            s = np.asarray(d["s"], float)
            mask = np.ones(s.size, dtype=bool)
            if s_min is not None:
                mask &= s >= s_min
            if s_cutoff is not None:
                mask &= s <= s_cutoff
            n_keep = int(np.count_nonzero(mask))
            if n_keep == s.size:
                continue
            if n_keep < 4:
                print(f"WARNING [{label} – {method}]: only {n_keep} points in "
                      f"[{lo_txt}, {hi_txt}] m -> left untrimmed.")
                continue
            for key in ("x", "y", "z", "s", "t", "kappa"):
                arr = d.get(key)
                if arr is not None and np.ndim(arr) > 0 and np.size(arr) == mask.size:
                    d[key] = np.asarray(arr)[mask]
            d["spline_z"]     = sc.make_splrep(d["s"], d["z"], s=smoothing_factor)
            d["spline_kappa"] = sc.make_splrep(d["s"], d["kappa"], s=smoothing_factor)
            print(f"  [{label} – {method}]: trimmed to s in [{lo_txt}, {hi_txt}] m "
                  f"({s.size} -> {n_keep} points)")

VALIDATION_FILES   = _find_by_prefix(RESULTS_DIR, VALIDATION_PREFIXES)
CALCULATION_FILES  = _find_by_prefix(RESULTS_DIR, CALCULATION_PREFIXES)

results_validation = {} # Access variables by method name example: results_validation["AHN4 DTM"]["x"]
try:
    for fname in VALIDATION_FILES:
        fpath = os.path.join(RESULTS_DIR, fname + ".npz")
        data = np.load(fpath, allow_pickle=True)

        method = str(data["method"].flat[0])
        kappa_variants = [k for k in data.files if k.startswith("kappa")]
        kappa_key = "kappa" if "kappa" in data.files else (kappa_variants[0] if kappa_variants else None)
        core = {
            "x":     data["x"] if "x" in data.files else None,
            "y":     data["y"]if "y" in data.files else None,
            "z":     data["z"] if "z" in data.files else None,
            "s":     data["s"] if "s" in data.files else None,
            "t":     data["t"] if "t" in data.files else None,
            "kappa": data[kappa_key] if kappa_key else None,
        }
        extra_keys = [k for k in data.files if k not in ("x", "y", "z", "s", "t", "kappa", "method") and k != kappa_key]
        extras = {k: data[k] for k in extra_keys}

        results_validation[method] = {**core, **extras}
        print(f"Loaded: {method}")
except FileNotFoundError:
    sys.exit("Error: No files found for VALIDATION. Check filename and if does not have .npz")

results_calculation = {} # Access variables by method name example: results_calculation["AHN4 DTM"]["x"]
try:
    for fname in CALCULATION_FILES:
        fpath = os.path.join(RESULTS_DIR, fname + ".npz")
        data = np.load(fpath, allow_pickle=True)

        method = str(data["method"].flat[0])
        kappa_variants = [k for k in data.files if k.startswith("kappa")]
        kappa_key = "kappa" if "kappa" in data.files else (kappa_variants[0] if kappa_variants else None)
        core = {
            "x":     data["x"] if "x" in data.files else None,
            "y":     data["y"] if "y" in data.files else None,
            "z":     data["z"] if "z" in data.files else None,
            "s":     data["s"] if "s" in data.files else None,
            "t":     data["t"] if "t" in data.files else None,
            "kappa": data[kappa_key] if kappa_key else None,
        }
        extra_keys = [k for k in data.files if k not in ("x", "y", "z", "s", "t", "kappa", "method") and k != kappa_key]
        extras = {k: data[k] for k in extra_keys}

        results_calculation[method] = {**core, **extras}
        print(f"Loaded: {method}")
except FileNotFoundError:
    sys.exit("Error: No files found for CALCULATION. Check filename and if does not have .npz")


# ── Per-method statistics ─────────────────────────────────────────────────────
print("\n── Check if data is consistent ──────────────────────────────────────")
per_method_stats = []
processed_files = set() # To avoid duplicate processing
for label, store in [("VALIDATION", results_validation), ("CALCULATION", results_calculation)]: # loops over the 2 data stores and gives it the correct label
    for method, d in store.items():
        key = (label, method)
        if key in processed_files:
            continue
        processed_files.add(key)
        x, y, z, s, kappa = d["x"],d["y"], d["z"], d["s"], d["kappa"]

        if s is None or np.ndim(z) == 0 or np.size(z) == 0: # If s is missing we calculate it
            print(f"Warning: No s data for {label} - {method}, Calculating with x and y") # BUG Does this ever happen?
            s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

        if z is None or np.ndim(z) == 0 or np.size(z) == 0: # If z is missing we calculate it with the slope and s
            print(f"Warning: No z data for {label} - {method}, Calculating with slope")
            slope_deg = d["slope_deg"]
            dz_seg = np.tan(np.radians(slope_deg)) * np.diff(s)
            z = np.concatenate([[0], np.cumsum(dz_seg)])


        if len(z) != len(s):
            print(f"---------------------------------------------------------------------------------------")
            print(f"Error [{label} – {method}]: z has {len(z)} points but s has {len(s)} points. FIX THIS")
            print(f"---------------------------------------------------------------------------------------")
           


        valid = np.isfinite(z) # Safety measure to ensure we only work with valid data points
        z, s = z[valid], s[valid] # We only keep the valid points


        sp_z = sc.make_splrep(s, z, s=SMOOTHENING_FACTOR) # Function of all the valid points
        if kappa is None or np.ndim(kappa) == 0 or np.size(kappa) == 0: # If kappa is missing we calculate it with the spline of z
            # Get the derivatives analytically
            dzds = sp_z.derivative(nu=1)(s)
            slope_deg = np.degrees(np.arctan(dzds))
            d2zds2 = sp_z.derivative(nu=2)(s)
            kappa = abs(d2zds2) / (1.0 + dzds ** 2) ** 1.5 # Curvature formula for a 2D curve defined by z(s)
            
        sp_k = sc.make_splrep(s, kappa, s=SMOOTHENING_FACTOR)

        store[method]["spline_z"] = sp_z
        store[method]["spline_kappa"] = sp_k
        store[method]["kappa"] = kappa
        store[method]["s"] = s
        store[method]["z"] = z

        s_lin = np.linspace(s[0], s[-1], 1000)

        if SMOOTHENING_FACTOR != 0:
            delta_z       = sp_z(s) - z
            rmse       = np.sqrt(np.mean((delta_z) ** 2))
            mae      = np.mean(np.abs(delta_z))
            max_diff   = np.max(np.abs(delta_z))

            per_method_stats.append((label, method, rmse, mae, max_diff))
            print(f"\n  [{label} – {method}]")
            print(f"    RMSE  (spline vs raw z): {rmse:.6f} m")
            print(f"    MAE   (spline vs raw z): {mae:.6f} m")
            print(f"    Max Δ (spline vs raw z): {max_diff:.6f} m")
print("No issues")
 """""
if PHYSICAL_MEASUREMENT:
    phys_key = next(k for k in results_validation if "Physical_meas" in k)
    dist_grid = np.arange(-5.0, 20.0, 0.222) # 0.222m is the distance between physical measurement points
    results_validation[phys_key]["s"] = dist_grid
    z = np.zeros(len(dist_grid))

    phys_idx = 0
    for i, d in enumerate(dist_grid): 
        # Logic here is we are DISTANCE_FROM_PHYS_MEAS_POINT away from the first physical measurement point so we assign z as 0
        # Untill this distance is reached, then we take the points of the physical measurement up until 20m.
        if d >= DISTANCE_FROM_PHYS_MEAS_POINT and phys_idx < len(results_validation[phys_key]["z"]):
            z[i] = results_validation[phys_key]["z"][phys_idx] # already in meters
            phys_idx += 1

    sp_z = sc.make_splrep(dist_grid, z, s=SMOOTHENING_FACTOR)
    dzds   = sp_z.derivative(nu=1)(dist_grid)
    d2zds2 = sp_z.derivative(nu=2)(dist_grid)
    kappa  = abs(d2zds2) / (1.0 + dzds ** 2) ** 1.5
    sp_k   = sc.make_splrep(dist_grid, kappa, s=SMOOTHENING_FACTOR)

    results_validation[phys_key]["z"] = z
    results_validation[phys_key]["spline_z"]     = sp_z
    results_validation[phys_key]["spline_kappa"] = sp_k
    results_validation[phys_key]["kappa"]        = kappa
"""
# ── Align all profiles to the physical measurement elevation-start ────────────
s_start_cutoff = None
if ALIGN_TO_REFERENCE:
    print("\n── Aligning profiles to reference elevation-start ───────────────────")
    s_start_cutoff = align_profiles_to_reference(
        [("VALIDATION", results_validation), ("CALCULATION", results_calculation)],
        reference_substr=REFERENCE_SUBSTR,
        smoothing_factor=SMOOTHENING_FACTOR,
        abs_slope=ALIGN_ABS_SLOPE,
        abs_baseline_tol=ALIGN_ABS_BASELINE,
        smooth_window_m=ALIGN_SMOOTH_WIN_M,
        min_rise_len_m=ALIGN_MIN_RISE_LEN_M,
    )

# ── Trim all profiles to [slope-start, physical measurement peak] ─────────────
if TRIM_AT_PHYS_PEAK or TRIM_AT_SLOPE_START:
    print("\n── Trimming profiles (slope-start / physical peak) ──────────────────")
    s_cutoff = find_profile_peak_s(results_validation, PHYS_MEAS_SUBSTR) if TRIM_AT_PHYS_PEAK else None
    if TRIM_AT_PHYS_PEAK and s_cutoff is None:
        print("WARNING: physical-measurement peak not found -> no right cutoff applied.")
    s_left = s_start_cutoff if TRIM_AT_SLOPE_START else None
    if TRIM_AT_SLOPE_START and s_left is None:
        print("WARNING: slope-start not available (alignment off/failed) -> no left cutoff applied.")
    if s_cutoff is not None or s_left is not None:
        lo = f"{s_left:.3f}" if s_left is not None else "-inf"
        hi = f"{s_cutoff:.3f}" if s_cutoff is not None else "+inf"
        print(f"Global window: s in [{lo}, {hi}] m.")
        trim_profiles_to_cutoff(
            [("VALIDATION", results_validation), ("CALCULATION", results_calculation)],
            s_cutoff=s_cutoff,
            s_min=s_left,
            smoothing_factor=SMOOTHENING_FACTOR,
        )

# ── Cross comparison (validation vs calculation, different methods only) ──────
comparisons = []
for vm, vd in results_validation.items(): # Vm is validation method, vd is validation data 
    for cm, cd in results_calculation.items():
        s_min  = max(vd["s"].min(), cd["s"].min())
        s_max  = min(vd["s"].max(), cd["s"].max())
        s_row  = np.linspace(s_min, s_max, 500)
        z_v    = vd["spline_z"](s_row)
        z_c    = cd["spline_z"](s_row)
        diff   = z_v - z_c
        rmse   = np.sqrt(np.mean(diff ** 2))
        mae    = np.mean(np.abs(diff))
        threshold_error = np.mean(np.abs(diff) < THRESHOLD)

        comparisons.append((vm, cm, s_row, diff, rmse, mae, threshold_error))
        print(f"\n  [{vm}  vs  {cm}]")
        print(f"    RMSE:     {rmse:.4f} m")
        print(f"    MAE:      {mae:.4f} m")
        print(f"    In threshold: {threshold_error:.4f} m")


# ── CSV export ────────────────────────────────────────────────────────────────
raw_t = results_calculation[next(iter(results_calculation))]["t"]
t = str(raw_t.flat[0]) if hasattr(raw_t, "flat") else str(raw_t)
csv_path = os.path.join(r"D:\Validation_results\Statistics", f"{t}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Cross comparison (validation vs calculation)"])
    writer.writerow(["Val method", "Calc method", "RMSE (m)", "MAE (m)", "In threshold"])
    for vm, cm, _, __, rmse, mae, max_d in comparisons:
        writer.writerow([vm, cm, f"{rmse:.6f}", f"{mae:.6f}", f"{max_d:.6f}"])

print(f"\nStatistics saved to {csv_path}")


# ── Plotting ──────────────────────────────────────────────────────────────────
n_rows = 2 + (1 if comparisons else 0)
fig, axes = plt.subplots(n_rows, 1, figsize=(12, 5 * n_rows))

ax_z = axes[0]
ax_k = axes[1]
ax_d = axes[2] if comparisons else None

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plotted = set()
color_idx = 0
for label, store, marker, ls in [
    ("Val",  results_validation,  "o", "-"),
    ("Calc", results_calculation, "x", "--"),
]:
    for method, d in store.items():
        tag = f"{label}: {method}"
        if tag in plotted:  # Avoid duplicate legend entries
            continue
        plotted.add(tag)
        c = colors[color_idx % len(colors)] # Safety measure in case there are more methods than colors, it just cycles through the colors again.
        s_lin = np.linspace(d["s"][0], d["s"][-1], 500) 

        # Elevation
        ax_z.scatter(d["s"], d["z"], s=20, color=c, marker=marker, alpha=0.7, zorder=3)
        ax_z.plot(s_lin, d["spline_z"](s_lin), color=c, linestyle=ls, label=tag) # Easier to check lines then dots but not needed

        # Curvature
        ax_k.scatter(d["s"], d["kappa"], s=25, color=c, marker=marker, zorder=3)
        ax_k.plot(s_lin, d["spline_kappa"](s_lin), color=c, linestyle=ls, label=tag)

        color_idx += 1

ax_z.set_xlabel("s (m)")
ax_z.set_ylabel("z (m, relative to bike)")
ax_z.set_title("Elevation profile")
ax_z.legend()
ax_z.grid(True)

ax_k.set_xlabel("s (m)")
ax_k.set_ylabel("κ (1/m)")
ax_k.set_title("Curvature profile")
ax_k.legend()
ax_k.set_yscale("log")  
ax_k.set_ylim(bottom=BOTTOM_LIMIT_KAPPA)
ax_k.grid(True)

if ax_d is not None:
    for vm, cm, x_cmp, diff, _, _, _ in comparisons:
        ax_d.plot(x_cmp, diff, label=f"{vm} − {cm}")
    ax_d.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_d.set_xlabel("s (m)")
    ax_d.set_ylabel("Δz (m)")
    ax_d.set_title("Elevation difference (Validation − Calculation)")
    ax_d.legend()
    ax_d.grid(True)

plt.tight_layout(h_pad=6)
plt.show()
