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

RESULTS_DIR = r"D:\Validation_results\2026_05_22\10_42_19\1779439339"
DISTANCE_FROM_PHYS_MEAS_POINT = 3.1711242130880257 #Measured in foxglove or inbetween 2 gps points,
BOTTOM_LIMIT_KAPPA = 1e-6 # Minimum curvature to show on the plot (to have a better visualisation)
# ── Prefixes of the result files to load (filenames without .npz extension) ──
VALIDATION_PREFIXES = [
    "RANSAC_patchwork",
]

CALCULATION_PREFIXES = [
    "height-deriv_csf",
    "height-deriv_patchwork",
    "PCA_csf",
    "PCA_patchwork",
    "RANSAC_csf",
    "RANSAC_patchwork",
]

SMOOTHENING_FACTOR = 0  # Adjust as needed (0 = no smoothing, higher = more smoothing) # Keep it at 0 i think
# ─────────────────────────────────────────────────────────────────────────────

def _find_by_prefix(directory, prefixes):
    matches = []
    for fname in os.listdir(directory):
        if fname.endswith(".npz") and any(fname.startswith(p) for p in prefixes):
            matches.append(fname[:-4])
    return matches

VALIDATION_FILES   = _find_by_prefix(RESULTS_DIR, VALIDATION_PREFIXES)
CALCULATION_FILES  = _find_by_prefix(RESULTS_DIR, CALCULATION_PREFIXES)

results_validation = {} # Access variables by method name example: results_validation["AHN4 DTM"]["x"]
try:
    for fname in VALIDATION_FILES:
        fpath = os.path.join(RESULTS_DIR, fname + ".npz")
        data = np.load(fpath, allow_pickle=True)

        method = str(data["method"].flat[0])
        core = {
            "x":     data["x"] if "x" in data.files else None,
            "y":     data["y"]if "y" in data.files else None,
            "z":     data["z"] if "z" in data.files else None,
            "s":     data["s"] if "s" in data.files else None,
            "t":     data["t"] if "t" in data.files else None,
            "kappa": data["kappa"] if "kappa" in data.files else None,
        }
        extra_keys = [k for k in data.files if k not in ("x", "y", "z", "s", "t","kappa", "method")]
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
        core = {
            "x":     data["x"] if "x" in data.files else None,
            "y":     data["y"] if "y" in data.files else None,
            "z":     data["z"] if "z" in data.files else None,
            "s":     data["s"] if "s" in data.files else None,
            "t":     data["t"] if "t" in data.files else None,
            "kappa": data["kappa"] if "kappa" in data.files else None,
        }
        extra_keys = [k for k in data.files if k not in ("x", "y", "z", "s", "t", "kappa", "method")]
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
        max_d  = np.max(np.abs(diff))
        comparisons.append((vm, cm, s_row, diff, rmse, mae, max_d))
        print(f"\n  [{vm}  vs  {cm}]")
        print(f"    RMSE:     {rmse:.4f} m")
        print(f"    MAE:      {mae:.4f} m")
        print(f"    Max diff: {max_d:.4f} m")


# ── CSV export ────────────────────────────────────────────────────────────────
raw_t = results_calculation[next(iter(results_calculation))]["t"]
t = str(raw_t.flat[0]) if hasattr(raw_t, "flat") else str(raw_t)
csv_path = os.path.join(r"D:\Validation_results\Statistics", f"{t}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Cross comparison (validation vs calculation)"])
    writer.writerow(["Val method", "Calc method", "RMSE (m)", "MAE (m)", "Max diff (m)"])
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
