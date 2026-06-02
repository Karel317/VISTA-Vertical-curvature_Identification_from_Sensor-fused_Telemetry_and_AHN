import os
import sys
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt

date_folder = "2026_05_22"
time_folder = "14_30_00"
RESULTS_DIR = os.path.join(r"D:\Validation_results", date_folder, time_folder)

# ── List the result files to load (filenames without .npz extension) ──────────
VALIDATION_FILES = [
    "AHN5_DSM_1779439914",
    "AHN5_DTM_1779439914",
    "Physical_meas_stationsbrug_1779439914.817101"
]

CALCULATION_FILES = [
    "CSF_height-deriv_1779439917",
    "CSF_PCA_1779439917",
    "CSF_RANSAC_1779439917",
    "EKF_curvature_validation_1779439917.9729304",
    "PW_height-deriv_1779439917",
    "PW_PCA_1779439917",
    "PW_RANSAC_1779439917",
    "Z_positional_tracking_1779439917.9729304",

]

SMOOTHENING_FACTOR = 0  # Adjust as needed (0 = no smoothing, higher = more smoothing)
# ─────────────────────────────────────────────────────────────────────────────

results_validation = {} # Access variables by method name example: results["AHN4 DTM"]["x"]
try:
    for fname in VALIDATION_FILES:
        fpath = os.path.join(RESULTS_DIR, fname + ".npz")
        print(fpath)
        data = np.load(fpath, allow_pickle=True)

        method = str(data["method"].flat[0])
        core = {
            "x":     data["x"],
            "y":     data["y"],
            "z":     data["z"],
            "s":     data["s"],
            "t":     data["t"],
            "kappa": data["kappa"],
        }
        extra_keys = [k for k in data.files if k not in ("x", "y", "z", "s", "t","kappa", "method")]
        extras = {k: data[k] for k in extra_keys}

        results_validation[method] = {**core, **extras}
        print(f"Loaded: {method}")
        print(f"  Core   : x{data['x'].shape}, y{data['y'].shape}, z{data['z'].shape}, "
            f"s{data['s'].shape}, t={float(data['t'][0]):.2f}, kappa={data['kappa'].shape}")
        if extras:
            print(f"  Extras : {list(extras.keys())}")
except FileNotFoundError:
    sys.exit("Error: No files found for VALIDATION. Check filename and if does not have .npz")

results_calculation = {} # Access variables by method name example: results["AHN4 DTM"]["x"]
try:
    for fname in CALCULATION_FILES:
        fpath = os.path.join(RESULTS_DIR, fname + ".npz")
        data = np.load(fpath, allow_pickle=True)

        method = str(data["method"].flat[0])
        core = {
            "x":     data["x"],
            "y":     data["y"],
            "z":     data["z"],
            "s":     data["s"],
            "t":     data["t"],
            "kappa": data["kappa"],
        }
        extra_keys = [k for k in data.files if k not in ("x", "y", "z", "s", "t", "kappa", "method")]
        extras = {k: data[k] for k in extra_keys}

        results_calculation[method] = {**core, **extras}
        print(f"Loaded: {method}")
        print(f"  Core   : x{data['x'].shape}, y{data['y'].shape}, z{data['z'].shape}, "
            f"s{data['s'].shape}, t={float(data['t'][0]):.2f}, kappa={data['kappa'].shape}")
        if extras:
            print(f"  Extras : {list(extras.keys())}")
except FileNotFoundError:
    sys.exit("Error: No files found for CALCULATION. Check filename and if does not have .npz")

# ── Per-method statistics ─────────────────────────────────────────────────────
print("\n── Per-method spline statistics ──────────────────────────────────────")
seen = set()
for label, store in [("VALIDATION", results_validation), ("CALCULATION", results_calculation)]:
    for method, d in store.items():
        key = (label, method)
        if key in seen:
            continue
        seen.add(key)
        x, y, z, s = d["x"],d["y"], d["z"], d["s"]

        if s is not None:
            distance = s
        else:
            distance = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

        if z is None or np.ndim(z) == 0 or np.size(z) == 0:
            print(f"Warning: No z data for {label} - {method}, Calculating with slope")
            slope_deg = d["slope_deg"]
            dz_seg = np.tan(np.radians(slope_deg)) * np.diff(distance)
            z = np.concatenate([[0], np.cumsum(dz_seg)])
            store[method]["z"] = z

            
        
        valid = np.isfinite(z)
        z, distance = z[valid], distance[valid]
        sp_z = sc.make_splrep(distance, z, s=SMOOTHENING_FACTOR)
        if d.get("kappa") is not None:
            kappa = d["kappa"]
        else:
            # Get the derivatives analytically
            dzds = sp_z.derivative(nu=1)(distance)
            slope_deg = np.degrees(np.arctan(dzds))
            d2zds2 = sp_z.derivative(nu=2)(distance)
            kappa = abs(d2zds2) / (1.0 + dzds ** 2) ** 1.5
            
        sp_k = sc.make_splrep(distance, kappa, s=SMOOTHENING_FACTOR)
        store[method]["spline_z"] = sp_z
        store[method]["spline_kappa"] = sp_k
        store[method]["kappa"] = kappa
        store[method]["distance"] = distance
        store[method]["z"] = z
        dist_lin = np.linspace(distance[0], distance[-1], 1000)
        #area_z     = np.trapezoid(np.abs(sp_z(dist_lin)), dist_lin)
        #area_kappa = np.trapezoid(np.abs(sp_k(dist_lin)), dist_lin)

        z_spline      = sp_z(distance) # The spline evaluated at the original positions
        delta_z       = z_spline - z
        rmse       = np.sqrt(np.mean((delta_z) ** 2))
        mae      = np.mean(np.abs(delta_z))
        max_diff   = np.max(np.abs(delta_z))

        print(f"\n  [{label} – {method}]")
        #print(f"    Area under z spline:     {area_z:.4f} m²")
        #print(f"    Area under kappa spline: {area_kappa:.6f} m")
        print(f"    RMSE  (spline vs raw z): {rmse:.6f} m")
        print(f"    MAE   (spline vs raw z): {mae:.6f} m")
        print(f"    Max Δ (spline vs raw z): {max_diff:.6f} m")

# ── Cross comparison (validation vs calculation, different methods only) ──────
comparisons = []
for vm, vd in results_validation.items():
    for cm, cd in results_calculation.items():
        dist_min  = max(vd["distance"].min(), cd["distance"].min())
        dist_max  = min(vd["distance"].max(), cd["distance"].max())
        dist_row  = np.linspace(dist_min, dist_max, 500)
        z_v    = vd["spline_z"](dist_row)
        z_c    = cd["spline_z"](dist_row)
        diff   = z_v - z_c
        rmse   = np.sqrt(np.mean(diff ** 2))
        max_d  = np.max(np.abs(diff))
        comparisons.append((vm, cm, dist_row, diff, rmse, max_d))
        print(f"\n  [{vm}  vs  {cm}]")
        print(f"    RMSE:     {rmse:.4f} m")
        print(f"    Max diff: {max_d:.4f} m")

# ── Plotting ──────────────────────────────────────────────────────────────────
# ── Plotting ──────────────────────────────────────────────────────────────────
n_rows = 2 + (1 if comparisons else 0)
fig, axes = plt.subplots(n_rows, 1, figsize=(12, 5 * n_rows))

# When n_rows == 1, plt.subplots returns a single Axes, not an array.
# Wrapping with np.atleast_1d makes indexing safe in every case.
axes = np.atleast_1d(axes)

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
        if tag in plotted:
            continue
        plotted.add(tag)
        c = colors[color_idx % len(colors)]
        dist_lin = np.linspace(d["distance"][0], d["distance"][-1], 500)

        # Elevation
        ax_z.scatter(d["distance"], d["z"], s=20, color=c, marker=marker, alpha=0.7, zorder=3)
        ax_z.plot(dist_lin, d["spline_z"](dist_lin), color=c, linestyle=ls, label=tag)

        # Curvature (uncommented so ax_k actually gets data)
        ax_k.scatter(d["distance"], d["kappa"], s=25, color=c, marker=marker, zorder=3)
        ax_k.plot(dist_lin, d["spline_kappa"](dist_lin), color=c, linestyle=ls, label=tag)

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
ax_k.set_yscale("log")   # separate statement, not inside legend()
ax_k.grid(True)

if ax_d is not None:
    for vm, cm, x_cmp, diff, rmse, max_d in comparisons:
        ax_d.plot(x_cmp, diff, label=f"{vm} − {cm}")
    ax_d.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_d.set_xlabel("s (m)")
    ax_d.set_ylabel("Δz (m)")
    ax_d.set_title("Elevation difference (Validation − Calculation)")
    ax_d.legend()
    ax_d.grid(True)
    # Note: do NOT use log scale here — Δz is signed and will go negative.
    # If you really want log-like behavior, use symlog:
    # ax_d.set_yscale("symlog", linthresh=1e-3)

plt.tight_layout(h_pad=6)
plt.show()
