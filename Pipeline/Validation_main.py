import os
import sys
import csv
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt


# ── Files ─────────────────────────────────────────────────────────────────────
# Physical-measurement results may be in a different timestamp folder than the
# LiDAR/calculation results (they are run separately and saved under their own timestamp).
VALIDATION_DIR   = r"C:\Users\karel\OneDrive\BEP\11_04_21\11_04_21\1779440661"   # physical meas .npz files
CALCULATION_DIR  = r"C:\Users\karel\OneDrive\BEP\11_04_21\11_04_21\1779440661"   # lidar/calculation .npz files

# Prefixes of the result files to load (filenames without the .npz extension).
VALIDATION_PREFIXES = [
    "Physical_meas_Sint_Jorispad_Brug",
    #"Physical_meas_Sint_Jorispad_Brug_backwards",
]
CALCULATION_PREFIXES = [
    "AHN5_DSM",
    "AHN5_DTM",
    #"EKF_curvature_validation",
    #"KISS_ICP",
    #"Z_positional_tracking",
    #"height-deriv_csf",
    #"height-deriv_patchwork",
    #"PCA_csf",
    #"PCA_patchwork",
    #"RANSAC_csf",
    "RANSAC_patchwork",
]

# ── Comparison input ───────────────────────────────────────────────────────────────────
THRESHOLD          = 0.1     # m   : |Δz| a threshold used in the comparison as a metric

# ── General ───────────────────────────────────────────────────────────────────
SMOOTHENING_FACTOR = 0       #       spline smoothing (0 = none, higher = smoother)
BOTTOM_LIMIT_KAPPA = 1e-6    # 1/m : lower y-limit for the (log-scaled) curvature plot
PHYS_MEAS_SUBSTR   = "Physical_meas"  # substring identifying the physical measurement profile

# ── Evaluation window ─────────────────────────────────────────────────────────
# Window = [START_VALIDATION - WINDOW_BACK_M, min(START_VALIDATION + WINDOW_FWD_M, peak)]
ALIGN_TO_REFERENCE = True
START_VALIDATION   = 0.0     # m   : user-given s where the physical measurement rises
WINDOW_BACK_M      = 5.0     # m   : keep from START_VALIDATION - 5 m ...
WINDOW_FWD_M       = 20.0    # m   : ... up to START_VALIDATION + 20 m (or the peak, whichever is first)

# ── Per-method fine alignment (corrects the ~±1 m offset between methods) ──────
FINE_ALIGN          = True
FINE_ALIGN_FIELD    = "z"    #       score on elevation ("z", robust) or curvature ("kappa")
FINE_ALIGN_SEARCH_M = 2.0    # m   : slide each method ±this many metres
FINE_ALIGN_STEP_M   = 0.02   # m   : search resolution
FINE_ALIGN_W_MAE    = 0.5    #       score = W*MAE_norm + (1-W)*(1-correlation)
# ─────────────────────────────────────────────────────────────────────────────


def load_results(prefixes, source_label, results_dir):
    """Load the .npz result files in results_dir whose name starts with any of
    `prefixes` into a dict"""
    files = [
        fname[:-4]
        for fname in os.listdir(results_dir)
        if fname.endswith(".npz") and any(fname.startswith(p) for p in prefixes)
    ]
    store = {}
    for fname in files:
        data = np.load(os.path.join(results_dir, fname + ".npz"), allow_pickle=True)

        method         = str(data["method"].flat[0])
        kappa_variants = [k for k in data.files if k.startswith("kappa")]
        kappa_key      = "kappa" if "kappa" in data.files else (kappa_variants[0] if kappa_variants else None)

        core = {
            "x":     data["x"] if "x" in data.files else None,
            "y":     data["y"] if "y" in data.files else None,
            "z":     data["z"] if "z" in data.files else None,
            "s":     data["s"] if "s" in data.files else None,
            "t":     data["t"] if "t" in data.files else None,
            "kappa": data[kappa_key] if kappa_key else None,
        }
        skip   = {"x", "y", "z", "s", "t", "kappa", "method", kappa_key}
        extras = {k: data[k] for k in data.files if k not in skip}

        store[method] = {**core, **extras}
        print(f"Loaded ({source_label}): {method}")
    return store


def find_profile_peak_s(store, method_substr, ref_start):
    """Return the distance (s) of the maximum z of the first profile whose name contains
    `method_substr`, or None if it cannot be found."""
    for method, d in store.items():
        if method_substr in method:
            z = np.asarray(d["z"], float)
            s = np.asarray(d["s"], float)
            if z.size == 0 or s.size != z.size or not np.any(np.isfinite(z)):
                print(f"WARNING: peak profile '{method}' has no usable z/s.")
                return None
            mask = (s >= ref_start + 4.0) & np.isfinite(z)
            if not np.any(mask):
                print(f"WARNING: no finite z values in front of ref_start={ref_start} for '{method}'.")
                return None
            return float(s[mask][int(np.argmax(z[mask]))])
    print(f"WARNING: no profile matching '{method_substr}' found for peak.")
    return None


def trim_profiles_to_cutoff(store_list, start_cutoff, s_cutoff=None, s_min=None, smoothing_factor=0.0):
    """Cut every profile to s in [s_min, s_cutoff] and refit its splines so everything is sampled at the same time."""
    min_txt = f"{s_min:.3f}" if s_min is not None else "-inf"
    max_txt = f"{s_cutoff:.3f}" if s_cutoff is not None else "+inf"
    for label, store in store_list:
        for method, d in store.items():
            s    = np.asarray(d["s"], float)
            mask = np.ones(s.size, dtype=bool)
            if s_min is not None:
                mask &= s >= (s_min - start_cutoff)
            if s_cutoff is not None:
                mask &= s <= (s_cutoff - start_cutoff)

            n_keep = int(np.count_nonzero(mask))
            if n_keep == s.size:
                continue
            if n_keep < 4:
                print(f"WARNING [{label} – {method}]: only {n_keep} points in "
                      f"[{min_txt}, {max_txt}] m -> left untrimmed.")
                continue

            for key in ("x", "y", "z", "s", "t", "kappa"):
                arr = d.get(key)
                if arr is not None and np.ndim(arr) > 0 and np.size(arr) == mask.size:
                    d[key] = np.asarray(arr)[mask]
            d["spline_z"]     = sc.make_splrep(d["s"], d["z"],     s=smoothing_factor)
            d["spline_kappa"] = sc.make_splrep(d["s"], d["kappa"], s=smoothing_factor)
            print(f"  [{label} – {method}]: trimmed to s in [{min_txt}, {max_txt}] m "
                  f"({s.size} -> {n_keep} points)")


def fine_align_to_window(store_list, ref_store, ref_substr, win_left, win_right,
                         search_m=2.0, step_m=0.05, w_mae=0.5, field="z",
                         smoothing_factor=0.0):
    """Slide each method ±search_m to best match the physical
    reference inside [win_left, win_right], then apply the best shift.

    Score per candidate shift δ: w_mae * MAE_norm + (1 - w_mae) * (1 - corr),
    where the method curve is evaluated as spline(s - δ) against the reference
    on a uniform grid over the window. Lower is better. MAE is normalised by the
    reference's value range so it is comparable to the (1 - correlation) term.
    """
    spline_key = "spline_z" if field == "z" else "spline_kappa"

    # Locate the physical reference profile.
    ref = next((d for m, d in ref_store.items() if ref_substr in m), None)
    if ref is None:
        print("WARNING: fine-align reference not found -> skipped.")
        return

    s_grid     = np.linspace(win_left, win_right, 200)
    ref_vals   = np.asarray(ref[spline_key](s_grid), float)
    ref_finite = ref_vals[np.isfinite(ref_vals)]
    ref_range  = (ref_finite.max() - ref_finite.min()) if ref_finite.size else 0.0
    if not np.isfinite(ref_range) or ref_range <= 0:
        ref_range = 1.0
    ref_std = np.std(ref_vals)

    shifts = np.arange(-search_m, search_m + 1e-9, step_m)
    for label, store in store_list:
        for method, d in store.items():
            if ref_substr in method:
                continue  # never shift the reference itself
            sp = d[spline_key]
            best_shift, best_score = 0.0, np.inf
            for delta in shifts:
                vals = np.asarray(sp(s_grid - delta), float)
                diff = vals - ref_vals
                m    = np.isfinite(diff)
                if np.count_nonzero(m) < 4:
                    continue
                mae_norm = np.mean(np.abs(diff[m])) / ref_range
                if np.std(vals[m]) < 1e-12 or ref_std < 1e-12:
                    corr = 0.0
                else:
                    corr = np.corrcoef(vals[m], ref_vals[m])[0, 1]
                    if not np.isfinite(corr):
                        corr = 0.0
                score = w_mae * mae_norm + (1 - w_mae) * (1 - corr)
                if score < best_score:
                    best_score, best_shift = score, float(delta)

            s_new = d["s"] + best_shift
            d["s"] = s_new
            d["spline_z"]     = sc.make_splrep(s_new, d["z"],     s=smoothing_factor)
            d["spline_kappa"] = sc.make_splrep(s_new, d["kappa"], s=smoothing_factor)
            print(f"  [{label} – {method}]: fine shift {best_shift:+.3f} m "
                  f"(score {best_score:.4f})")


# ── Load result files ─────────────────────────────────────────────────────────
# Access variables by method name, e.g. results_validation["AHN4 DTM"]["x"].
try:
    results_validation = load_results(VALIDATION_PREFIXES, "VALIDATION", VALIDATION_DIR)
except FileNotFoundError:
    sys.exit("Error: No files found for VALIDATION. Check the filename and that it ends in .npz")

try:
    results_calculation = load_results(CALCULATION_PREFIXES, "CALCULATION", CALCULATION_DIR)
except FileNotFoundError:
    sys.exit("Error: No files found for CALCULATION. Check the filename and that it ends in .npz")


# ── Per-method preparation (splines, derived s/z/kappa) ───────────────────────
print("\n── Check if data is consistent ──────────────────────────────────────")
for label, store in [("VALIDATION", results_validation), ("CALCULATION", results_calculation)]:
    for method, d in store.items():
        x, y, z, s, kappa = d["x"], d["y"], d["z"], d["s"], d["kappa"]

        if s is None or np.ndim(z) == 0 or np.size(z) == 0:   # derive s from x, y if missing
            print(f"Warning: No s data for {label} - {method}, calculating from x and y")
            s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])

        if z is None or np.ndim(z) == 0 or np.size(z) == 0:   # derive z from slope and s if missing
            print(f"Warning: No z data for {label} - {method}, calculating from slope")
            dz_seg = np.tan(np.radians(d["slope_deg"])) * np.diff(s)
            z = np.concatenate([[0], np.cumsum(dz_seg)])

        if len(z) != len(s):
            print(z)
            print(s)
            sys.exit(f"Error [{label} – {method}]: z has {len(z)} points but s has "
                     f"{len(s)} points. FIX THIS")

        valid = np.isfinite(z)        # keep only finite points
        z, s  = z[valid], s[valid]

        sp_z = sc.make_splrep(s, z, s=SMOOTHENING_FACTOR)
        if kappa is None or np.ndim(kappa) == 0 or np.size(kappa) == 0:   # derive kappa from z spline if missing
            dzds   = sp_z.derivative(nu=1)(s)
            d2zds2 = sp_z.derivative(nu=2)(s)
            kappa  = abs(d2zds2) / (1.0 + dzds ** 2) ** 1.5   # curvature of a 2D curve z(s)
        sp_k = sc.make_splrep(s, kappa, s=SMOOTHENING_FACTOR)

        d["spline_z"]     = sp_z
        d["spline_kappa"] = sp_k
        d["kappa"]        = kappa
        d["s"]            = s
        d["z"]            = z

        if SMOOTHENING_FACTOR != 0:
            delta_z  = sp_z(s) - z
            rmse     = np.sqrt(np.mean(delta_z ** 2))
            mae      = np.mean(np.abs(delta_z))
            max_diff = np.max(np.abs(delta_z))
            print(f"\n  [{label} – {method}]")
            print(f"    RMSE  (spline vs raw z): {rmse:.6f} m")
            print(f"    MAE   (spline vs raw z): {mae:.6f} m")
            print(f"    Max Δ (spline vs raw z): {max_diff:.6f} m")

print("Done with reading every input file")


# ── Reference start: user-given s where the physical measurement rises ────────
ref_start = 0.0
if ALIGN_TO_REFERENCE:
    ref_start = START_VALIDATION
    print(f"\nReference start (user-given) s={ref_start:.3f} m")

# ── Build the evaluation window, fine-align each method, then trim to it ───────
# Window: [ref_start - WINDOW_BACK_M, min(ref_start + WINDOW_FWD_M, physical peak)].

peak_s    = find_profile_peak_s(results_validation, PHYS_MEAS_SUBSTR, ref_start)
win_left  = ref_start - WINDOW_BACK_M
win_right = ref_start + WINDOW_FWD_M
if peak_s is not None and peak_s:
    print(f"Physical-measurement peak at s={peak_s:.3f} m.")
    win_right = min(win_right, peak_s)
else:
    print("No physical-measurement peak found -> using ref_start + "
            f"{WINDOW_FWD_M:.1f} m as the right edge.")

if win_right <= win_left:
    print(f"WARNING: degenerate window [{win_left:.3f}, {win_right:.3f}] m "
            "-> no fine-align / trim applied.")
else:
    print(f"Evaluation window: s in [{win_left:.3f}, {win_right:.3f}] m "
            f"(ref_start={ref_start:.3f} m).")

    # Per-method fine alignment to the best-fitting shift inside the window.
    if FINE_ALIGN:
        print("\n── Fine-aligning each method to best fit in window ──────────────────")
        fine_align_to_window(
            [("VALIDATION", results_validation), ("CALCULATION", results_calculation)],
            ref_store=results_validation,
            ref_substr=PHYS_MEAS_SUBSTR,
            win_left=win_left,
            win_right=win_right,
            search_m=FINE_ALIGN_SEARCH_M,
            step_m=FINE_ALIGN_STEP_M,
            w_mae=FINE_ALIGN_W_MAE,
            field=FINE_ALIGN_FIELD,
            smoothing_factor=SMOOTHENING_FACTOR,
        )

    # Trim every profile to the common window for a fair comparison.
    print("\n── Trimming profiles to evaluation window ───────────────────────────")
    trim_profiles_to_cutoff(
        [("VALIDATION", results_validation), ("CALCULATION", results_calculation)],
        ref_start,
        s_min=win_left,
        s_cutoff=win_right,
        smoothing_factor=SMOOTHENING_FACTOR,
    )

    # ── Re-zero every profile so the rise starts at the origin (0, 0) ──────
    # z -> z - z(0)  (each profile's height at s=0 in data-space becomes 0)
    print("\n── Re-zeroing z so the height at the start equals 0 ────────────────")
    for label, store in [("VALIDATION", results_validation), ("CALCULATION", results_calculation)]:
        for method, d in store.items():
            z0 = float(d["spline_z"](0.0))   # height at s=0 in data-space (origin = start_cutoff)
            d["z"] = np.asarray(d["z"], float) - z0
            d["spline_z"]     = sc.make_splrep(d["s"], d["z"],     s=SMOOTHENING_FACTOR)
            d["spline_kappa"] = sc.make_splrep(d["s"], d["kappa"], s=SMOOTHENING_FACTOR)
            print(f"  [{label} – {method}]: z -= {z0:+.3f} m")


# ── Cross comparison (validation vs calculation) ──────────────────────────────
# Curvature (kappa) RMSE/MAE kept as before; height (z) RMSE/MAE added, and the
# in-threshold fraction is based on the height difference |Δz|.
comparisons = []
for vm, vd in results_validation.items():       # vm: validation method, vd: validation data
    for cm, cd in results_calculation.items():
        s_min = max(vd["s"].min(), cd["s"].min())
        s_max = min(vd["s"].max(), cd["s"].max())
        s_row = np.linspace(s_min, s_max, 500)  # uniform sample grid for both profiles

        kappa_diff = vd["spline_kappa"](s_row) - cd["spline_kappa"](s_row)
        z_diff     = vd["spline_z"](s_row)     - cd["spline_z"](s_row)

        rmse_kappa   = np.sqrt(np.mean(kappa_diff ** 2))
        mae_kappa    = np.mean(np.abs(kappa_diff))
        rmse_z       = np.sqrt(np.mean(z_diff ** 2))
        mae_z        = np.mean(np.abs(z_diff))
        frac_in_thr  = np.mean(np.abs(z_diff) < THRESHOLD)

        comparisons.append({
            "val": vm, "calc": cm, "s": s_row,
            "kappa_diff": kappa_diff, "z_diff": z_diff,
            "rmse_kappa": rmse_kappa, "mae_kappa": mae_kappa,
            "rmse_z": rmse_z, "mae_z": mae_z,
            "frac_in_threshold": frac_in_thr,
        })
        print(f"\n  [{vm}  vs  {cm}]")
        print(f"    RMSE (kappa):  {rmse_kappa:.4f} 1/m")
        print(f"    MAE  (kappa):  {mae_kappa:.4f} 1/m")
        print(f"    RMSE (height): {rmse_z:.4f} m")
        print(f"    MAE  (height): {mae_z:.4f} m")
        print(f"    In threshold (|Δz| < {THRESHOLD} m): {frac_in_thr:.4f}")


# ── CSV export ────────────────────────────────────────────────────────────────
raw_t    = results_calculation[next(iter(results_calculation))]["t"]
t        = str(raw_t.flat[0]) if hasattr(raw_t, "flat") else str(raw_t)
csv_path = os.path.join(r"C:\Users\karel\OneDrive\BEP\11_02_12", f"{t}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Cross comparison (validation vs calculation)"])
    writer.writerow(["Val method", "Calc method",
                     "RMSE kappa (1/m)", "MAE kappa (1/m)",
                     "RMSE height (m)", "MAE height (m)",
                     f"In threshold (|delta_z| < {THRESHOLD} m)"])
    for c in comparisons:
        writer.writerow([c["val"], c["calc"],
                         f"{c['rmse_kappa']:.6f}", f"{c['mae_kappa']:.6f}",
                         f"{c['rmse_z']:.6f}", f"{c['mae_z']:.6f}",
                         f"{c['frac_in_threshold']:.6f}"])

print(f"\nStatistics saved to {csv_path}")


# ── Plotting ──────────────────────────────────────────────────────────────────
n_rows = 2 + (1 if comparisons else 0)
fig, axes = plt.subplots(n_rows, 1, figsize=(12, 5 * n_rows))
ax_z = axes[0]
ax_k = axes[1]
ax_d = axes[2] if comparisons else None

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plotted   = set()
color_idx = 0
for label, store, marker, ls in [
    ("Val",  results_validation,  "o", "-"),
    ("Calc", results_calculation, "x", "--"),
]:
    for method, d in store.items():
        tag = f"{label}: {method}"
        if tag in plotted:   # avoid duplicate legend entries
            continue
        plotted.add(tag)
        c     = colors[color_idx % len(colors)]   # cycle colours if there are more methods than colours
        s_lin = np.linspace(d["s"][0], d["s"][-1], 500)

        # Elevation
        ax_z.scatter(d["s"], d["z"], s=20, color=c, marker=marker, alpha=0.7, zorder=3)
        ax_z.plot(s_lin, d["spline_z"](s_lin), color=c, linestyle=ls, label=tag)

        # Curvature
        ax_k.scatter(d["s"], d["kappa"], s=25, color=c, marker=marker, zorder=3)
        ax_k.plot(s_lin, d["spline_kappa"](s_lin), color=c, linestyle=ls, label=tag)

        color_idx += 1

ax_z.set_xlabel("s (m)")
ax_z.set_ylabel("z (m, relative to rise start)")
ax_z.set_title("Elevation profile")
ax_z.set_xlim(-WINDOW_BACK_M, WINDOW_FWD_M)
ax_z.legend()
ax_z.grid(True)

ax_k.set_xlabel("s (m)")
ax_k.set_ylabel("κ (1/m)")
ax_k.set_title("Curvature profile")
ax_k.set_xlim(-WINDOW_BACK_M, WINDOW_FWD_M)
ax_k.legend()
ax_k.set_yscale("log")
ax_k.set_ylim(bottom=BOTTOM_LIMIT_KAPPA)
ax_k.grid(True)

if ax_d is not None:
    for c in comparisons:
        ax_d.plot(c["s"], c["z_diff"], label=f"{c['val']} − {c['calc']}")
    ax_d.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_d.set_xlabel("s (m)")
    ax_d.set_ylabel("Δz (m)")
    ax_d.set_title("Elevation difference (Validation − Calculation)")
    ax_d.set_xlim(-WINDOW_BACK_M, WINDOW_FWD_M)
    ax_d.legend()
    ax_d.grid(True)

plt.tight_layout(h_pad=6)
plt.show()
