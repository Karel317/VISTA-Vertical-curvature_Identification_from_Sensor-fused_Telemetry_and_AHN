import os
import sys
import csv
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt

# TODO
# angles when you are on a bridge
# slope spline moving average filter - should be done


# ── Files ─────────────────────────────────────────────────────────────────────
# Physical-measurement results may be in a different timestamp folder than the
# LiDAR/calculation results (they are run separately and saved under their own timestamp).
VALIDATION_DIR   = r"D:\Validation_results\Physical_validation_data"   # physical meas .npz files
CALCULATION_DIR  = r"D:\Validation_results\2026_05_22\10_35_55\1779438955"   # lidar/calculation .npz files

# D:\Validation_results\2026_05_22\10_42_57\1779439377
# D:\Validation_results\2026_05_22\10_44_36\1779439476

# D:\Validation_results\2026_05_22\10_51_56\1779439916
# D:\Validation_results\2026_05_22\10_52_21\1779439941
# D:\Validation_results\2026_05_22\10_52_44\1779439964
# D:\Validation_results\2026_05_22\10_52_57\1779439977
# D:\Validation_results\2026_05_22\10_53_17\1779439997

# D:\Validation_results\2026_05_22\11_02_44\1779440564
# D:\Validation_results\2026_05_22\11_04_20\1779440660

# Prefixes of the result files to load (filenames without the .npz extension).
VALIDATION_PREFIXES = [
    #"Physical_meas_Sint_Jorispad_Brug",
    #"Physical_meas_Sint_Jorispad_Brug_backwards",
    "Physical_meas_Flat_25m",
    #"Physical_meas_Proteus_Brug",
    #"Physical_meas_Station_Brug",
    #"Physical_meas_Station_Brug_backwards"
]
CALCULATION_PREFIXES = [
    #"AHN5_DSM",
    #"AHN5_DTM",
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

# ── Comparison input ───────────────────────────────────────────────────────────────────
THRESHOLD          = 0.1    # m   : |Δz| a threshold used in the comparison as a metric

# ── General ───────────────────────────────────────────────────────────────────
SMOOTHENING_FACTOR = 0.5        #       spline smoothing as a *per-point* budget:
                                #       scipy's s is an absolute sum-of-squared-residuals
                                #       budget, so the effective s = factor * N points
                                #       (1.0 -> s == number of points; 0 = no smoothing)
BOTTOM_LIMIT_KAPPA = 1e-8    # 1/m : lower y-limit for the (log-scaled) curvature plot
PHYS_MEAS_SUBSTR   = "Physical_meas"  # substring identifying the physical measurement profile

# ── Evaluation window ─────────────────────────────────────────────────────────
# Window = [START_VALIDATION, right], right = physical peak, or the end of the
# physical measurement when there is no peak. After trimming, the window start
# is re-origined to s=0.
ALIGN_TO_REFERENCE = True
START_VALIDATION   = 0.0     # m   : user-given s where the physical measurement rises
WINDOW_BACK_M      = 5
WINDOW_FWD_M       = 20.0    # m   : fallback right edge (START_VALIDATION + this) if no peak/end is found

# ── Per-method fine alignment (corrects the ~±1 m offset between methods) ──────
FINE_ALIGN          = True
FINE_ALIGN_FIELD    = "z"    #       score on elevation ("z", robust) or curvature ("kappa")
FINE_ALIGN_SEARCH_M = 0.0    # m   : slide each method ±this many metres
FINE_ALIGN_STEP_M   = 0.02   # m   : search resolution
FINE_ALIGN_W_MAE    = 0.5    #       score = W*MAE_norm + (1-W)*(1-correlation)

ON_BRIDGE = False
# ─────────────────────────────────────────────────────────────────────────────


def _spl(x, y, factor):
    """make_splrep with the smoothing budget scaled to the number of points.
    scipy's `s` is an absolute sum-of-squared-residuals budget that must grow
    with the sample count, so we pass s = factor * len(y) (factor == per-point
    budget; 1.0 makes s equal to the number of data points)."""
    return sc.make_splrep(x, y, s=factor * len(y))


def _name_matches_prefix(stem, prefix):
    """True if `stem` is `prefix`, or `prefix` followed only by a `_<digits>`
    timestamp suffix. This keeps e.g. 'AHN5_DTM_1779439997' but rejects a
    sibling like 'Physical_meas_Station_Brug_backwards' that would otherwise
    collide with (and silently overwrite) 'Physical_meas_Station_Brug'."""
    if stem == prefix:
        return True
    if stem.startswith(prefix):
        rest = stem[len(prefix):]
        return rest.startswith("_") and rest[1:].isdigit()
    return False


def load_results(prefixes, source_label, results_dir):
    """Load the .npz result files in results_dir whose name starts with any of
    `prefixes` into a dict"""
    files = [
        fname[:-4]
        for fname in os.listdir(results_dir)
        if fname.endswith(".npz") and any(_name_matches_prefix(fname[:-4], p) for p in prefixes)
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


def find_profile_peak_s(store, method_substr):
    """Return the physical-local distance (s) of the maximum z of the first
    profile whose name contains `method_substr`, or None if it cannot be found.
    Works purely in the physical measurement's own coordinate frame."""
    for method, d in store.items():
        if method_substr in method:
            z = np.asarray(d["z"], float)
            s = np.asarray(d["s"], float)
            if z.size == 0 or s.size != z.size or not np.any(np.isfinite(z)):
                print(f"WARNING: peak profile '{method}' has no usable z/s.")
                return None
            mask = np.isfinite(z)
            zf = z[mask]
            if np.ptp(zf) < 1e-9:   # flat reference (e.g. Physical_meas_Flat) has no real peak
                print(f"Profile '{method}' is flat -> no peak; window falls back to the end.")
                return None
            return float(s[mask][int(np.argmax(zf))])
    print(f"WARNING: no profile matching '{method_substr}' found for peak.")
    return None


def trim_physical_validation(store_list, left_P, right_P, smoothing_factor):
    """Cut the physical-measurement profile to its own (physical-local) window
    [left_P, right_P], then re-origin so left_P becomes s=0 and refit its splines.
    left_P already encodes the bridge/non-bridge mode (0 off-bridge,
    START_VALIDATION - WINDOW_BACK_M on the bridge)."""
    for label, store in store_list:
        for method, d in store.items():
            s    = np.asarray(d["s"], float)
            mask = (s >= left_P) & (s <= right_P)

            n_keep = int(np.count_nonzero(mask))
            if n_keep < 4:
                print(f"WARNING [{label} – {method}]: only {n_keep} points in "
                    f"[{left_P:.3f}, {right_P:.3f}] m -> left untrimmed.")
                continue

            for key in ("x", "y", "z", "s", "t", "kappa"):
                arr = d.get(key)
                if arr is not None and np.ndim(arr) > 0 and np.size(arr) == mask.size:
                    d[key] = np.asarray(arr)[mask]
            d["s"] = np.asarray(d["s"], float) - left_P   # re-origin: window start -> 0
            d["spline_z"]     = _spl(d["s"], d["z"],     smoothing_factor)
            d["spline_kappa"] = _spl(d["s"], d["kappa"], smoothing_factor)
            print(f"  [{label} – {method}]: trimmed to s in "
                f"[{left_P:.3f}, {right_P:.3f}] m and re-origined "
                f"({s.size} -> {n_keep} points)")
    return None


def trim_profiles_to_window(store_list, win_left, win_right, smoothing_factor=0.0):
    """Cut every profile to s in [win_left, win_right] (data-space), then
    re-origin so win_left becomes s=0 and refit its splines.

    A method that has fewer than 4 points inside the window cannot be placed in
    the common (re-origined) frame, so it is dropped from its store entirely —
    otherwise it would be plotted/compared in its raw coordinates and look
    misaligned (e.g. a method whose odometry under-scaled distance never reaches
    the window)."""
    for label, store in store_list:
        drop = []
        for method, d in store.items():
            s    = np.asarray(d["s"], float)
            mask = (s >= win_left) & (s <= win_right)

            n_keep = int(np.count_nonzero(mask))
            if n_keep < 4:
                print(f"WARNING [{label} – {method}]: only {n_keep} of {s.size} points in "
                      f"[{win_left:.3f}, {win_right:.3f}] m (its s=[{s.min():.2f}, {s.max():.2f}]) "
                      f"-> DROPPED (cannot align to the window).")
                drop.append(method)
                continue

            for key in ("x", "y", "z", "s", "t", "kappa"):
                arr = d.get(key)
                if arr is not None and np.ndim(arr) > 0 and np.size(arr) == mask.size:
                    d[key] = np.asarray(arr)[mask]
            d["s"] = np.asarray(d["s"], float) - win_left   # re-origin: window start -> 0
            d["spline_z"]     = _spl(d["s"], d["z"],     smoothing_factor)
            d["spline_kappa"] = _spl(d["s"], d["kappa"], smoothing_factor)
            print(f"  [{label} – {method}]: trimmed to s in "
                  f"[{win_left:.3f}, {win_right:.3f}] m and re-origined "
                  f"({s.size} -> {n_keep} points)")

        for method in drop:   # remove un-alignable methods after iterating
            del store[method]


def fine_align_to_window(store_list, ref_store, ref_substr, left_C, right_C,
                         offset, search_m=2.0, step_m=0.05, w_mae=0.5,
                         smoothing_factor=0.0):
    """Slide each calculation method ±search_m to best match the physical
    reference over the calc-frame window [left_C, right_C]. The reference lives
    in the physical-local frame, so it is evaluated at (s_grid - offset) to bring
    it into the calc frame, where s_calc = s_phys + offset.

    The comparison grid is the *fixed* window [left_C, right_C] (which maps onto
    the reference's valid range, so the reference is never extrapolated); only
    the per-method shift delta sweeps ±search_m. Keeping the grid independent of
    search_m means enlarging the search range cannot move the best fit on its
    own — it only lets the optimum be found further out."""
    ref = next((d for m, d in ref_store.items() if ref_substr in m), None)
    if ref is None:
        print("WARNING: fine-align reference not found -> skipped.")
        return

    s_grid     = np.linspace(left_C, right_C, 500)
    ref_vals   = np.asarray(ref["spline_z"](s_grid - offset), float)
    ref_range  = (ref_vals.max() - ref_vals.min())
    ref_std = np.std(ref_vals)

    shifts = np.arange(-search_m, search_m + 1e-9, step_m)
    for label, store in store_list:
        for method, d in store.items():
            if ref_substr in method:
                continue  # never shift the reference itself
            sp = d["spline_z"]
            best_shift, best_score = 0.0, np.inf
            for delta in shifts:
                vals = np.asarray(sp(s_grid - delta), float)
                m    = np.isfinite(vals) & np.isfinite(ref_vals)
                if np.count_nonzero(m) < 4:
                    continue
                # Remove each curve's vertical offset before measuring the height
                # mismatch, so the score reflects *shape* alignment rather than the
                # constant gap between the two height datums (fine-align runs before
                # the z -= z[0] re-zeroing). Without this, a baseline difference on a
                # monotonic ramp biases the optimum: the method slides along its own
                # slope to absorb the offset instead of aligning features.
                vc = vals[m] - np.mean(vals[m])
                rc = ref_vals[m] - np.mean(ref_vals[m])
                mae_norm = np.mean(np.abs(vc - rc)) / ref_range
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
            d["spline_z"]     = _spl(s_new, d["z"],     smoothing_factor)
            d["spline_kappa"] = _spl(s_new, d["kappa"], smoothing_factor)
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
        sp_z = _spl(s, z, SMOOTHENING_FACTOR)   # always needed for spline_z / height
        if d.get("slope_deg") is not None:
            # Reproduce the notebook: curvature from the slope, not from z. The
            # plane-fit methods store the slope as an *angle* (deg), so the
            # height gradient is dz/ds = tan(slope); the Frenet denominator uses
            # that ratio (1 + (dz/ds)^2), and the numerator is one derivative of it.
            dzds    = np.tan(np.radians(np.asarray(d["slope_deg"], float)[valid]))
            sp_dzds = _spl(s, dzds, SMOOTHENING_FACTOR)
            d2zds2  = sp_dzds.derivative(nu=1)(s)
            kappa   = d2zds2 / (1.0 + dzds ** 2) ** 1.5       # signed: + valley, - crest
        else:
            dzds   = sp_z.derivative(nu=1)(s)
            d2zds2 = sp_z.derivative(nu=2)(s)
            kappa  = d2zds2 / (1.0 + dzds ** 2) ** 1.5        # signed curvature of a 2D curve z(s)
        sp_k = _spl(s, kappa, SMOOTHENING_FACTOR)

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
# The physical reference and the calculation methods live in different local
# frames, related by  s_calc = s_phys + offset.  The offset flips sign with
# ON_BRIDGE: off-bridge we are START_VALIDATION *in front of* the measurement
# (offset = +START_VALIDATION); on the bridge we are START_VALIDATION *into* it
# (offset = -START_VALIDATION).  The window is built in the physical-local frame
# (left_P, right_P) and mapped to the calc frame (left_C, right_C).
offset = -START_VALIDATION if ON_BRIDGE else START_VALIDATION

peak_P = find_profile_peak_s(results_validation, PHYS_MEAS_SUBSTR)    # physical-local peak, or None
end_P  = float(np.max(next(iter(results_validation.values()))["s"]))  # physical end (L_P)

if ON_BRIDGE:
    left_P   = START_VALIDATION - WINDOW_BACK_M
    fwd_edge = START_VALIDATION + WINDOW_FWD_M
    right_P  = min(peak_P, fwd_edge) if peak_P is not None else min(end_P, fwd_edge)
else:
    left_P   = 0.0
    right_P  = min(peak_P, end_P) if peak_P is not None else min(end_P, WINDOW_FWD_M)

left_C, right_C = left_P + offset, right_P + offset   # calc-frame window
window_len = right_P - left_P   # post-trim length (same in both frames), used for plot x-limits

if right_P <= left_P:
    print(f"WARNING: degenerate window: physical [{left_P:.3f}, {right_P:.3f}] m "
            "-> no fine-align / trim applied.")
else:
    print(f"Evaluation window: physical s in [{left_P:.3f}, {right_P:.3f}] m, "
            f"calc s in [{left_C:.3f}, {right_C:.3f}] m (offset={offset:+.3f} m).")

    # Per-method fine alignment to the best-fitting shift inside the window.
    if FINE_ALIGN:
        print("\n── Fine-aligning each method to best fit in window ──────────────────")
        fine_align_to_window(
            [("CALCULATION", results_calculation)],
            ref_store=results_validation,
            ref_substr=PHYS_MEAS_SUBSTR,
            left_C=left_C,
            right_C=right_C,
            offset=offset,
            search_m=FINE_ALIGN_SEARCH_M,
            step_m=FINE_ALIGN_STEP_M,
            w_mae=FINE_ALIGN_W_MAE,
            smoothing_factor=SMOOTHENING_FACTOR,
        )

    # Trim every profile to its window and re-origin its start to s=0. Physical
    # validation is in its own frame, so it is trimmed with (left_P, right_P);
    # the calculation methods use the mapped calc-frame window (left_C, right_C).
    print("\n── Trimming profiles to evaluation window ───────────────────────────")
    trim_physical_validation(
        [("VALIDATION", results_validation)],
        left_P=left_P,
        right_P=right_P,
        smoothing_factor=SMOOTHENING_FACTOR,
    )
    trim_profiles_to_window(
        [("CALCULATION", results_calculation)],
        left_C,
        right_C,
        smoothing_factor=SMOOTHENING_FACTOR,
    )

    # ── Re-zero every profile so the rise starts at the origin (0, 0) ──────
    # z -> z - z(0)  (each profile's height at s=0 in data-space becomes 0)
    print("\n── Re-zeroing z so the height at the start equals 0 ────────────────")
    for label, store in [("VALIDATION", results_validation), ("CALCULATION", results_calculation)]:
        for method, d in store.items():
            z0 = float(d["spline_z"](0.0))   # height at s=0 in data-space (origin = start_cutoff)
            d["z"] = np.asarray(d["z"], float) - z0
            d["spline_z"]     = _spl(d["s"], d["z"],     SMOOTHENING_FACTOR)
            d["spline_kappa"] = _spl(d["s"], d["kappa"], SMOOTHENING_FACTOR)
            print(f"  [{label} – {method}]: z -= {z0:+.3f} m")


# ── Cross comparison (validation vs calculation) ──────────────────────────────
# Curvature (kappa) RMSE/MAE kept as before; height (z) RMSE/MAE added, and the
# in-threshold fraction is based on the height difference |Δz|.
comparisons = []

# Physical (reference) curvature mean/median, computed ONCE per validation profile
# on its own full (trimmed) grid. Doing this here — instead of inside the calc loop
# on the per-method overlap grid (s_row) — gives a single physical mean/median that
# does not change from one calc method to the next.
ref_stats = {}
for vm, vd in results_validation.items():
    s_ref            = np.linspace(vd["s"].min(), vd["s"].max(), 500)
    val_kappa_full   = vd["spline_kappa"](s_ref)
    ref_stats[vm]    = (np.mean(val_kappa_full), np.median(val_kappa_full))

for vm, vd in results_validation.items():       # vm: validation method, vd: validation data
    mean_kappa_ref, median_kappa_ref = ref_stats[vm]   # one physical mean/median for all methods
    for cm, cd in results_calculation.items():
        s_min = max(vd["s"].min(), cd["s"].min())
        s_max = min(vd["s"].max(), cd["s"].max())
        s_row = np.linspace(s_min, s_max, 500)  # uniform sample grid for both profiles

        val_kappa  = vd["spline_kappa"](s_row)          # reference (validation) curvature on the grid
        calc_kappa = cd["spline_kappa"](s_row)          # calc-method curvature on the grid
        kappa_diff = val_kappa - calc_kappa
        z_diff     = vd["spline_z"](s_row)     - cd["spline_z"](s_row)

        rmse_kappa       = np.sqrt(np.mean(kappa_diff ** 2))
        mae_kappa        = np.mean(np.abs(kappa_diff))
        mean_kappa       = np.mean(calc_kappa)          # mean curvature of the calc method
        median_kappa     = np.median(calc_kappa)        # median curvature of the calc method
        rmse_z       = np.sqrt(np.mean(z_diff ** 2))
        mae_z        = np.mean(np.abs(z_diff))
        frac_in_thr  = np.mean(np.abs(z_diff) < THRESHOLD)

        comparisons.append({
            "val": vm, "calc": cm, "s": s_row,
            "kappa_diff": kappa_diff, "z_diff": z_diff,
            "rmse_kappa": rmse_kappa, "mae_kappa": mae_kappa,
            "mean_kappa": mean_kappa, "median_kappa": median_kappa,
            "mean_kappa_ref": mean_kappa_ref, "median_kappa_ref": median_kappa_ref,
            "rmse_z": rmse_z, "mae_z": mae_z,
            "frac_in_threshold": frac_in_thr,
        })
        print(f"\n  [{vm}  vs  {cm}]")
        print(f"    RMSE (kappa):        {rmse_kappa:.4f} 1/m")
        print(f"    MAE  (kappa):        {mae_kappa:.4f} 1/m")
        print(f"    Mean (kappa):        {mean_kappa:.4f} 1/m")
        print(f"    Median (kappa):      {median_kappa:.4f} 1/m")
        print(f"    Mean (kappa, ref):   {mean_kappa_ref:.4f} 1/m")
        print(f"    Median (kappa, ref): {median_kappa_ref:.4f} 1/m")
        print(f"    RMSE (height): {rmse_z:.4f} m")
        print(f"    MAE  (height): {mae_z:.4f} m")
        print(f"    In threshold (|Δz| < {THRESHOLD} m): {frac_in_thr:.4f}")


# ── CSV export ────────────────────────────────────────────────────────────────
raw_t    = results_calculation[next(iter(results_calculation))]["t"]
t        = str(raw_t.flat[0]) if hasattr(raw_t, "flat") else str(raw_t)
csv_path = os.path.join(r"D:\Validation_results\Statistics", f"{t}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Cross comparison (validation vs calculation)"])
    writer.writerow(["Val method", "Calc method",
                     "RMSE kappa (1/m)", "MAE kappa (1/m)",
                     "Mean kappa (1/m)", "Median kappa (1/m)",
                     "Mean kappa ref (1/m)", "Median kappa ref (1/m)",
                     "RMSE height (m)", "MAE height (m)",
                     f"In threshold (|delta_z| < {THRESHOLD} m)"])
    for c in comparisons:
        writer.writerow([c["val"], c["calc"],
                         f"{c['rmse_kappa']:.6f}", f"{c['mae_kappa']:.6f}",
                         f"{c['mean_kappa']:.6f}", f"{c['median_kappa']:.6f}",
                         f"{c['mean_kappa_ref']:.6f}", f"{c['median_kappa_ref']:.6f}",
                         f"{c['rmse_z']:.6f}", f"{c['mae_z']:.6f}",
                         f"{c['frac_in_threshold']:.6f}"])

print(f"\nStatistics saved to {csv_path}")


# ── Publication-quality plotting (IEEE/CVPR style) ────────────────────────────
# Two stacked panels (elevation + curvature, log-scaled and clipped to a sensible
# range so artefact zero-crossings don't dominate) with a single shared legend to
# the right. Sibling methods share a colour and are told apart by line style.

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "cm",            # Computer Modern for the κ glyph
    "axes.unicode_minus": False,
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "legend.fontsize":   8,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.linewidth":    0.8,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
})

GT_COLOR = "#0072B2"   # ground-truth blue (reserved; not reused by any method)

# Internal method name -> short publication label.
SHORT_NAMES = {
    "AHN5_DSM":                 "AHN5 DSM",
    "AHN5_DTM":                 "AHN5 DTM",
    "EKF_curvature_validation": "EKF",
    "KISS_ICP":                 "KISS-ICP",
    "Z_positional_tracking":    "ZED positional tracking",
    "height-deriv_csf":         "Height-deriv (CSF)",
    "height-deriv_patchwork":   "Height-deriv (Patchwork++)",
    "PCA_csf":                  "PCA (CSF)",
    "PCA_patchwork":            "PCA (Patchwork++)",
    "RANSAC_csf":               "RANSAC (CSF)",
    "RANSAC_patchwork":         "RANSAC (Patchwork++)",
}

# Legend order: LiDAR methods, then IMU/GPS methods, then the AHN5 aerial reference.
# (name, colour, line style). Sibling backends (CSF "--" vs Patchwork++ "-.") share
# a colour and are distinguished by line style, so near-identical colours are never
# relied on alone. Standalone methods use dotted lines with their own colour.
METHOD_STYLE = [
    ("height-deriv_csf",         "#D55E00", "--"),   # LiDAR ground segmentation
    ("height-deriv_patchwork",   "#D55E00", "-."),
    ("PCA_csf",                  "#CC79A7", "--"),
    ("PCA_patchwork",            "#CC79A7", "-."),
    ("RANSAC_csf",               "#117733", "--"),
    ("RANSAC_patchwork",         "#117733", "-."),
    ("KISS_ICP",                 "#AA4499", ":"),     # LiDAR odometry
    ("EKF_curvature_validation", "#E41A1C", ":"),     # IMU/GPS (red)
    ("Z_positional_tracking",    "#FF7F0E", ":"),     # (orange)
    ("AHN5_DSM",                 "#666666", "--"),     # AHN5 aerial reference (greyscale)
    ("AHN5_DTM",                 "#000000", "-."),
]


def _norm(name):
    return name.lower().replace(" ", "").replace("-", "").replace("_", "").replace("+", "")


# Map (normalised) stored method keys so we still find them if the saved
# "method" string differs slightly from the prefix above.
_calc_by_norm = {_norm(k): (k, d) for k, d in results_calculation.items()}
_gt_key       = next(iter(results_validation))
_gt           = results_validation[_gt_key]
_x_max        = float(np.asarray(_gt["s"], float).max())


def _draw_panels(ax_z, ax_k):
    """Draw elevation + curvature for the ground truth and every method onto the
    two axes. Returns ordered (handles, labels) for a shared legend."""
    handles, labels = [], []

    # Ground truth: thick solid blue with circular markers (data points).
    sg  = np.asarray(_gt["s"], float)
    sgl = np.linspace(sg.min(), sg.max(), 600)
    h_gt, = ax_z.plot(sgl, _gt["spline_z"](sgl), color=GT_COLOR, lw=1.9, ls="-", zorder=6)
    ax_z.plot(sg, _gt["z"], ls="none", marker="o", ms=3.2, color=GT_COLOR,
              mfc=GT_COLOR, mec="white", mew=0.4, zorder=7)
    ax_k.plot(sgl, _gt["spline_kappa"](sgl), color=GT_COLOR, lw=1.9, ls="-", zorder=6)
    ax_k.plot(sg, _gt["kappa"], ls="none", marker="o", ms=3.2, color=GT_COLOR,
              mfc=GT_COLOR, mec="white", mew=0.4, zorder=7)
    handles.append(h_gt); labels.append("Ground truth")

    # Methods: thinner lines, colour + line style consistent across panels.
    for name, color, ls in METHOD_STYLE:
        found = _calc_by_norm.get(_norm(name))
        if found is None:
            continue                      # method absent or dropped during trimming
        _, d = found
        s  = np.asarray(d["s"], float)
        sl = np.linspace(s.min(), s.max(), 600)
        h, = ax_z.plot(sl, d["spline_z"](sl), color=color, lw=1.1, ls=ls, zorder=3)
        ax_k.plot(sl, d["spline_kappa"](sl), color=color, lw=1.1, ls=ls, zorder=3)
        handles.append(h); labels.append(SHORT_NAMES.get(name, name))

    return handles, labels


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.7", ls="--", lw=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(0, _x_max)


def build_profile_figure():
    """Elevation (a) + log-scaled curvature (b), shared legend on the right."""
    fig, (ax_z, ax_k) = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
    handles, labels = _draw_panels(ax_z, ax_k)

    for ax in (ax_z, ax_k):
        _style_axes(ax)
    ax_z.set_title("(a) Height profile (m)")
    ax_k.set_title("(b) Curvature (1/m)")
    ax_z.set_ylabel("z (m, relative to rise start)")
    ax_k.set_ylabel(r"$\kappa$ (1/m)")
    ax_k.set_xlabel("s (m)")

    # symlog keeps the log-like spread across orders of magnitude while still
    # showing the sign (+ valley / - crest); linear within ±linthresh around 0.
    #ax_k.set_yscale("symlog", linthresh=1e-2)
    ax_k.set_ylim(-0.1, 0.1)
    ax_k.axhline(0.0, color="0.5", lw=0.6, zorder=1)

    fig.tight_layout(rect=(0, 0, 0.72, 1.0))   # reserve the right ~28% for the legend
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.73, 0.5),
               ncol=1, frameon=False, handlelength=2.0, labelspacing=0.6)
    return fig


FIG_DIR = r"C:\Users\Leons\OneDrive - Delft University of Technology\BEP\Figures"
os.makedirs(FIG_DIR, exist_ok=True)

fig = build_profile_figure()
# Encode to an in-memory buffer first, then write the bytes with plain Python
# I/O. Letting savefig write straight to D: fails with OSError [Errno 22] on
# some external/network drives that reject PIL's single large write() call.
import io
_buf = io.BytesIO()
fig.savefig(_buf, format="png", bbox_inches="tight")
with open(os.path.join(FIG_DIR, "elevation_curvature_profiles.png"), "wb") as _f:
    _f.write(_buf.getvalue())
print(f"Saved figure: elevation_curvature_profiles.png  ->  {FIG_DIR}")

plt.show()
