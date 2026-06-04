#!/usr/bin/env python3
"""Validation runner.

Runs the modular validation scripts. Each script is expected to expose a
``run(...)`` function in the same style as ``Validation_curvature_IMU_GPS.py``
and ``AHN5_validation.py``.

Choose what runs in any of three ways:
  * flip the ``enabled`` flag in REGISTRY, or
  * ``python run_validations.py --only ahn5 curvature``, or
  * ``python run_validations.py --skip ahn5``
List everything with ``--list``.

Add a new validation by giving it ONE entry in REGISTRY (see the commented
example). Imports are lazy, so a module you don't run never has to import its
dependencies.

The validation module files must sit next to this file (or otherwise be on the
Python path); the line below adds this file's own folder to the path.
"""
import os
import sys
import argparse
import importlib
import time as _time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── SHARED CONFIG (one place to set the run) ─────────────────────────────────
TIME        = 1779439917.972930459          # reference Unix timestamp
OUTPUT_PATH = r"D:\Validation_results"      # base results folder
GPS_TOPIC   = "/navsat_topic"
PLOT        = False                         # note: each task's plot blocks until its window is closed

# Input files
GPS_MCAP  = r"D:\Data_gathered\2026_05_22\Rosbag\10_50_00\rosbag\rosbag_0.mcap"
IMU_INPUT = r"D:\Data_gathered\2026_05_22\Camera\10_50_00\back_22_05_2026-10_50_00.mcap"  # .mcap / .svo / .svo2

# ── REGISTRY ─────────────────────────────────────────────────────────────────
# name : (enabled, module_name, how-to-call-its-run)
# The lambda receives the imported module `m`; map SHARED CONFIG onto that
# module's own run() signature.
REGISTRY = {
    "ahn5": (
        True, "AHN5_validation",
        lambda m: m.run(GPS_MCAP, TIME, GPS_TOPIC, PLOT),
    ),
    "curvature": (
        True, "Validation_curvature_IMU_GPS",
        lambda m: m.run(IMU_INPUT, GPS_MCAP, TIME, GPS_TOPIC, PLOT, OUTPUT_PATH),
    ),

    # ── add more validations here, same pattern ──
    # "lidar": (
    #     True, "LiDAR_validation",
    #     lambda m: m.run(LIDAR_INPUT, TIME, GPS_TOPIC, PLOT, OUTPUT_PATH),
    # ),
}

# ── Runner ───────────────────────────────────────────────────────────────────

def _resolve(only, skip):
    """Turn --only / --skip into an ordered, validated list of task names."""
    only, skip = set(only or []), set(skip or [])
    for n in sorted((only | skip) - set(REGISTRY)):
        print(f"[warn] unknown task '{n}' — ignored")
    chosen = only if only else {n for n, (en, _, _) in REGISTRY.items() if en}
    return [n for n in REGISTRY if n in chosen and n not in skip]   # keep REGISTRY order


def run_all(only=None, skip=None):
    names = _resolve(only, skip)
    if not names:
        print("Nothing to run.")
        return {}

    results = {}
    for name in names:
        _, modname, call = REGISTRY[name]
        print(f"\n{'=' * 60}\n[run] {name}  ->  {modname}\n{'=' * 60}")
        t0 = _time.time()
        try:
            call(importlib.import_module(modname))
            results[name] = "ok"
            print(f"[done] {name}  ({_time.time() - t0:.1f}s)")
        except Exception:
            results[name] = "FAILED"
            traceback.print_exc()
            print(f"[FAILED] {name}  ({_time.time() - t0:.1f}s)")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for name in names:
        print(f"  {name:<24} {results[name]}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run selected validation modules.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only", nargs="+", metavar="NAME",
                   help="run only these tasks (overrides the enabled flags)")
    g.add_argument("--skip", nargs="+", metavar="NAME",
                   help="run all enabled tasks except these")
    p.add_argument("--list", action="store_true", help="list available tasks and exit")
    args = p.parse_args()

    if args.list:
        print("Available tasks:")
        for name, (en, modname, _) in REGISTRY.items():
            print(f"  {name:<24} {'on ' if en else 'off'}  {modname}")
    else:
        run_all(only=args.only, skip=args.skip)
