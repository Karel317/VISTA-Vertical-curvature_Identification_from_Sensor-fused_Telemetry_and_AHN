"""
pipeline_validation.py
─────────────────────────────────────────────────────────────────────────────
Runs every validation-method module in sequence on ONE shared input file and
ONE shared timestamp.

Each sub-module (validation_curvature.py, Physical_meas.py, run_kiss_icp.py, …)
must expose:

    run(input_file: str, timestamp: str) -> <anything>      # e.g. saved .npz paths

…and must keep its slow, INPUT-INDEPENDENT setup (heavy imports, building a
pyproj Transformer, loading a model, opening a DB, …) at MODULE level so it runs
exactly once — when this pipeline imports the module — and NOT again on every
run() call.  The per-file work (reading `input_file`, computing, saving) goes
inside run().

Why importlib instead of `import validation_curvature`?
The sub-modules live in folders whose names contain spaces / leading digits
("1. Validation Methods", "KISS ICP"). Those aren't valid package names, so a
normal `import` won't find them. Loading by explicit file path sidesteps that.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import importlib.util
import traceback
from pathlib import Path

# ── Shared inputs ─────────────────────────────────────────────────────────────
# Edit here, or override on the CLI with --input-file / --timestamp.
# NOTE: `input_file` is the *primary* recording for the session. Modules that
# need a second file (e.g. curvature needs both a ZED mcap and a GPS mcap) keep
# that extra path as a CONFIG constant inside their own module, OR derive it
# from input_file if you adopt a "session folder" convention.
INPUT_FILE = r"D:\Data_gathered\2026_05_22\Camera\10_50_00\back_22_05_2026-10_50_00.mcap"
TIMESTAMP  = "1779439917.972930459"

# ── Module registry: (logical name, path to the .py relative to THIS file) ────
# Comment out anything you haven't refactored yet — missing files are skipped,
# so the pipeline still runs with whatever is ready.
HERE = Path(__file__).parent
MODULES = [
    ("validation_curvature", HERE / "validation_curvature.py"),
    ("physical_meas",        HERE / "Physical_meas.py"),
    ("run_kiss_icp",         HERE / "KISS ICP" / "run_kiss_icp.py"),
    # ("validation_main",    HERE / "validation_main.py"),   # optional final aggregate+plot
]


def load_module(name: str, path: Path):
    """Import a module from an explicit file path.

    The module's top-level code (its one-time slow setup) executes HERE, once.
    Importing the same path again in the same process reuses the loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # ← triggers the module-level setup block
    return module


def run_pipeline(input_file: str, timestamp: str, modules=MODULES) -> dict:
    """Import each module once, then call its run(input_file, timestamp).

    One module failing (or being missing) never stops the others — the error is
    printed and the pipeline moves on.
    """
    results: dict = {}
    for name, path in modules:
        path = Path(path)
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")

        if not path.exists():
            print(f"  [skip] not implemented yet: {path}")
            results[name] = None
            continue

        try:
            mod = load_module(name, path)
            if not hasattr(mod, "run"):
                print(f"  [skip] {name} has no run(input_file, timestamp)")
                results[name] = None
                continue
            results[name] = mod.run(input_file, timestamp)
            print(f"  [ok]  {name} -> {results[name]}")
        except Exception as exc:                # keep going even if one blows up
            results[name] = None
            print(f"  [FAIL] {name}: {exc}")
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    for name, out in results.items():
        print(f"  {'ok  ' if out is not None else 'FAIL/skip'}  {name}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run every validation module on a shared input file + timestamp."
    )
    parser.add_argument("--input-file", default=INPUT_FILE,
                        help="Shared input passed to every run() (default: %(default)s)")
    parser.add_argument("--timestamp", default=TIMESTAMP,
                        help="Shared reference timestamp passed to every run() (default: %(default)s)")
    parser.add_argument("--only", nargs="*", metavar="NAME",
                        help="Run only these module names (default: all in the registry).")
    args = parser.parse_args()

    selected = MODULES if not args.only else [(n, p) for n, p in MODULES if n in args.only]
    run_pipeline(args.input_file, args.timestamp, selected)
