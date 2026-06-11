"""
bin_to_ply.py — convert KITTI-style .bin point clouds to .ply for visualisation.

Give it one or more .bin file paths (or folders) and it writes a .ply next to
each input file (same name, .ply extension).

The .bin files are flat float32 arrays, one of:
    N x 5  -> x, y, z, intensity, ring   (nuScenes / Patchwork++ notebook output)
    N x 4  -> x, y, z, intensity         (KITTI / Velodyne style)
    N x 3  -> x, y, z
Column width is detected in order 5 → 4 → 3 so nuScenes files are not
misread as KITTI when N is divisible by 4.

Usage
-----
  # edit INPUT_PATHS below, then just run it:
  python "Useful scripts/bin_to_ply.py"

  # or pass paths on the command line (files and/or folders):
  python "Useful scripts/bin_to_ply.py" path/to/cloud.bin path/to/folder
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# ── EDIT ME: files or folders to convert (used when no CLI args are given) ─────
INPUT_PATHS = [
    r"D:\Filtered bins\1779439916349_csf.bin",

]
WRITE_ASCII = True    # ASCII .ply: slower/bigger but most online viewers need it
VISUALIZE   = True    # open an Open3D window for each converted cloud
PLOT_SZ     = True    # show a BEV + s-z profile plot for debugging
# ──────────────────────────────────────────────────────────────────────────────


def load_bin(path: Path) -> np.ndarray:
    """Load a .bin point cloud as an (N, 3/4/5) float32 array.

    Checks column width in order 5 → 4 → 3 so nuScenes-format files
    (x, y, z, intensity, ring) are not misread as KITTI-format when N
    happens to be divisible by 4, which shifts the x/y/z columns out of
    phase and produces a garbled/duplicated-looking cloud.
    """
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 5 == 0:
        return raw.reshape(-1, 5)          # nuScenes: x, y, z, intensity, ring
    if raw.size % 4 == 0:
        return raw.reshape(-1, 4)          # KITTI: x, y, z, intensity
    if raw.size % 3 == 0:
        return raw.reshape(-1, 3)          # x, y, z
    raise ValueError(
        f"{path.name}: size ({raw.size} floats) is not divisible by 3, 4, or 5."
    )


def plot_sz(xyz: np.ndarray, title: str = "") -> None:
    """BEV coloured by Z (top) + s-Z elevation profile (bottom).

    's' is the cumulative horizontal distance along the point sequence
    after sorting by X, giving a rough road-profile axis.
    """
    # Sort by X so the profile reads left-to-right
    order = np.argsort(xyz[:, 0])
    pts   = xyz[order]

    # s = cumulative 2-D distance (X-Y plane only)
    dxy = np.diff(pts[:, :2], axis=0)
    s   = np.concatenate([[0.0], np.cumsum(np.linalg.norm(dxy, axis=1))])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title or "Ground cloud debug", fontsize=11)

    # ── BEV coloured by Z ──────────────────────────────────────────
    sc = axes[0].scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
                         s=1, cmap="viridis")
    axes[0].set_xlabel("X forward [m]")
    axes[0].set_ylabel("Y left [m]")
    axes[0].set_title("Bird's-eye view — colour = Z height")
    axes[0].set_aspect("equal")
    axes[0].grid(True, lw=0.3)
    plt.colorbar(sc, ax=axes[0], label="Z [m]", shrink=0.8)

    # ── s-Z elevation profile ──────────────────────────────────────
    axes[1].scatter(s, pts[:, 2], s=1, c="steelblue", alpha=0.4)
    axes[1].set_xlabel("s — cumulative distance [m]")
    axes[1].set_ylabel("Z height [m]")
    axes[1].set_title("Elevation profile (s-Z)")
    axes[1].grid(True, lw=0.3)

    plt.tight_layout()
    plt.show()


def bin_to_ply(bin_path: Path) -> Path:
    """Convert a single .bin to a .ply written next to it. Returns the .ply path."""
    data = load_bin(bin_path)
    xyz = data[:, :3].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # If an intensity column is present, map it to grayscale colours so the .ply
    # carries the intensity visually (normalised per-cloud).
    if data.shape[1] >= 4:
        inten = data[:, 3].astype(np.float64)
        rng = np.ptp(inten)
        if rng > 0:
            g = (inten - inten.min()) / rng
            pcd.colors = o3d.utility.Vector3dVector(np.repeat(g[:, None], 3, axis=1))

    ply_path = bin_path.with_suffix(".ply")
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=WRITE_ASCII)
    print(f"  {bin_path.name} -> {ply_path.name}  ({len(xyz):,} points)")
    if PLOT_SZ:
        plot_sz(xyz, title=bin_path.name)
    if VISUALIZE:
        o3d.visualization.draw_geometries([pcd], window_name=ply_path.name)
    return ply_path


def iter_bin_files(paths):
    """Yield every .bin file from the given file/folder paths."""
    for p in paths:
        p = Path(p)
        if p.is_dir():
            yield from sorted(p.rglob("*.bin"))
        elif p.suffix.lower() == ".bin":
            yield p
        else:
            print(f"  skipping (not a .bin or folder): {p}")


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else INPUT_PATHS
    bin_files = list(iter_bin_files(paths))
    if not bin_files:
        print("No .bin files found. Edit INPUT_PATHS or pass paths on the command line.")
        return
    print(f"Converting {len(bin_files)} file(s):")
    for f in bin_files:
        try:
            bin_to_ply(f)
        except Exception as e:
            print(f"  ERROR on {f}: {e}")


if __name__ == "__main__":
    main()
