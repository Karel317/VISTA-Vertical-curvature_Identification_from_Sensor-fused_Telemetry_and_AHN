"""
mcap_utils.py — shared MCAP helpers for AHN5.py and EKF.py.
Place this file in the same folder as AHN5.py and EKF.py.
"""
import os
import tempfile
from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


# ── cut_mcap ─────────────────────────────────────────────────────────────────

def cut_mcap(input_file, timestamp, window_s, topics=None):
    """
    Cut a slice of +/- window_s seconds around timestamp from input_file.
    Writes the result to a temp file (cached by input path + timestamp + window).
    Returns the path to the cut MCAP.

    Parameters
    ----------
    input_file : str | Path
    timestamp  : float   Unix seconds (the reference TIME)
    window_s   : float   half-width of the cut window in seconds
    topics     : list[str] | None   if given, only these topics are kept
    """
    from kappe.cut import CutSplits, CutSettings, cutter

    input_file = Path(input_file)
    start = timestamp - window_s
    end   = timestamp + window_s

    # Cache in system temp so the slow cut only runs once per input+window
    cache_dir = Path(tempfile.gettempdir()) / "mcap_cut_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    st = input_file.stat()
    topic_tag = ("_" + "_".join(t.strip("/").replace("/", "-") for t in topics)
                 if topics else "")
    cache_name = (
        f"{input_file.stem}_{int(timestamp)}_{int(window_s)}s"
        f"_{st.st_size}_{int(st.st_mtime)}{topic_tag}.mcap"
    )
    out_path = cache_dir / cache_name

    if out_path.exists():
        print(f"Using cached cut MCAP: {out_path}")
        return str(out_path)

    print(f"Cutting MCAP +/-{window_s:.0f}s around TIME (first run, then cached)...")
    tmp_path = out_path.with_suffix(".tmp.mcap")

    splits   = [CutSplits(start=start, end=end, name=tmp_path.name)]
    settings = CutSettings(splits=splits, keep_tf_tree=True)
    cutter(input_file, tmp_path.parent, settings)

    tmp_path.rename(out_path)
    print(f"Cut MCAP cached: {out_path}")
    return str(out_path)


# ── shared_cut_mcap ──────────────────────────────────────────────────────────
# AHN5 (GPS), EKF (GPS) and KISS-ICP (LiDAR) all slice the SAME main rosbag.
# Instead of each module cutting the full bag separately, they request ONE shared
# cut that keeps every topic any of them needs, over the widest window. Because
# the cut_mcap cache key (input + TIME + window + topics) is then identical for
# all three, whichever module runs first creates the slice and the others reuse
# it — the bag is read once, not three times.
GPS_TOPIC           = "/navsat_topic"           # GPS topic in the main rosbag
LIDAR_TOPIC         = "/rslidar/M1P_deskewed"   # LiDAR topic in the main rosbag
SHARED_CUT_TOPICS   = [GPS_TOPIC, LIDAR_TOPIC]  # order matters: defines the cache key
SHARED_CUT_WINDOW_S = 30.0                      # widest +/- window any stage needs


def shared_cut_mcap(input_file, timestamp, extra_topics=None, window_s=SHARED_CUT_WINDOW_S):
    """Cut the main rosbag once, keeping all topics the pipeline needs, so AHN5/
    EKF (GPS) and KISS-ICP (LiDAR) reuse a single slice. Returns the cut path.

    `extra_topics` adds any topic not already in SHARED_CUT_TOPICS (e.g. a GPS
    topic that differs from the default). Topic order is kept stable so the cache
    key matches across modules.
    """
    topics = list(SHARED_CUT_TOPICS)
    for t in (extra_topics or []):
        if t not in topics:
            topics.append(t)
    return cut_mcap(input_file, timestamp, window_s, topics=topics)


# ── load_gps ─────────────────────────────────────────────────────────────────

def load_gps(input_file, gps_topic, timestamp, window_s):
    """
    Load GPS messages from a +/- window_s slice around timestamp.
    Returns dict with keys "t", "lat", "lon" (all numpy arrays).

    Parameters
    ----------
    input_file : str | Path
    gps_topic  : str    ROS2 topic name, e.g. "/navsat_topic"
    timestamp  : float  Unix seconds (the reference TIME)
    window_s   : float  half-width of the read window in seconds
    """
    import numpy as np

    # Cut via the SHARED slice (GPS + LiDAR topics) so AHN5/EKF and KISS-ICP reuse
    # one cut of the bag. Reading below still filters to +/- window_s. The cut
    # window is at least SHARED_CUT_WINDOW_S so KISS-ICP can reuse the same file.
    cut_file = shared_cut_mcap(input_file, timestamp, extra_topics=[gps_topic],
                               window_s=max(window_s, SHARED_CUT_WINDOW_S))

    t_min = timestamp - window_s
    t_max = timestamp + window_s

    ts, lats, lons = [], [], []

    with open(cut_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=[gps_topic]
        ):
            t = message.log_time / 1e9  # nanoseconds → seconds
            if t < t_min or t > t_max:
                continue
            ts.append(t)
            lats.append(decoded.latitude)
            lons.append(decoded.longitude)

    if not ts:
        raise RuntimeError(
            f"No GPS messages found on topic '{gps_topic}' in "
            f"{input_file} within +/-{window_s}s of {timestamp}."
        )

    return {
        "t":   np.array(ts),
        "lat": np.array(lats),
        "lon": np.array(lons),
    }