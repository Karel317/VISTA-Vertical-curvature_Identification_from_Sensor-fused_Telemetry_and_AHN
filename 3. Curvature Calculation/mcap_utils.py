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

    # First cut to a small window so reading is fast
    cut_file = cut_mcap(input_file, timestamp, window_s, topics=[gps_topic])

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