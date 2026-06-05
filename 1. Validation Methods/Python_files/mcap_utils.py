"""Shared MCAP helpers for the validation modules.

Two things every GPS-reading validation method needs, in one place so they are
done once and shared:

* ``cut_mcap`` — slice an MCAP to a small time window (optionally a topic subset)
  and cache the cut on disk. The slow full-file scan of a big bag happens only on
  the first call; later runs (even in a new process) reuse the cut.

* ``load_gps`` — cut + decode the GPS topic into numpy arrays, with an extra
  in-process cache. When several modules in one run (e.g. AHN5_validation and
  Validation_curvature_IMU_GPS) ask for the same bag/topic/timestamp, the bag is
  decoded only once and the arrays are shared.
"""
import os
import hashlib
import tempfile

import numpy as np
from mcap.reader import make_reader
from mcap.writer import Writer as McapWriter
from mcap_ros2.decoder import DecoderFactory


def cut_mcap(src_path, timestamp, window_s, topics=None):
    """Slice an MCAP to [timestamp - window_s, timestamp + window_s] and cache it.

    The first call copies the windowed messages to a temp file in the system temp
    dir; later calls with the same source/timestamp/window/topics reuse that
    cached cut, so the slow full-file scan happens only once. Passing `topics`
    keeps only those topics (e.g. drop bulky LiDAR point clouds the GPS/IMU reads
    never use), shrinking the cut dramatically. Messages are copied raw (no
    decode) with their schemas/channels re-registered, so log/publish times are
    preserved.

    Returns the path to the cut MCAP.
    """
    src_path = str(src_path)
    start_ns = int((timestamp - window_s) * 1_000_000_000)
    end_ns   = int((timestamp + window_s) * 1_000_000_000)

    cache_dir = os.path.join(tempfile.gettempdir(), "mcap_cut_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Cache key includes source size + mtime (so a changed source invalidates it)
    # and a short tag for the requested topics (so different topic sets don't clash).
    # Use a stable hash (hashlib) — Python's built-in hash() of strings is salted
    # per process, which would change the filename every run and defeat the cache.
    st = os.stat(src_path)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    if not topics:
        topic_tag = "all"
    else:
        topic_tag = hashlib.md5("|".join(sorted(topics)).encode()).hexdigest()[:8]
    cut_path = os.path.join(
        cache_dir,
        f"{stem}_{int(timestamp)}_{int(window_s)}s_{st.st_size}_{int(st.st_mtime)}_{topic_tag}.mcap",
    )

    if os.path.exists(cut_path):
        print(f"Using cached cut MCAP: {cut_path}")
        return cut_path

    print(f"Cutting MCAP to +/-{window_s:.0f}s around TIME (first run; cached afterwards)...")
    tmp_path = cut_path + ".tmp"
    with open(src_path, "rb") as fin, open(tmp_path, "wb") as fout:
        reader = make_reader(fin)
        writer = McapWriter(fout)
        writer.start(profile="ros2", library="mcap_cut")

        schema_map  = {}   # src schema id  -> writer schema id
        channel_map = {}   # src channel id -> writer channel id

        for schema, channel, message in reader.iter_messages(
            topics=topics, start_time=start_ns, end_time=end_ns
        ):
            if channel.id not in channel_map:
                schema_id = 0
                if schema is not None:
                    if schema.id not in schema_map:
                        schema_map[schema.id] = writer.register_schema(
                            name=schema.name, encoding=schema.encoding, data=schema.data)
                    schema_id = schema_map[schema.id]
                channel_map[channel.id] = writer.register_channel(
                    topic=channel.topic,
                    message_encoding=channel.message_encoding,
                    schema_id=schema_id,
                    metadata=channel.metadata,
                )
            writer.add_message(
                channel_id=channel_map[channel.id],
                log_time=message.log_time,
                data=message.data,
                publish_time=message.publish_time,
                sequence=message.sequence,
            )

        writer.finish()

    os.replace(tmp_path, cut_path)   # atomic: a partial cut never looks cached
    print(f"Cut MCAP saved: {cut_path}")
    return cut_path


# In-process cache of decoded GPS arrays, keyed by (abs path, topic, timestamp,
# window). Shared across all modules that `import mcap_utils` in one run.
_GPS_CACHE = {}


def load_gps(gps_file, gps_topic, timestamp, window_s):
    """Cut (disk-cached) + decode the GPS topic to numpy arrays, decoded once.

    Returns a dict with arrays ``t`` (Unix s), ``lat``, ``lon``, ``alt``. The
    decode is cached in-process, so a second caller with the same
    bag/topic/timestamp/window gets the already-loaded arrays instead of reading
    the bag again.
    """
    key = (os.path.abspath(str(gps_file)), gps_topic, round(float(timestamp), 6), float(window_s))
    if key in _GPS_CACHE:
        print(f"Reusing already-loaded GPS data ({gps_topic})")
        return _GPS_CACHE[key]

    cut = cut_mcap(gps_file, timestamp, window_s, topics=[gps_topic])

    gps = {"t": [], "lat": [], "lon": [], "alt": []}
    with open(cut, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=[gps_topic]):
            gps["t"].append(message.publish_time / 1e9)
            gps["lat"].append(ros_msg.latitude)
            gps["lon"].append(ros_msg.longitude)
            gps["alt"].append(getattr(ros_msg, "altitude", float("nan")))

    if len(gps["t"]) == 0:
        raise RuntimeError(
            f"No messages on GPS topic '{gps_topic}' within +/-{window_s}s of TIME.")

    for k in gps:
        gps[k] = np.array(gps[k])

    print(f"GPS samples : {len(gps['t'])}")
    _GPS_CACHE[key] = gps
    return gps
