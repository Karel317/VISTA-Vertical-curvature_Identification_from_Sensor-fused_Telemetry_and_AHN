# ============================================================
# INPUTS — configure these before running
# ============================================================
INPUT_FILE    = r"C:\Users\Leons\OneDrive - Delft University of Technology\BEP\rosbag_0.mcap"
OPTIONAL_NAME = "kuitbrug"   # set to "" to omit from filename
START_TIME    = 1779439322.712    # Unix timestamp in seconds (float)
END_TIME      = 1779439356.62369227    # Unix timestamp in seconds (float)
# ============================================================

from pathlib import Path
from kappe.cut import CutSplits, CutSettings, cutter

# Parse date and time from the fixed path structure:
# ...\<date>\Rosbag\<time>\rosbag\rosbag_0.mcap
parts    = Path(INPUT_FILE).parts
date_str = parts[-5]   # e.g. "2026_04_29"
time_str = parts[-3]   # e.g. "15_10_00"

# Build output folder
output_folder = Path(r"C:\Users\Leons\OneDrive - Delft University of Technology\BEP") / date_str
output_folder.mkdir(parents=True, exist_ok=True)

# Build output filename
name_parts      = [p for p in [OPTIONAL_NAME, date_str, time_str] if p]
output_filename = "_".join(name_parts) + ".mcap"

# Trim: reads from INPUT_FILE (original untouched), writes trimmed copy to output_folder
splits   = [CutSplits(start=START_TIME, end=END_TIME, name=output_filename)]
settings = CutSettings(splits=splits, keep_tf_tree=True)
cutter(Path(INPUT_FILE), output_folder, settings)

print(f"Saved trimmed file to: {output_folder / output_filename}")
