import glob
import os
import pandas as pd

folder = r"D:\Validation_results\Statistics"
output_file = os.path.join(folder, "merged_output.csv")

csv_files = [
    f for f in glob.glob(os.path.join(folder, "*.csv"))
    if os.path.basename(f) != "merged_output.csv"
]

if not csv_files:
    print("No CSV files found.")
else:
    frames = []
    for path in csv_files:
        df = pd.read_csv(path, skiprows=1)  # skip the title row
        df.insert(0, "source_file", os.path.basename(path))
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_file, index=False)
    print(f"Merged {len(csv_files)} file(s) -> {output_file}")
    print(merged.to_string())
