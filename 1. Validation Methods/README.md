# File context and how to use them
Every file's context and usage will be explained briefly.

---

## Physical_meas.py

This script reads height-increment measurements, collected physically along a route, from a CSV file. It computes a cumulative height profile for each measurement column, plots the elevation curves over distance, and saves the result as a compressed `.npz` file in the standardised format used by `Validation_main.py` for cross-comparison with the calculated profiles.

**Input** \
A `.csv` file named `Physical_meas_data.csv` placed in the same folder as the script. Each column contains height increments in centimetres and the first row contains the column headers used as plot labels.

**Adjustable parameters** \
`SAVE_DIR` — folder where the `.npz` output is written \
`TIME` — Unix timestamp to associate with this run (must match the timestamps used in `Validation_main.py` for the results to be paired correctly) \
`method` — label string used as the filename prefix and stored in the `.npz` as the method identifier \
Distance per increment — the constant `0.222` on line 20 sets the metres per measurement step; update this to match the actual interval used during the physical measurement

**Output** \
A compressed `.npz` file saved as `<method>_<TIME>.npz` in `SAVE_DIR`, containing: \
`z` — cumulative height profile (m) \
`s` — distance along the profile (m) \
`method` — label string
