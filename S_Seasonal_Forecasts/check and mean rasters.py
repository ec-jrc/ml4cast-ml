from pathlib import Path
import rasterio
import numpy as np

# ==========================================================
# User parameters
# ==========================================================
root = Path(r"V:\foodsec\Projects\ECMWF-SEAS5\Archive\MSMM_stream-Emeans")

month = 1
horizon = 4

# ==========================================================
# Load yearly mean rasters
# ==========================================================
arrays = []
years = []

profile = None

for year in range(2000, 2026):

    infile = root / f"{year}{month:02d}01" / f"SF_r_mean_H{horizon}.tif"

    if not infile.exists():
        print(f"Missing {infile}")
        continue

    with rasterio.open(infile) as src:

        arr = src.read(1).astype(np.float32)

        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        arrays.append(arr)
        years.append(year)

        if profile is None:
            profile = src.profile

arrays = np.stack(arrays)

# ==========================================================
# Compute multiannual mean
# ==========================================================
climatology = np.nanmean(arrays, axis=0)

# ==========================================================
# Save climatology (optional)
# ==========================================================
profile.update(dtype="float32", count=1)

outfile = root / f"Climatology_M{month:02d}_H{horizon}.tif"

with rasterio.open(outfile, "w", **profile) as dst:
    dst.write(climatology.astype(np.float32), 1)

print(f"Saved {outfile}")

# ==========================================================
# Compute anomalies
# ==========================================================
for i, year in enumerate(years):

    anomaly = arrays[i] - climatology

    outfile = (
        root
        / f"{year}{month:02d}01"
        / f"SF_r_anomaly_H{horizon}.tif"
    )

    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(anomaly.astype(np.float32), 1)

    print(f"Saved {outfile}")