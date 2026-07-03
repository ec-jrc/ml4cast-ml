import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os

nc_file = r"V:\asap\asap_analysis\ERA5 from C3S\b96dade6843f6dd9d9a0fef09ecb5256.nc"
output_tif = r"V:\asap\asap_analysis\ERA5 from C3S\2025_era5_dek18_fix_mm2.tif"
outdir = r"V:\asap\asap_analysis\ERA5 from C3S"

ds = xr.open_dataset(nc_file)
layer = ds["tp"]

# Fix 1: Convert 0-360 longitudes to -180/180
layer = layer.assign_coords(
    longitude=(((layer.longitude + 180) % 360) - 180)
)
# Re-sort so longitudes go from -180 → 180
layer = layer.sortby("longitude")

# Sum along time dimension
arr_sum = layer.sum(dim="valid_time")
data = arr_sum.values * 1000 #(m to mm)

# Fix 2: Flip vertically if latitudes are ascending (ERA5 is often N→S)
lat = layer["latitude"].values
lon = layer["longitude"].values

if lat[0] < lat[-1]:
    # Latitudes are ascending (S→N), flip so image is N→S (top→bottom)
    data = np.flipud(data)
    lat = lat[::-1]

# Fix 3: Use correct bounds order (west, south, east, north)
transform = from_bounds(
    lon.min(), lat.min(),   # west, south
    lon.max(), lat.max(),   # east, north
    data.shape[1],          # width  = number of columns = len(lon)
    data.shape[0]           # height = number of rows    = len(lat)
)

with rasterio.open(
    output_tif,
    "w",
    driver="GTiff",
    height=data.shape[0],
    width=data.shape[1],
    count=1,
    dtype=data.dtype,
    crs="EPSG:4326",
    transform=transform,
) as dst:
    dst.write(data, 1)

print(f"Saved: {output_tif}")
print(f"Lon range: {lon.min():.2f} → {lon.max():.2f}")
print(f"Lat range: {lat.min():.2f} → {lat.max():.2f}")
print(f"Data shape: {data.shape}")


# -----------------------------
# Save one TIFF per band
# -----------------------------
for i in range(layer.sizes["valid_time"]):

    band = layer.isel({"valid_time": i})

    data = band.values * 1000

    # Flip if latitude descending
    if lat[0] < lat[-1]:
        data = np.flipud(data)
    # outfile = o

    varname = "tp"
    z_name = "valid_time"
    outfile = os.path.join(outdir, f"{varname}_{z_name}_{i:03d}.tif")

    with rasterio.open(
        outfile,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    print(f"Saved: {outfile}")

