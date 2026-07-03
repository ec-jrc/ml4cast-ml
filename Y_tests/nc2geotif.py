import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os

nc_file = r"V:\asap\asap_analysis\AgERA5\Precipitation flux\2025\Precipitation-Flux_C3S-glob-agric_AgERA5_20250625_final-v2.0.0.nc"
output_tif = r"V:\asap\asap_analysis\ERA5 from C3S\agERA5\Daily_Precipitation-Flux_C3S-glob-agric_AgERA5_20250625_final-v2.0.0.tif"
ds = xr.open_dataset(nc_file)
print(ds)
layer = ds["Precipitation_Flux"]

# Fix 1: Convert 0-360 longitudes to -180/180
layer = layer.assign_coords(
    longitude=(((layer.lon + 180) % 360) - 180)
)
# Re-sort so longitudes go from -180 → 180
layer = layer.sortby("lon")


data = layer.values

# Fix 2: Flip vertically if latitudes are ascending (ERA5 is often N→S)
lat = layer["lat"].values
lon = layer["lon"].values

if lat[0] < lat[-1]:
    # Latitudes are ascending (S→N), flip so image is N→S (top→bottom)
    data = np.flipud(data)
    lat = lat[::-1]

data = data[0, :, :]
print(data.shape)
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