from osgeo import gdal
import rasterio
import numpy as np

# ------------------------------
# INPUTS
# ------------------------------
input_raster = r'X:\Active Projects\HOT SPOT SYSTEM\Pop layer GHSL\GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0\GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0.tif'
binary_raster = r'X:\Active Projects\HOT SPOT SYSTEM\Pop layer GHSL\GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0\binary_temp.tif'
distance_raster  = r'X:\Active Projects\HOT SPOT SYSTEM\Pop layer GHSL\GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0\dist_temp.tif'
buffer_distance_m = 6000  # 3 km 3000 for crops, 6000 renge (https://link.springer.com/article/10.1186/s13570-019-0150-z?utm_source=chatgpt.com;
min_pop_in_100m_4_boolean = 0
output_raster = r'X:\Active Projects\HOT SPOT SYSTEM\Pop layer GHSL\GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0\GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_buffered_'+ str(buffer_distance_m) + '.tif'
delete_temp = False         # delete intermediate rasters
# ------------------------------


# ============================================================
# STEP 1 — Convert input raster → boolean raster (value > 0 → 1)
# Block-based, BigTIFF
# ============================================================
print("STEP 1: Creating Boolean raster...")

with rasterio.open(input_raster) as src:

    profile = src.profile
    profile.update(
        dtype=rasterio.uint8,
        nodata=None,        # Boolean raster: no nodata
        tiled=True,
        compress="lzw",
        bigtiff="YES"
    )

    with rasterio.open(binary_raster, "w", **profile) as dst:
        for ji, window in src.block_windows(1):
            block = src.read(1, window=window)
            binary_block = (block > min_pop_in_100m_4_boolean).astype(np.uint8)
            dst.write(binary_block, 1, window=window)

print(f"Boolean raster created: {binary_raster}")


# ============================================================
# STEP 2 — Compute distance raster using GDAL ComputeProximity
# Memory-safe, BigTIFF
# ============================================================
print("STEP 2: Computing distance raster (GDAL Proximity)...")

src = gdal.Open(binary_raster)
gt = src.GetGeoTransform()
proj = src.GetProjection()
xsize = src.RasterXSize
ysize = src.RasterYSize

driver = gdal.GetDriverByName("GTiff")
dist_ds = driver.Create(
    distance_raster,
    xsize, ysize, 1,
    gdal.GDT_Float32,
    options=[
        "TILED=YES",
        "COMPRESS=LZW",
        "BIGTIFF=YES"
    ]
)

dist_ds.SetGeoTransform(gt)
dist_ds.SetProjection(proj)

# Proximity: compute distance in map units (meters)
gdal.ComputeProximity(
    src.GetRasterBand(1),
    dist_ds.GetRasterBand(1),
    ["VALUES=1", "DISTUNITS=GEO"]
)

dist_ds = None
print(f"Distance raster created: {distance_raster}")


# ============================================================
# STEP 3 — Threshold distance raster → 10 km buffer
# Block-based, BigTIFF
# ============================================================
print("STEP 3: Creating 10 km buffered raster...")

with rasterio.open(distance_raster) as src:
    profile = src.profile
    profile.update(
        dtype=rasterio.uint8,
        nodata=0,          # nodata = 0 for boolean output
        compress="lzw",
        tiled=True,
        bigtiff="YES"
    )

    with rasterio.open(output_raster, "w", **profile) as dst:
        for ji, window in src.block_windows(1):
            dist_block = src.read(1, window=window)
            buffered_block = (dist_block <= buffer_distance_m).astype(np.uint8)
            dst.write(buffered_block, 1, window=window)

print(f"Buffered raster created: {output_raster}")


# ============================================================
# STEP 4 — Cleanup temporary distance raster (optional)
# ============================================================
if delete_temp:
    try:
        os.remove(binary_raster)
        os.remove(distance_raster)
        print("Temporary files removed.")
    except:
        print("Could not delete temporary files (safe to ignore).")

print("PROCESS COMPLETED SUCCESSFULLY.")