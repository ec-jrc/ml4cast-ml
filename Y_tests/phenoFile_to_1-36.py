import os
import numpy as np
import rasterio

# full path to ENVI image (the .img file)
fns = [r"V:\asap\asap_analysis\2026 Pheno4Droughts\CF_FPR_LTA_ENVIs1.img", r"V:\asap\asap_analysis\2026 Pheno4Droughts\CF_FPR_LTA_ENVIs2.img", r"V:\asap\asap_analysis\2026 Pheno4Droughts\CF_FPR_LTA_ENVIe1.img", r"V:\asap\asap_analysis\2026 Pheno4Droughts\CF_FPR_LTA_ENVIe2.img"]
# fn = r"V:\asap\asap_analysis\2026 Pheno4Droughts\CF_FPR_LTA_ENVIs1.img"
for fn in fns:

    # output file: same folder, ENVI format
    base, _ = os.path.splitext(fn)
    fn2 = base + "_1-36.tif"

    with rasterio.open(fn) as src:
        V = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    # Prepare output (uint8, nodata=255)
    T = np.full(V.shape, 255, dtype=np.uint8)

    # Valid data mask
    if nodata is not None:
        valid = V != nodata
    else:
        valid = np.ones(V.shape, dtype=bool)

    # Apply transformation rules
    T[(V <= 36) & valid] = V[(V <= 36) & valid]
    T[(V > 36) & (V <= 72) & valid] = V[(V > 36) & (V <= 72) & valid] - 36
    T[(V > 72) & (V <= 108) & valid] = V[(V > 72) & (V <= 108) & valid] - 72
    # >108 remains 255

    # Update profile for GeoTIFF
    profile.update(
        driver="GTiff",
        dtype="uint8",
        nodata=255,
        count=1,
        compress="lzw"   # optional but recommended
    )

    # Write GeoTIFF
    with rasterio.open(fn2, "w", **profile) as dst:
        dst.write(T, 1)

    print("Saved:", fn2)