"""
Compare AgERA5 dekad TIF files vs ERA5 ENVI IMG files.

Edit the TWO lines in the USER CONFIG section (YEAR and VARIABLE) and run.
All paths, patterns and plot settings are selected automatically.

Supported variables:  tav  |  rain  |  evpt  |  rad

Dekad mapping (YYYYMMDD -> dekad index 0-based):
    Jan d1=0101, d2=0111, d3=0121
    Feb d1=0201, d2=0211, d3=0221
    ... same pattern for all months (36 dekads per year)
"""

import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import rasterio.warp


# ==============================================================================
# USER CONFIG  -  only these two lines need editing
# ==============================================================================

YEAR     = 2025
VARIABLE = "chirps"   # options: "tav", "rain", "evpt", "rad"

# ==============================================================================
# END OF USER CONFIG  -  no need to edit below this line
# ==============================================================================

# ------------------------------------------------------------------------------
# Per-variable settings  (paths, patterns, units, plot range)
# TIF dirs use a {year} placeholder so they stay correct when YEAR changes.
# ------------------------------------------------------------------------------
_SETTINGS = {
     "chirps": dict(
        tif_dir      = r"\\s-jrciprap266p.ies.jrc.it\data\asap.9.0\data\indicators_rain\rain",
        tif_pattern  = "rain_{year}????.tif",
        envi_dir     = r"Y:\indicators_rain\rain",
        envi_pattern = "rain_{year}????.img",
        envi_regex   = r"rain_{year}(\d{{4}})\.img",
        unit_label   = "mm",
        plot_vmin    =   0,
        plot_vmax    = 1600,
        tif_k2c      = False,
    )
}

if VARIABLE not in _SETTINGS:
    raise ValueError(f"Unknown VARIABLE '{VARIABLE}'. Choose from: {list(_SETTINGS)}")

_s = _SETTINGS[VARIABLE]

TIF_DIR      = _s["tif_dir"].format(year=YEAR)
TIF_PATTERN  = _s["tif_pattern"].format(year=YEAR)
TIF_REGEX    = r"rain_{year}(\d{{4}})\.tif".format(year=YEAR)
ENVI_DIR     = _s["envi_dir"]
ENVI_PATTERN = _s["envi_pattern"].format(year=YEAR)
ENVI_REGEX   = _s["envi_regex"].format(year=YEAR)

UNIT_LABEL   = _s["unit_label"]
LABEL_AGERA5 = "v3" # "AgERA5 TIF"
LABEL_ERA5   = "v2" #"ERA5 ENVI"
PLOT_VMIN    = _s["plot_vmin"]
PLOT_VMAX    = _s["plot_vmax"]
TIF_CONVERT_KELVIN_TO_CELSIUS = _s["tif_k2c"]

# -- No-data sentinel values ---------------------------------------------------
NODATA_ENVI = -9999.0
NODATA_TIF  = -9999.0

# -- Output root directory (subfolder named VARIABLE is created inside) --------
BASE_OUT_DIR = r"V:\asap\asap_analysis\chirps v3 vs v2\scatter_plots_90_percentile"

# -- Common resampling grid (0.25 deg global, matches ERA5 native resolution) --
COMMON_RES    = 0.05
COMMON_LEFT   = -180.0
COMMON_RIGHT  =  180.0
COMMON_TOP    =   50
COMMON_BOTTOM =  -50.0

# -- Percentile used to cap the hexbin colorbar --------------------------------
HEXBIN_COLORBAR_PERCENTILE = 90


# -- DEKAD DATE HELPERS --------------------------------------------------------
def dekad_index_to_mmdd(idx):
    """Convert 0-based dekad index (0-35) to MMDD string (e.g. 0 -> '0101')."""
    month = idx // 3 + 1
    day   = [1, 11, 21][idx % 3]
    return f"{month:02d}{day:02d}"


def tif_dekad_number(idx):
    """Return zero-padded 1-based dekad number as it appears in TIF filenames."""
    return f"{idx + 1:02d}"


# -- ENVI HEADER READER --------------------------------------------------------
def read_envi_header(hdr_path):
    """Parse an ENVI .hdr file into a dict."""
    meta = {}
    with open(hdr_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, _, val = line.partition("=")
                meta[key.strip().lower()] = val.strip()
    return meta


def parse_envi_map_info(meta):
    """
    Extract (x_ul, y_ul, x_res, y_res) from the ENVI 'map info' field.
    Example: {Geographic Lat/Lon, 1, 1, -179.875, 75.125, 0.25, 0.25, WGS-84}
    """
    raw   = meta.get("map info", "").strip("{}")
    parts = [p.strip() for p in raw.split(",")]
    return float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])


# -- RASTER LOADERS ------------------------------------------------------------
def load_envi_img(img_path):
    """
    Read an ENVI flat-binary .img file using its .hdr sidecar.
    Returns (data as float32 with NaN for nodata, geo_info_dict).
    """
    hdr_path = img_path.replace(".img", ".hdr")
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"Header not found: {hdr_path}")

    meta    = read_envi_header(hdr_path)
    samples = int(meta["samples"])
    lines   = int(meta["lines"])
    dtype   = {1: "<i1", 2: "<i2", 3: "<i4", 4: "<f4", 5: "<f8",
                12: "<u2", 13: "<u4"}[int(meta.get("data type", 4))]

    with open(img_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=dtype).reshape(lines, samples)

    x_ul, y_ul, x_res, y_res = parse_envi_map_info(meta)
    geo = dict(x_ul=x_ul, y_ul=y_ul, x_res=x_res, y_res=y_res,
               samples=samples, lines=lines)

    data = raw.astype(np.float32)
    data[data == NODATA_ENVI] = np.nan
    return data, geo


def load_tif(tif_path):
    """
    Read a GeoTIFF and return (data, transform, crs).
    Applies Kelvin->Celsius conversion if TIF_CONVERT_KELVIN_TO_CELSIUS is True.
    No-data values are replaced with NaN.
    """
    with rasterio.open(tif_path) as src:
        data      = src.read(1).astype(np.float32)
        nodata    = src.nodata if src.nodata is not None else NODATA_TIF
        transform = src.transform
        crs       = src.crs

    data[data == nodata] = np.nan

    if TIF_CONVERT_KELVIN_TO_CELSIUS:
        valid = np.isfinite(data)
        data[valid] -= 273.15

    return data, transform, crs


# -- REPROJECTION TO COMMON GRID -----------------------------------------------
def build_common_grid():
    """Return (transform, width, height) for the common global grid."""
    width     = int((COMMON_RIGHT - COMMON_LEFT)   / COMMON_RES)
    height    = int((COMMON_TOP   - COMMON_BOTTOM) / COMMON_RES)
    transform = from_bounds(COMMON_LEFT, COMMON_BOTTOM,
                            COMMON_RIGHT, COMMON_TOP,
                            width, height)
    return transform, width, height


def _reproject_to_common(src_array, src_transform, src_crs,
                          common_transform, width, height):
    """Bilinear reprojection of any array onto the common grid."""
    dst = np.full((height, width), np.nan, dtype=np.float32)
    rasterio.warp.reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=common_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def envi_to_common_grid(data_envi, geo, common_transform, width, height):
    """Resample ENVI array onto the common grid."""
    from rasterio.transform import from_origin
    src_transform = from_origin(
        geo["x_ul"] - geo["x_res"] / 2,
        geo["y_ul"] + geo["y_res"] / 2,
        geo["x_res"],
        geo["y_res"],
    )
    return _reproject_to_common(data_envi, src_transform, "EPSG:4326",
                                 common_transform, width, height)


def tif_to_common_grid(data_tif, src_transform, src_crs,
                        common_transform, width, height):
    """Resample TIF array onto the common grid."""
    return _reproject_to_common(data_tif, src_transform, src_crs,
                                 common_transform, width, height)


# -- FILE DISCOVERY ------------------------------------------------------------
def discover_files():
    """
    Scan TIF_DIR and ENVI_DIR for matching files.
    Returns two dicts keyed by 0-based dekad index -> file path.
    """
    tif_files  = sorted(glob.glob(os.path.join(TIF_DIR,  TIF_PATTERN)))
    envi_files = sorted(glob.glob(os.path.join(ENVI_DIR, ENVI_PATTERN)))

    def tif_key(path):
        m = re.search(TIF_REGEX, os.path.basename(path))
        # if not m:
        #     return None
        # return int(m.group(1)) - 1   # 1-based -> 0-based
        # m = re.search(ENVI_REGEX, os.path.basename(path))
        if not m:
            return None
        mmdd = m.group(1)
        month = int(mmdd[:2])
        day = int(mmdd[2:])
        d_in_month = {1: 0, 11: 1, 21: 2}.get(day)
        if d_in_month is None:
            return None
        return (month - 1) * 3 + d_in_month

    def envi_key(path):
        m = re.search(ENVI_REGEX, os.path.basename(path))
        if not m:
            return None
        mmdd       = m.group(1)
        month      = int(mmdd[:2])
        day        = int(mmdd[2:])
        d_in_month = {1: 0, 11: 1, 21: 2}.get(day)
        if d_in_month is None:
            return None
        return (month - 1) * 3 + d_in_month

    tif_dict  = {tif_key(p): p  for p in tif_files  if tif_key(p)  is not None}
    envi_dict = {envi_key(p): p for p in envi_files if envi_key(p) is not None}

    return tif_dict, envi_dict


# -- SCATTERPLOT ---------------------------------------------------------------
def make_scatterplot(x, y, label_x, label_y, title, out_path,
                     alpha=0.15, s=1.5):
    """
    Draw a scatterplot of x vs y (1-D, valid pixels only).
    Adds 1:1 line, linear regression, N / R2 / RMSE / Bias stats.
    The hexbin colorbar is capped at HEXBIN_COLORBAR_PERCENTILE of bin counts
    so the colour range is not dominated by the densest outlier bins.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        print(f"  Skipping {title}: only {len(x)} valid pairs.")
        return

    slope, intercept, r, _, _ = stats.linregress(x, y)
    r2   = r ** 2
    rmse = np.sqrt(np.mean((y - x) ** 2))
    bias = np.mean(y - x)

    fig, ax = plt.subplots(figsize=(7, 6))

    if len(x) > 50_000:
        hb = ax.hexbin(x, y, gridsize=120, cmap="YlOrRd", mincnt=1,
                       linewidths=0.2)
        hb.set_clim(vmax=np.percentile(hb.get_array(), HEXBIN_COLORBAR_PERCENTILE))
        plt.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(x, y, s=s, alpha=alpha, color="steelblue", rasterized=True)

    ax.plot([PLOT_VMIN, PLOT_VMAX], [PLOT_VMIN, PLOT_VMAX],
            "k--", lw=1.2, label="1:1 line")

    xfit = np.linspace(PLOT_VMIN, PLOT_VMAX, 200)
    ax.plot(xfit, slope * xfit + intercept, "r-", lw=1.5,
            label=f"Fit: y = {slope:.3f}x + {intercept:.2f}")

    stats_text = (f"N = {len(x):,}\n"
                  f"R2 = {r2:.4f}\n"
                  f"RMSE = {rmse:.2f} {UNIT_LABEL}\n"
                  f"Bias = {bias:+.2f} {UNIT_LABEL}")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    ax.set_xlabel(f"{label_x} ({UNIT_LABEL})", fontsize=11)
    ax.set_ylabel(f"{label_y} ({UNIT_LABEL})", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(PLOT_VMIN, PLOT_VMAX)
    ax.set_ylim(PLOT_VMIN, PLOT_VMAX)
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_combined_scatterplot(all_x, all_y, out_dir):
    """
    Combined hexbin scatterplot across all processed dekads.
    Colorbar is capped at HEXBIN_COLORBAR_PERCENTILE of bin counts.
    """
    all_x_cat = np.concatenate(all_x)
    all_y_cat = np.concatenate(all_y)

    mask = np.isfinite(all_x_cat) & np.isfinite(all_y_cat)
    all_x_cat, all_y_cat = all_x_cat[mask], all_y_cat[mask]

    if len(all_x_cat) < 10:
        print("  Skipping combined plot: insufficient valid pairs.")
        return

    slope, intercept, r, _, _ = stats.linregress(all_x_cat, all_y_cat)
    r2   = r ** 2
    rmse = np.sqrt(np.mean((all_y_cat - all_x_cat) ** 2))
    bias = np.mean(all_y_cat - all_x_cat)

    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(all_x_cat, all_y_cat, gridsize=150,
                   cmap="YlOrRd", mincnt=1, linewidths=0)
    hb.set_clim(vmax=np.percentile(hb.get_array(), HEXBIN_COLORBAR_PERCENTILE))
    plt.colorbar(hb, ax=ax, label="Pixel count")

    ax.plot([PLOT_VMIN, PLOT_VMAX], [PLOT_VMIN, PLOT_VMAX],
            "k--", lw=2, label="1:1 line", zorder=5)

    xfit = np.linspace(PLOT_VMIN, PLOT_VMAX, 300)
    ax.plot(xfit, slope * xfit + intercept, "r-", lw=2,
            label=f"Fit: y = {slope:.3f}x + {intercept:.2f}", zorder=6)

    stats_text = (f"N total = {len(all_x_cat):,}\n"
                  f"Dekads  = {len(all_x)}\n"
                  f"R2      = {r2:.4f}\n"
                  f"RMSE    = {rmse:.2f} {UNIT_LABEL}\n"
                  f"Bias    = {bias:+.2f} {UNIT_LABEL}")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    ax.set_xlabel(f"{LABEL_AGERA5} ({UNIT_LABEL})", fontsize=12)
    ax.set_ylabel(f"{LABEL_ERA5} ({UNIT_LABEL})", fontsize=12)
    ax.set_title(
        f"v3 vs v2  |  {VARIABLE.upper()}  |  All {YEAR} Dekads Combined",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_xlim(PLOT_VMIN, PLOT_VMAX)
    ax.set_ylim(PLOT_VMIN, PLOT_VMAX)
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"scatter_ALL_dekads_{YEAR}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -- MAIN ----------------------------------------------------------------------
def main():
    import csv

    out_dir = os.path.join(BASE_OUT_DIR, VARIABLE)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"minmax_{VARIABLE}_{YEAR}.csv")

    print(f"\n{'=' * 60}")
    print(f"  Variable : {VARIABLE.upper()}")
    print(f"  Year     : {YEAR}")
    print(f"  Output   : {out_dir}")
    print(f"  CSV      : {csv_path}")
    print(f"{'=' * 60}\n")

    common_transform, c_width, c_height = build_common_grid()

    tif_dict, envi_dict = discover_files()
    common_keys = sorted(set(tif_dict) & set(envi_dict))

    if not common_keys:
        print("No matching dekad pairs found. Check directory paths and filenames.")
        return

    print(f"Found {len(common_keys)} matching dekad pairs: "
          f"dekads {[k + 1 for k in common_keys]}\n")

    all_x = []
    all_y = []

    # Raw valid pixels accumulated for the ALL summary row
    tif_all_vals  = []
    envi_all_vals = []

    csv_rows = []  # collected during the loop, written once at the end

    for idx in common_keys:
        mmdd   = dekad_index_to_mmdd(idx)
        dk_num = tif_dekad_number(idx)
        label  = f"Dekad {dk_num} ({YEAR}-{mmdd[:2]}-{mmdd[2:]})"
        print(f"Processing {label} ...")

        tif_data, tif_transform, tif_crs = load_tif(tif_dict[idx])
        tif_valid = tif_data[np.isfinite(tif_data)]
        tif_min   = float(np.min(tif_valid))
        tif_max   = float(np.max(tif_valid))
        tif_mean  = float(np.mean(tif_valid))
        print(f"  TIF  this dekad : min={tif_min:.3f}  max={tif_max:.3f}  avg={tif_mean:.3f} {UNIT_LABEL}")
        tif_all_vals.append(tif_valid)

        envi_data, envi_geo = load_envi_img(envi_dict[idx])
        envi_valid = envi_data[np.isfinite(envi_data)]
        envi_min   = float(np.min(envi_valid))
        envi_max   = float(np.max(envi_valid))
        envi_mean  = float(np.mean(envi_valid))
        print(f"  ENVI this dekad : min={envi_min:.3f}  max={envi_max:.3f}  avg={envi_mean:.3f} {UNIT_LABEL}")
        envi_all_vals.append(envi_valid)

        csv_rows.append({"dekad": int(dk_num), "min": round(tif_min,  3),
                         "max": round(tif_max,  3), "average": round(tif_mean,  3),
                         "type": "TIFF (v3)"})
        csv_rows.append({"dekad": int(dk_num), "min": round(envi_min, 3),
                         "max": round(envi_max, 3), "average": round(envi_mean, 3),
                         "type": "ENVI (v2)"})

        tif_g  = tif_to_common_grid(tif_data, tif_transform, tif_crs,
                                     common_transform, c_width, c_height)
        envi_g = envi_to_common_grid(envi_data, envi_geo,
                                      common_transform, c_width, c_height)

        valid   = np.isfinite(tif_g) & np.isfinite(envi_g)
        n_valid = valid.sum()
        print(f"  Valid pixel pairs : {n_valid:,}")

        x = tif_g[valid].ravel()
        y = envi_g[valid].ravel()

        out_pair = os.path.join(out_dir, f"scatter_dekad_{dk_num}_{mmdd}.png")
        make_scatterplot(
            x, y,
            label_x=f"{LABEL_AGERA5} - dekad {dk_num}",
            label_y=f"{LABEL_ERA5}   - dekad {dk_num}",
            title=f"v3 vs v2  |  {VARIABLE.upper()}  |  {label}",
            out_path=out_pair,
        )

        all_x.append(x)
        all_y.append(y)
        print()

    # -- ALL row: stats across every dekad's raw valid pixels ------------------
    tif_concat  = np.concatenate(tif_all_vals)
    envi_concat = np.concatenate(envi_all_vals)

    csv_rows.append({"dekad": "ALL",
                     "min":     round(float(np.min(tif_concat)),  3),
                     "max":     round(float(np.max(tif_concat)),  3),
                     "average": round(float(np.mean(tif_concat)), 3),
                     "type": "TIFF (v3)"})
    csv_rows.append({"dekad": "ALL",
                     "min":     round(float(np.min(envi_concat)),  3),
                     "max":     round(float(np.max(envi_concat)),  3),
                     "average": round(float(np.mean(envi_concat)), 3),
                     "type": "ENVI (v2)"})

    # -- Write CSV -------------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dekad", "min", "max", "average", "type"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved CSV: {csv_path}\n")

    # -- Combined scatterplot --------------------------------------------------
    if all_x:
        print("Generating combined scatterplot ...")
        make_combined_scatterplot(all_x, all_y, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()