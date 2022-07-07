import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
import dm_local_constants as dm_cst


target_country = 'Ukraine' #'South Africa'
ASAP_IMAGE_TEMPLATE = dm_cst.NDVI_FN_TEMPLATE_time.format('20011001')

if target_country == 'South Africa':
    ADMIN_SHAPE = dm_cst.ADMIN_SHAPE_Global_gaul1
    OUT_TEMPLATE = dm_cst.TEMPLATE_OUTPUT_ZAwest
    COUNTRY = target_country
    adm0field = 'name0'
elif target_country == 'Ukraine':
    ADMIN_SHAPE = dm_cst.ADMIN_SHAPE_Global_gaul1
    OUT_TEMPLATE = dm_cst.TEMPLATE_OUTPUT_Ukraine
    COUNTRY = target_country
    adm0field = 'adm0_name'

features = gpd.read_file(ADMIN_SHAPE)
features = features[features[adm0field] == COUNTRY]

# bounding box
bbox = features.bounds
# get crs calculate window
with rasterio.open(ASAP_IMAGE_TEMPLATE) as src:
    # window=from_bounds(left, bottom, right, top, src.transform),
    win = from_bounds(bbox.minx.min(), bbox.miny.min(), bbox.maxx.max(), bbox.maxy.max(), src.transform)
    win = win.round_lengths()
    win = win.round_offsets()
    tmp = win.todict()
    tmp['height'] += 2
    tmp['width'] += 2
    win = Window(**tmp)

    data = src.read(1, window=win)
    win_transform = src.window_transform(win)
    crs = src.crs


with rasterio.open(OUT_TEMPLATE, 'w+', driver='GTiff',
                   height=win.height,
                   width=win.width,
                   count=1,
                   dtype=data.dtype,
                   crs=crs,
                   transform=win_transform,
                   ) as dst:
    dst.write(data, 1)