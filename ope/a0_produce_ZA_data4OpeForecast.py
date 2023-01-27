from preprocess import b20_LoadCsv_savePickle, b60_build_features
from ope import b100_usetting_and_mod_manager_nrt_v2,b200_gather_outputs_nrt
import src.constants as cst

target = 'ZAsummer'
forecast_month = 'January'
forcast_year = 2023 #this is the harvest year, so in operation during 2022, it will be 2023

forecasting_times = {'November': 1, 'December': 2, 'January': 3, 'February': 4, 'March': 5, 'April': 6, 'May': 7,}
# store the updated data in: -PY_data-ML1_data_input-Predictors_4OPE
if True:
    b20_LoadCsv_savePickle.LoadCsv_savePickle(target, 'Predictors_4OPE', ope_run=True)
    b60_build_features.build_features(target, ope_run=True)

if False: #Set to true for forecasting all the months (used for testing of year 2021), set to false for operational yield forecast of a single month
    for fm in forecasting_times.keys():
        print(fm)
        dirOutModel = b100_usetting_and_mod_manager_nrt_v2.nrt_model_manager(target=target, forecasting_times= forecasting_times, forecast_month=fm, current_year=forcast_year)
        b200_gather_outputs_nrt.main(target=target, folder=dirOutModel)
else:
    dirOutModel = b100_usetting_and_mod_manager_nrt_v2.nrt_model_manager(target=target, forecasting_times= forecasting_times, forecast_month=forecast_month, current_year=forcast_year)
    b200_gather_outputs_nrt.main(target=target, folder=dirOutModel)


def rpss_map_pheno_integ(nc, out_img, overlay_shp=None, bbox=None, plot_width=6, plot_height=6,
             plot_adjust=(0.01, 0.15, 0.99, 0.975), outdpi=300, parallels_conf=(10, (0, 1, 0, 0), 1),
             meridians_conf=(10, (0, 0, 0, 1), 1), meshgrid=True, add_coastlines=False, overlay_line_width=0.5):
    """
    Generates a plot for RPSS variable from an NC file into the indicated output
    :param nc: path to nc file
    :param out_img: path to out_img
    :param overlay_shp: patht to overlay. Default: None
    :param bbox: extend of the picture in lat lon bbox. Default None (extent of the data)
    :param plot_width: width of the plot in inches
    :param plot_height: height of the plot in inches
    :param plot_adjust: subplot adjust parameters according matplotlib https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.subplots_adjust.html .(left,botom,rigth, top)  Defaults =(0.01, 0.15, 0.99, 0.975)
    :param outdpi: out resolution in dpi. Default 300
    :param parallels_conf: parallel configuration array wit 3 values: firts: parallel intervals in degrees ,
            second: array of 4 one-zero values meaning labels for parallels  labels = [left,right,top,bottom] ,
            Third: extra offset for meridians grid.
            Default: [10,[0,1,0,0],1].
    :param meridians_conf: meridians configuration array wit 3 values: firts: meridians intervals in degrees ,
            second: array of 4 one-zero values meaning labels for meridians  labels = [left,right,top,bottom] .
            Default: [10,[0,0,0,1],1].
    :param meshgrid: indicates in data must be converter in a rectangular grid (needed for initial world tested data)
    :param add_coastlines: todo desc param
    :param add_coastLines: adds coastlines to the basemap. Default: false
    :param overlay_line_width: line width for the overlay layer. Default: 0.5
    :return: void
    """
    # store datasetas lonts and lats
    fh = Dataset(nc, mode='r')
    lons = fh.variables['longitude'][:]
    lats = fh.variables['latitude'][:]
    min_lons = min(lons)
    min_lats = min(lats)
    max_lons = max(lons)
    max_lats = max(lats)

    # init map
    if bbox is None:
        bbox = [min_lons - 0.5, min_lats - 0.5, max_lons + 0.5, max_lats + 0.5]

    m, fig = init_map(bbox=bbox, plot_width=plot_width, plot_height=plot_height, plot_adjust=plot_adjust, parallels_conf=parallels_conf,
                      meridians_conf=meridians_conf, add_coastlines=add_coastlines)

    # add_shp_layer
    add_shp_layer(m, overlay_shp, line_width=overlay_line_width)

    # add pclolor data
    data = fh.variables["rpss"][:]
    vmin = -0.5
    vmax = 0.5
    # meshgrid it's needed for global data
    if meshgrid:
        mesh_lons, mesh_lats = np.meshgrid(lons, lats)
        # add pcolor layers
        im1 = m.pcolor(mesh_lons, mesh_lats, data, latlon=True, vmin=vmin, vmax=vmax, shading="auto")
    else:
        im1 = m.pcolor(lons, lats, data, latlon=True, vmin=vmin, vmax=vmax, shading="auto")

    # register color maps in plt class
    attach_color_ramp_to_image('red', 'white', 'blue', im=im1, name='rpss')

    # add_cboxes
    add_colorbox(fig, im1, "RPSS", 0, 1)

    # plot
    fig.savefig(out_img, dpi=outdpi)