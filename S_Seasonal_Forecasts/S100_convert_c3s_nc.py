# To use gdal from command line open OSGeo4W Shell
# cd V:\foodsec\Projects\ECMWF-SEAS5\asap download 2024 06 06 p
# v:
# gdalinfo ecmwf-51_2024_06_6.nc
# gdal_translate NETCDF:"ecmwf-51_2024_06_6.nc":tprate ecmwf-51_2024_06_6.tif
# gdal_translate -ot Float32 -scale 3.181800470872867e-07 9.71067713749883e-12 -a_nodata -32767 NETCDF:"ecmwf-51_2024_06_6.nc":tprate ecmwf-51_2024_06_6_scaled.tif


# gdal_translate -a_srs EPSG:4326 -unscale -ot Float64 NETCDF:"ecmwf-51_2024_06_6.nc":tprate ecmwf-51_2024_06_6_unscaled_64.tif
# gdal_calc --allBands=A -A ecmwf-51_2024_06_6_unscaled_64.tif --outfile=ecmwf-51_2024_06_6_mmday_64.tif --calc="A*(A!=-32767)*86400000" --NoDataValue=-32767