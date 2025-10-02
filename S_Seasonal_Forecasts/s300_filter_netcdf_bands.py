import xarray as xr
import pandas as pd

nc_file_in = r'V:\asap\Seasonal_forecasts\monthly-precipitation\ecmwf-51\historical\monthly-means\06\ecmwf-51_06.nc'
nc_file_out = r'V:\asap\Seasonal_forecasts\monthly-precipitation\ecmwf-51\historical\monthly-means\06\ecmwf-51_06-06.nc'
# Open the NetCDF file
ds = xr.open_dataset(nc_file_in)
# Time is expressed in hours since 1900-01-01 00:00:00.0
# Define the base date
base_date = pd.to_datetime('1900-01-01 00:00:00')

# Convert the 'time' dimension to datetime
# time_dates = base_date + pd.to_timedelta(ds.time.values, unit='h')

# Create a new 'time' coordinate with the datetime values
# ds = ds.assign_coords(time=time_dates)
# Print all elements of the 'time' dimension
print("All time elements:")
print(ds.time.values)

# Filter the dataset to include only data from month 11 (November)
ds_nov = ds.where(ds.time.dt.month == 11, drop=True)
# Save the filtered dataset to a new NetCDF file
ds_nov.to_netcdf(nc_file_out)


# import netCDF4 as nc
# import numpy as np
# from datetime import datetime, timedelta
#
# nc_file = r'V:\asap\Seasonal_forecasts\monthly-precipitation\ecmwf-51\historical\monthly-means\06\ecmwf-51_06.nc'
#
# ds = nc.Dataset(nc_file, 'r')
#
# vars = ['number', 'time']
#
# # Get the time variable
# time_var = ds.variables['time']
#
#
# # Convert the time variable to a date
# time_dates = nc.num2date(time_var[:], time_var.units)
#
# # Print the dates
# for i, time_date in enumerate(time_dates):
#     print(f"Time[{i}] = {time_date.year}-{time_date.month:02d}-{time_date.day:02d}")
#
#
# if 'number' in ds.variables:
#     print("\nValues of dimension 'number':")
#     print(ds.variables['number'][:])
# if 'time' in ds.variables:
#     print(f"The type of the variable 'time' is: {ds.variables['time'].dtype}")
#     print("\nValues of dimension 'time':")
#     print(ds.variables['time'][:])
#
#
# print(ds.dimensions)