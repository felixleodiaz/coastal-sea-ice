# general libraries
import earthaccess
import xarray as xr
import dask
import numpy as np

# CNSIDC library
from CNSIDC import cnget

# fetch data 1990-2020 for NSIDC 0051 and 0079

ds0051 = cnget('NSIDC-0051', '1990-01-01', '2020-12-31')
ds0079 = cnget('NSIDC-0079', '1990-01-01', '2020-12-31')

# average every january first

# ds0051_avg = ds0051.F17_ICECON.mean(dim='time')
# ds0079_avg = ds0079.F17_ICECON.mean(dim='time')

# get difference array

# ds_dif = ds0051_avg - ds0079_avg

# plot these

# print(ds0051_avg.plot(cmap='viridis', cbar_kwargs={'label': 'Sea Ice Concentration (%)'}))
# print(ds0079_avg.plot(cmap='viridis', cbar_kwargs={'label': 'Sea Ice Concentration (%)'}))
# print(ds_dif.plot(cmap='viridis', cbar_kwargs={'label': 'Sea Ice Concentration (%)'}))