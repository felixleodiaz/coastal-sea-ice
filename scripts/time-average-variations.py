# general libraries
import earthaccess
import xarray as xr
import dask
import numpy as np

# CNSIDC library
from CNSIDC import ice_get, ice_combine

# fetch data 1990-2020 for NSIDC 0051 and 0079

ds51 = ice_get('NSIDC-0051', '1990-01-01', '2020-12-31')
ds51 = ice_combine(ds51)

ds79 = ice_get('NSIDC-0079', '1990-01-01', '2020-12-31')
ds79 = ice_combine(ds79)

# check info

print(ds51.info())
print(ds79.info())

# average entire dataset

ds51_avg = ds51.icecon.mean(dim='time')
ds79_avg = ds79.icecon.mean(dim='time')

# get difference array

ds_avg_dif = ds51_avg - ds79_avg

# plot these

print(ds51_avg.plot(cmap='coolwarm', cbar_kwargs={'label': 'Sea Ice Concentration (%)'}))
print(ds79_avg.plot(cmap='coolwarm', cbar_kwargs={'label': 'Sea Ice Concentration (%)'}))
print(ds_avg_dif.plot(cmap='coolwarm', cbar_kwargs={'label': 'Sea Ice Concentration (%)'}))