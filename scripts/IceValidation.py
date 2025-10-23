# import libraries

import earthaccess
import xarray as xr
import dask
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from rasterio import features
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj, Transformer
from pathlib import Path
import glob
import fnmatch

# colormap for plotting sea ice throughout rest of project

cmap = plt.get_cmap("Blues_r").copy()
cmap.set_bad(color='lightgray')

## PASSIVE MICROWAVE SECTION ##
print('Reading in NASA Team and Boostrap data from NASA Eartaccess')

# authenticate NASA earth access

auth = earthaccess.login(strategy='interactive', persist = True)

# search NASA database for Team

results = earthaccess.search_data(
    short_name= 'NSIDC-0051',
    temporal=('2023-01-01', '2023-01-31'),
    bounding_box=(-180, 0, 180, 90),
    cloud_hosted=True
)

import re
filtered_results = [
    g for g in results
    if re.search(r'_20\d{6}_', g.data_links(access='external')[0])
]

# get files from NSIDC

files_team = earthaccess.open(filtered_results)

# search NASA database for Bootstrap

results = earthaccess.search_data(
    short_name= 'NSIDC-0079',
    temporal=('2023-01-01', '2023-01-31'),
    bounding_box=(-180, 0, 180, 90),
    cloud_hosted=True
)

filtered_results = [
    g for g in results
    if re.search(r'_20\d{6}_', g.data_links(access='external')[0])
]

# get files from NSIDC

files_bootstrap = earthaccess.open(filtered_results)

# stream team into xarray

ds = xr.open_mfdataset(
    files_team, 
    parallel = True, 
    concat_dim="time", 
    combine="nested", 
    data_vars='minimal', 
    coords='minimal', 
    compat='override'
)

# change ice concentration variable to something universal

icecon_vars = sorted([var for var in ds.data_vars if 'ICECON'in var], reverse=True)
icecon = ds[icecon_vars].to_array("source").max("source", skipna=True)

# add back to dataset and clean up
ds = ds.assign(team_icecon=icecon).drop_vars(icecon_vars)

# stream bootstrap into xarray
ds_bootstrap = xr.open_mfdataset(
    files_bootstrap, 
    parallel = True, 
    concat_dim="time", 
    combine="nested", 
    data_vars='minimal', 
    coords='minimal', 
    compat='override'
)

# change ice concentration variable to something universal

icecon_vars = sorted([var for var in ds_bootstrap.data_vars if 'ICECON'in var], reverse=True)
icecon = ds_bootstrap[icecon_vars].to_array("source").max("source", skipna=True)

# add back to dataset and clean up
ds = ds.assign(bootstrap_icecon=icecon)

# read in land file from geopandas and initialize transform (from NSIDC metadata)

land = gpd.read_file("../data/ne_10m_land/ne_10m_land.shp")
land = land.to_crs(epsg=3411)
transform = [25000, 0, -3830000, 0, -25000, 5830000]

# use transform to mask out coastal cells

land_mask = features.rasterize(
    ((geom, 1) for geom in land.geometry),
    out_shape=(448, 304),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# calculate distance from land using euclidian distance transform

distance_from_land = distance_transform_edt(land_mask == 0)

# convert to xarray.DataArray

distance_xr = xr.DataArray(
    distance_from_land,
    coords={'y': ds.y, 'x': ds.x},
    dims=('y', 'x'),
    name='distance_to_land_cells'
)

# add as data variable in ds

ds['edtl'] = distance_xr
print(ds)

## VISUAL ICE SECTION ##
print('Reading in and engineering visual data')

# read in files

folderpath = '../local_data/visual_ice/'
paths = Path(folderpath).glob('*20??*.csv')
pathlist = list(paths)

# data cleaning of visual datasets

visual = pd.concat(map(pd.read_csv, pathlist), ignore_index=True)

# convert things for xarray

row_to_lat = dict(enumerate(ds['x'].values))
col_to_lon = dict(enumerate(ds['y'].values))
visual["time"] = pd.to_datetime(visual["Date"], yearfirst=True)

visual['Row'] = visual['Row'] - 1
visual['Column'] = visual['Column'] - 1
visual['x'] = visual['Row'].map(row_to_lat)
visual['y'] = visual['Column'].map(col_to_lon)
visual = visual.drop_duplicates(subset=["Date", "x", "y"])

# convert pandas dataframe of visual things into chunked xarray dataset on time, lat, and lon

da_sparse = visual.set_index(['time', 'y', 'x']).to_xarray()
da_full = da_sparse.reindex_like(ds, method=None)
da_full = da_full.chunk({'time': 2})
da_full

# assign visual data to the NASA team dataset

ds = ds.assign(**{'visual_ice': da_full['SI frac']})

## ERROR CALCULATIONS ##
print('Starting error calculations')

# data cleaning (1.012 = coast, 1.016 = land)

condition = ((ds.visual_ice.notnull()) & (ds.team_icecon < 1.01))
ds_clean = ds.where(condition, other=np.nan).compute()

df_team = ds_clean.to_dataframe().reset_index().dropna()


condition = ((ds.visual_ice.notnull()) & (ds.bootstrap_icecon < 1.01))
ds_clean = ds.where(condition, other=np.nan).compute()

df_bootstrap = ds_clean.to_dataframe().reset_index().dropna()

## CONTINUE ERROR CALCULATION SECTION HERE ##

error = ((((ds_clean['F17_ICECON'] - ds_clean['visual_ice']))**2)**0.5)
error_avg = error.mean(dim=['time', 'x', 'y'], skipna=True)
print('RMS error is', error_avg.compute().item())

df_team.to_csv('team_error_dataframe.csv')
df_bootstrap.to_csv('bootstrap_error_dataframe.csv')