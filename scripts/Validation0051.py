# import libraries

import earthaccess
import xarray as xr
import dask
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.feature as cfeature
from rasterio import features
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from pathlib import Path

# authenticate NASA earth access

auth = earthaccess.login(strategy='interactive', persist = True)

# search NASA database

results = earthaccess.search_data(
    short_name='NSIDC-0051',
    temporal=('2012-01-01', '2012-03-20'),
    bounding_box=(-180, 0, 180, 90),
    cloud_hosted=True
)

# open files in earthaccess

files = earthaccess.open(results)
ds = xr.open_mfdataset(files, chunks={'time' : 10})

# read in land file from geopandas

land = gpd.read_file("../data/ne_10m_land/ne_10m_land.shp")
land = land.to_crs(epsg=3411)

# create affine transform and make sure arctic is not upside down

dx = float(ds.x.diff('x').mean())
dy = float(ds.y.diff('y').mean())
x0 = float(ds.x.min())
y0 = float(ds.y.max())

transform = [dx, 0, x0, 0, -abs(dy), y0]

# use transform to create land mask

land_mask = features.rasterize(
    ((geom, 1) for geom in land.geometry),
    out_shape=(ds.sizes['y'], ds.sizes['x']),
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

# change ice concentration variable to something universal

icecon_vars = [var for var in ds.data_vars if 'ICECON'in var]
icecon_vars.sort(reverse=True)

# create template
temp = ds[icecon_vars[0]]
icecon = xr.full_like(temp, np.nan).chunk(temp.chunks)

# loop through and replace NaN values with next latest sat if available
for var in icecon_vars:
    data = ds[var]
    icecon = xr.where(xr.ufuncs.isnan(icecon), data, icecon)

# save dataset without extra sat data
ds["pmw_icecon"] = icecon
ds = ds.drop_vars(icecon_vars)

# read in visual files

folderpath = '../local_data/earth_engine_demos/'
pathlist = Path(folderpath).glob("*.csv")

# convert rows and columns into lats and lons

row_to_lat = dict(enumerate(ds['x'].values))
col_to_lon = dict(enumerate(ds['y'].values))

# loop through files

for i, file in enumerate(pathlist):

    # read in single file and map to lat / lon

    visual = pd.read_csv(str(file))
    visual["time"] = pd.to_datetime(visual["Date"], yearfirst=True)
    visual['x'] = visual['Row'].map(row_to_lat)
    visual['y'] = visual['Column'].map(col_to_lon)

    # convert to xarray

    da_sparse = visual.set_index(['time', 'y', 'x']).to_xarray()
    da_full = da_sparse.reindex_like(ds, method=None).chunk({'time': 2})

    # concatinate into the main dataset (or create new data variable for first file)

    if i == 0:
        ds['visual_icecon'] = da_full['SI frac']
    else:
        ds['visual_icecon'] = xr.concat([ds['visual_icecon'], da_full['SI frac']], dim='time')

# data cleaning (1.012 = coast, 1.016 = land)

condition = ((ds.visual_icecon.notnull()) & (ds.pmw_icecon < 1.01))
ds_clean = ds.where(condition, other=np.nan).compute()

# assign a new column

ds.assign(error=lambda x: x.pmw_icecon - x.visual_icecon)

# download

ds.to_zarr("/scratch/fld1/sea_ice_team.zarr", consolidated=True)