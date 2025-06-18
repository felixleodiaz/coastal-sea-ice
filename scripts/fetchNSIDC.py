# import libraries

import earthaccess
import xarray as xr
import dask
import numpy as np

# authenticate (you need a NASA earthaccess account for this)

auth = earthaccess.login(strategy='interactive', persist = True)

# search for results (testing with just a couple days)

results = earthaccess.search_data(
    short_name='NSIDC-0051',
    temporal=('2021-11-01', '2021-12-01'),
    bounding_box=(-180, 0, 180, 90)
)

# initialize coastal masking function

import geopandas as gpd
import cartopy.feature as cfeature
from rasterio import features
from scipy.ndimage import convolve

def select_coastal(ds):

    # load land polygons and reproject to EPSG:3411

    land = gpd.read_file("../data/ne_10m_land/ne_10m_land.shp")
    land = land.to_crs(epsg=3411)

    # get transform for rasterizing

    dx = float(ds.x.diff('x').mean())  # 25000 meters
    dy = float(ds.y.diff('y').mean())  # 25000 meters
    x0 = float(ds.x.min())
    y0 = float(ds.y.min())
    transform = [dx, 0, x0, 0, -dy, y0]

    # rasterize land mask: 1 = land, 0 = ocean

    land_mask = features.rasterize(
        ((geom, 1) for geom in land.geometry),
        out_shape=(ds.sizes['y'], ds.sizes['x']),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # create coastal mask (within 3 grid cells of land)

    ocean = (land_mask == 0).astype(int)
    kernel = np.ones((7, 7))  # 3-cell radius
    land_neighbor_count = convolve(1 - ocean, kernel, mode='constant', cval=0)
    coastal_mask = (ocean == 1) & (land_neighbor_count > 0)

    # convert to xarray.DataArray

    coastal_mask_xr = xr.DataArray(
        coastal_mask,
        coords={'y': ds.y, 'x': ds.x},
        dims=('y', 'x')
    )

    # apply mask

    return ds.where(coastal_mask_xr)

# open results with xarray

files = earthaccess.open(results)
ds = xr.open_mfdataset(files, parallel=True, combine='by_coords', preprocess=select_coastal)

# save ds on hpcc scratch dir