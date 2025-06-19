## this is a library of functions that help get, process, and analyze coastal sea ice data from NSIDC products ##

# general libraries
import earthaccess
import xarray as xr
import dask
import numpy as np

# libraries for coastal masking function
import geopandas as gpd
import cartopy.feature as cfeature
from rasterio import features
from scipy.ndimage import convolve


## coastal mask preprocessing function ##

def select_coastal(ds):

    '''helper function for use in preprocessing to select coastal data'''
    
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

## function to import NSIDC 0051 data ##

def cnget(name : str, sdate : str, edate : str):

    '''function that returns coastal sea ice data for an NSIDC product'''

    # authenticate (you need a NASA earthaccess account for this)
    auth = earthaccess.login(strategy='interactive', persist = True)

    # search for results (testing with just a couple days)
    results = earthaccess.search_data(
        short_name=name,
        temporal=(sdate, edate),
        bounding_box=(-180, 0, 180, 90)
    )

    # open results with xarray
    files = earthaccess.open(results)
    ds = xr.open_mfdataset(files, parallel=True, combine='by_coords', preprocess=select_coastal)
    
    # return
    return ds