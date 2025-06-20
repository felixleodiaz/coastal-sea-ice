# general libraries
import earthaccess
import xarray as xr
import dask
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')

# CNSIDC library
from CNSIDC import ice_get, ice_combine

# fetch data 1990-2020 for NSIDC 0051 and 0079

ds51 = ice_get('NSIDC-0051', '1990-01-01', '2020-12-31')
ds51 = ice_combine(ds51)

ds79 = ice_get('NSIDC-0079', '1990-01-01', '2020-12-31')
ds79 = ice_combine(ds79)

# average entire dataset

ds51_avg = ds51.icecon.mean(dim='time')
ds79_avg = ds79.icecon.mean(dim='time')

# get difference array

ds_avg_dif = ds51_avg - ds79_avg

# Plot 1
fig, ax = plt.subplots(figsize=(8, 6))
ds51_avg.plot(ax=ax, cmap='PuBu', cbar_kwargs={'label': 'Sea Ice Concentration (%)'})
ax.set_title("Average Sea Ice Concentration NSIDC 0051: 1990–2020")
fig.savefig("../figures/NSIDC0051_TOTAL_AVG.png", dpi=300, bbox_inches='tight')
plt.show(fig)
plt.close(fig)

# Plot 2
fig, ax = plt.subplots(figsize=(8, 6))
ds79_avg.plot(ax=ax, cmap='PuBu', cbar_kwargs={'label': 'Sea Ice Concentration (%)'})
ax.set_title("Average Sea Ice Concentration NSIDC 0079: 1990–2020")
fig.savefig("../figures/NSIDC0079_TOTAL_AVG.png", dpi=300, bbox_inches='tight')
plt.show(fig)
plt.close(fig)

# Plot 3: Difference
fig, ax = plt.subplots(figsize=(8, 6))
ds_avg_dif.plot(ax=ax, cmap='bwr', cbar_kwargs={'label': 'Sea Ice Concentration (%)'})
ax.set_title("Difference in Tot Avg Sea Ice: NSIDC 0079 and NSIDC 0051: 1990–2020")
fig.savefig("../figures/TOTAL_AVG_DIF.png", dpi=300, bbox_inches='tight')
plt.show(fig)
plt.close(fig)