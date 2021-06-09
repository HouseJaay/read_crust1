import numpy as np
from os import path

##############################
# Set parameters
##############################
c1_path = "../crust1/" # path to crust1.0 file
vtype = "vp" # vp, vs or rho
latmin, latmax = 40, 45
lonmin, lonmax = 10, 15
dlat, dlon, ddep = 0.01, 0.01, 1 # grid size after interpolation

##############################
# Reading data
##############################
layer = np.loadtxt(path.join(c1_path, "crust1.bnds"))
data = np.loadtxt(path.join(c1_path, "crust1." + vtype))

lats = np.arange(89.5, -89.5-1, -1)
lons = np.arange(-179.5, 179.5+1, 1)
nlat, nlon = len(lats), len(lons)
grids = np.zeros([nlat*nlon, 2])
for i in range(nlat):
    grids[i*nlon: (i+1)*nlon, 0] = lats[i]
    grids[i*nlon: (i+1)*nlon, 1] = lons
