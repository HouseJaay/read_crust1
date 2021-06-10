import numpy as np
from os import path
from scipy.spatial import cKDTree


##############################
# Set parameters
##############################
c1_path = "../crust1/" # path to crust1.0 file
vtype = "vp" # vp, vs or rho
sel_layer = slice(3, None) # ignore water and ice layer.
latmin, latmax = 40, 45
lonmin, lonmax = 10, 15
depmin, depmax = -40, 2 # the deeeper, the lower
dlat, dlon, ddep = 0.01, 0.01, 1 # grid size after interpolation

##############################
# Reading data
##############################
layer = np.loadtxt(path.join(c1_path, "crust1.bnds"))[:, sel_layer]
data = np.loadtxt(path.join(c1_path, "crust1." + vtype))[:, sel_layer]

lats = np.arange(89.5, -89.5-1, -1)
lons = np.arange(-179.5, 179.5+1, 1)
nlat, nlon = len(lats), len(lons)
grids = np.zeros([nlat*nlon, 2])
for i in range(nlat):
    grids[i*nlon: (i+1)*nlon, 0] = lats[i]
    grids[i*nlon: (i+1)*nlon, 1] = lons

mask = (grids[:, 0] > latmin-1) & (grids[:, 0] < latmax+1) & \
        (grids[:, 1] > lonmin-1) & (grids[:, 1] < lonmax+1)
grids = grids[mask]
layer = layer[mask]
data = data[mask]

#####################################
# Vertical interpolation
#####################################
ideps = np.arange(depmax, depmin-ddep, -ddep)
def interp_dep(idep):
    # require odeps in descending order
    if idep > odeps[0]:
        return ovels[0]
    for i in range(1, len(odeps)):
        if idep > odeps[i]:
            du = abs(idep - odeps[i-1])
            db = abs(idep - odeps[i])
            return (ovels[i-1] * du**2 + ovels[i] * db**2) / (du**2 + db**2)
    return ovels[-1]
indp = np.vectorize(interp_dep)
# iterate over each grid
data_idep = np.zeros([len(data), len(ideps)])
for i in range(len(data)):
    odeps = layer[i][data[i]!=0]
    ovels = data[i][data[i]!=0]
    indp = np.vectorize(interp_dep)
    # print(odeps)
    # print(ovels)
    # print("\n")
    data_idep[i, :] = indp(ideps)

#####################################
# prepare horizontal interpolation
#####################################
tree = cKDTree(grids)
ilats = np.arange(latmin, latmax, dlat)
ilons = np.arange(lonmin, lonmax, dlon)
ila, ilo = np.meshgrid(ilats, ilons)
igrids = np.c_[ila.reshape(-1, 1), ilo.reshape(-1, 1)]
d, inds = tree.query(igrids, k=4, distance_upper_bound=1.5)
w = 1.0 / d**2

data_i = np.zeros(len(igrids) * len(ideps))
grids_fin = np.zeros([len(igrids) * len(ideps), 3])
ng = len(igrids)
for i in range(len(ideps)):
    grids_fin[i*ng: (i+1)*ng, 2] = ideps[i]
    grids_fin[i*ng: (i+1)*ng, :2] = igrids
    data_i[i*ng: (i+1)*ng] = np.sum(w * data_idep[inds, i], axis=1) / np.sum(w, axis=1)

##################################
# Write to file
##################################
np.savetxt("tmp.dat", np.c_[grids_fin, data_i], fmt='%.5f')
# np.save("tmp.npy", np.c_[grids_fin, data_i])