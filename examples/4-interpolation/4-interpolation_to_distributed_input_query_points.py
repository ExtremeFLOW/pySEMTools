# In this example, we provide a different set of points to each rank

# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np

# Get mpi info
comm = MPI.COMM_WORLD

from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry

msh = Mesh(comm, create_connectivity=True)
fld = FieldRegistry(comm)
fname = '../data/rbc0.f00001'
pynekread(fname, comm, data_dtype=np.double, msh=msh, fld=fld)

# Import helper functions
import pysemtools.interpolation.utils as interp_utils
import pysemtools.interpolation.pointclouds as pcs

# Create the points in a distributed way in z direction
# This will only really work for up to 8 ranks with these sizes

if comm.Get_size() > 8:
    print("This example is only valid for 1, 2, 4, 8 ranks or less")
    comm.Abort(1)

x_bbox = [0, 0.05]
y_bbox = [0, 2*np.pi]
z_bbox = [0, 1]
nx = 30
ny = 30
nz = 80
nz_in_rank = int(nz/comm.Get_size())

# Generate the 1D mesh
x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")

# Generate a 3D mesh
r_, th_, z_ = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
#r_, th_, z_ = np.meshgrid(x_1d, y_1d, z_1d)
x_ = r_*np.cos(th_)
y_ = r_*np.sin(th_)

r = np.copy(r_[:,:,nz_in_rank*comm.Get_rank():nz_in_rank*(comm.Get_rank()+1)])
th = np.copy(th_[:,:,nz_in_rank*comm.Get_rank():nz_in_rank*(comm.Get_rank()+1)])
x = np.copy(x_[:,:,nz_in_rank*comm.Get_rank():nz_in_rank*(comm.Get_rank()+1)])
y = np.copy(y_[:,:,nz_in_rank*comm.Get_rank():nz_in_rank*(comm.Get_rank()+1)])
z = np.copy(z_[:,:,nz_in_rank*comm.Get_rank():nz_in_rank*(comm.Get_rank()+1)])


print(x.shape)

# Array the points as a list of probes
xyz = interp_utils.transform_from_array_to_list(nx,ny,nz_in_rank,[x, y, z])

from pysemtools.interpolation.probes import Probes

probes = Probes(comm, probes = xyz, msh = msh, point_interpolator_type='multiple_point_legendre_numpy', max_pts=256, find_points_comm_pattern='point_to_point')

probes.interpolate_from_field_list(0, [fld.registry['w']], comm, write_data=True)
polar_fields = interp_utils.transform_from_list_to_array(nx,ny,nz_in_rank,probes.interpolated_fields)


# Check the results individually per rank
w = polar_fields[1]
w_2d = np.mean(w, axis=1)
levels = 500
levels = np.linspace(-0.07, 0.07, levels)
cmapp='RdBu_r'
fig, ax = plt.subplots(1, 1,figsize=(5, 5))
c1 = ax.tricontourf(r[:,0,:].flatten(), z[:,0,:].flatten() ,w_2d.flatten(), levels=levels, cmap=cmapp)
fig.colorbar(c1)
ax.set_xlabel("r")
ax.set_ylabel("z")
ax.set_xlim([0, 0.05])
ax.set_ylim([0, 1])
plt.show()