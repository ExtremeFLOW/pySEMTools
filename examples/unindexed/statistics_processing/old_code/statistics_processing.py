# Import general modules
import sys
import os
from pyevtk.hl import gridToVTK

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import matplotlib.pyplot as plt

# Import MPI
from mpi4py import MPI #equivalent to the use of MPI_init() in C
# Import IO helper functions
from pysemtools.io.utils import IoPathData as io_path_data
# Import modules for reading and writing
from pysemtools.io.ppymech.neksuite import preadnek
# Import the data types
from pysemtools.datatypes.msh import Mesh as msh_c
from pysemtools.datatypes.field import Field as field_c
from pysemtools.datatypes.utils import write_fld_file_from_list
# Import types asociated with interpolation
from pysemtools.interpolation.interpolator import get_bbox_from_coordinates, get_bbox_centroids_and_max_dist

# Split communicator for MPI - MPMD
worldcomm = MPI.COMM_WORLD
worldrank = worldcomm.Get_rank()
worldsize = worldcomm.Get_size()
col = 1
comm = worldcomm.Split(col,worldrank)
rank = comm.Get_rank()
size = comm.Get_size()

# Open input file to see path
f = open ("inputs.json", "r")
params_file = json.loads(f.read())
f.close()

# Start time
start_time = MPI.Wtime()

# Read the data paths from the input file
mesh_data = io_path_data(params_file["IO"]["mesh_data"])
field_data = io_path_data(params_file["IO"]["field_data"])
number_of_snapshots = params_file["number_of_snapshots"]
start_time = params_file["start_time"]

# Initialize the mesh file
path     = mesh_data.dataPath
casename = mesh_data.casename
index    = mesh_data.index
fname    = path+casename+'0.f'+str(index).zfill(5)
data     = preadnek(fname, comm)
msh      = msh_c(comm, data = data)
del data

# Initialize coef to get the mass matrix
#coef = coef_c(msh, comm)

# Create an array to store the mean field
mean_u = np.zeros_like(msh.x)
rt = 0 

# Read each statistics files and average them
for snapshot_id in range(0, number_of_snapshots):

    # Recieve the data from fortran
    path     = field_data.dataPath
    casename = field_data.casename
    index    = field_data.index
    fname=path+casename+'0.f'+str(index + snapshot_id).zfill(5)
    if comm.Get_rank() == 0: print(" ==== Reading file: {} ======".format(fname))
    fld_data = preadnek(fname, comm)

    # Get the data in field format
    fld = field_c(comm, data = fld_data)

    # Get the required fields
    t = (fld.t - start_time)
    dt = t - rt
    u = fld.fields["vel"][0]

    # Update the statistics
    mean_u = mean_u * rt
    mean_u = mean_u + u * dt  
    
    rt += dt

    mean_u = mean_u / rt


# Find the centroids of the elements
# Get the bounding box of the elements and their centroids
bbox = get_bbox_from_coordinates(msh.x, msh.y, msh.z)
bbox_centroids, bbox_max_dist = get_bbox_centroids_and_max_dist(bbox)

# Get the unique xy coordinates in the box (the homogenous direction)
structured = np.core.records.fromarrays(bbox_centroids[:,:2].transpose(), names='x, y')
unique_xy, indices = np.unique(structured, return_index=True)
simple_array = unique_xy.view(np.float64).reshape(unique_xy.shape + (-1,))
unique_xy = simple_array

# Allocate an array that will contain 1 element per unique xy coordinate
mean_x_xy = np.zeros((unique_xy.shape[0], msh.lz, msh.ly, msh.lx))
mean_y_xy = np.zeros((unique_xy.shape[0], msh.lz, msh.ly, msh.lx))
mean_u_xy = np.zeros((unique_xy.shape[0], msh.lz, msh.ly, msh.lx))

# Each rank will average what it has
for e in range(0, unique_xy.shape[0]):

    # Peform an average over all elements that share the same xy coordinates
    condition_one = bbox_centroids[:,0] == unique_xy[e,0]
    condition_two = bbox_centroids[:,1] == unique_xy[e,1]

    average_over_e = np.where(np.all([condition_one, condition_two], axis=0))[0]

    mean_x_xy[e,:,:,:] = np.mean(msh.x[average_over_e,:,:,:], axis=0)
    mean_y_xy[e,:,:,:] = np.mean(msh.y[average_over_e,:,:,:], axis=0)
    mean_u_xy[e,:,:,:] = np.mean(mean_u[average_over_e,:,:,:], axis=0)


# Gather the data 
# This works nicely in this case because the data was distributed nicely
# A more general solution needs to be found
averaging_weights = np.ones_like(mean_u_xy)
mean_u_xy_all = np.zeros((unique_xy.shape[0], msh.lz, msh.ly, msh.lx))
weights_xy_all = np.zeros((unique_xy.shape[0], msh.lz, msh.ly, msh.lx))
comm.Allreduce(mean_u_xy, mean_u_xy_all, op=MPI.SUM)
comm.Allreduce(averaging_weights, weights_xy_all, op=MPI.SUM)

# Do a bad average. Here assuming that all the elements in all ranks had the same weight, which might not be true.
# Consider adding proper integration weights
mean_u_xy_all = mean_u_xy_all / weights_xy_all

# Average the data now over the z direction # This is not so good. One would need to use proper integration weights over the element
mean_x_xy_all_2d = np.mean(mean_x_xy, axis = 1).reshape(mean_u_xy_all.shape[0], 1, mean_u_xy_all.shape[2], mean_u_xy_all.shape[3])
mean_y_xy_all_2d = np.mean(mean_y_xy, axis = 1).reshape(mean_u_xy_all.shape[0], 1, mean_u_xy_all.shape[2], mean_u_xy_all.shape[3])
mean_z_xy_all_2d = np.zeros_like(mean_x_xy_all_2d)
mean_u_xy_all_2d = np.mean(mean_u_xy_all, axis=1).reshape(mean_u_xy_all.shape[0], 1, mean_u_xy_all.shape[2], mean_u_xy_all.shape[3])


# Split the communicator so only the first rank writes the data
col = int(np.floor(((comm.Get_rank()/1))))
write_comm= comm.Split(color = col, key=rank)

# Write the data
fname="./"+"2d_stats"+'0.f'+str(1).zfill(5)
if comm.Get_rank() == 0: 
    # Write the data
    msh2d = msh_c(write_comm, x = mean_x_xy_all_2d, y = mean_y_xy_all_2d, z = mean_z_xy_all_2d)
    print(" ==== Writing file: {} ======".format(fname))
    write_fld_file_from_list(fname, write_comm, msh2d, [mean_u_xy_all_2d])
