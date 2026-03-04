# Import the data types
from mpi4py import MPI #equivalent to the use of MPI_init() in C
from pysemtools.io.ppymech.neksuite import preadnek, pwritenek
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field
from pysemtools.interpolation.probes import Probes
import pysemtools.interpolation.utils as interp_utils
import pysemtools.interpolation.pointclouds as pcs
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nx = 64
ny = 64
nz = 64
x_bbox = [0, 1]
y_bbox = [0, 1]
z_bbox = [0, 1]
x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")
x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

xyz = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])

print(xyz.shape)   
# Write xyz into a csv file

if rank == 0:
    with open('points.csv', 'w') as f:
        for i in range((xyz.shape[0])):
            f.write(f"{xyz[i][0]},{xyz[i][1]},{xyz[i][2]}\n")