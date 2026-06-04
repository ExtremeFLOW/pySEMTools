import torch
# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import cProfile

# Get mpi info
comm = MPI.COMM_WORLD

from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import FieldRegistry

msh = Mesh(comm, create_connectivity=True)
fld = FieldRegistry(comm)
pynekread('../data/rbc0.f00001', comm, data_dtype=np.double, msh=msh, fld=fld)

# Create the interpolation points
xyz = [msh.x.flatten(), -msh.y.flatten(), msh.z.flatten()]
xyz = np.array(xyz).T

# Import helper functions
import pysemtools.interpolation.utils as interp_utils
import pysemtools.interpolation.pointclouds as pcs


if comm.Get_rank() == 0 :
    # Generate the bounding box of the points
    x_bbox = [0, 0.05]
    y_bbox = [0, 2*np.pi]
    z_bbox = [0 , 1]
    nx = 30
    ny = 30
    nz = 80
    
    # Generate the 1D mesh
    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")

    # Generate a 3D mesh
    r, th, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x = r*np.cos(th)
    y = r*np.sin(th)

    # Array the points as a list of probes
    xyz = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])

    # Write the points for future use
    with open('points.csv', 'w') as f:
        for i in range((xyz.shape[0])):
            f.write(f"{xyz[i][0]},{xyz[i][1]},{xyz[i][2]}\n")
else:
    xyz = None



# Interpolate the interpolation module
from pysemtools.interpolation.probes import Probes

profile = cProfile.Profile()
profile.enable()

probes = Probes(comm, probes = xyz, msh = msh, point_interpolator_type='multiple_point_legendre_numpy', 
                max_pts=256, find_points_comm_pattern='rma', global_tree_nbins=int(2048*2), local_data_structure='rtree',
                global_tree_type='rank_bbox', find_points_iterative=[True, 1], find_points_tol=1e-7, elem_percent_expansion=0.01, find_points_max_iter=10)

profile.disable()
profile.dump_stats(file=f'probes_rank{comm.Get_rank()}.prof')

probes.interpolate_from_field_list(0, [msh.x], comm, write_data=False)

if comm.Get_rank() == 0:
    print(xyz.shape, probes.interpolated_fields.shape)
    print(np.allclose(probes.interpolated_fields[:,1], xyz[:,0]))