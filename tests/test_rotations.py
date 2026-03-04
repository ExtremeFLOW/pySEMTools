# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

import os
import sys

# Import general modules
import numpy as np
# Import relevant modules
from pysemtools.postprocessing.rotations import rotate_tensor, cartesian_to_cylindrical_rotation_matrix
import pysemtools.interpolation.pointclouds as pcs



def test_cartesian_to_cylindrical():

    # Generate the bounding box of the points
    r_bbox = [0, 1]
    th_bbox = [0, 2*np.pi]
    z_bbox = [0 , 1]
    nx = 3
    ny = 30
    nz = 3

    # Generate the 1D mesh
    r_1d = pcs.generate_1d_arrays(r_bbox, nx, mode="equal")
    th_1d = pcs.generate_1d_arrays(th_bbox, ny, mode="equal")
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal", gain=1)

    # Generate differetials (dr, dth, dz)
    dr_1d  = pcs.generate_1d_diff(r_1d)
    dth_1d = pcs.generate_1d_diff(th_1d, periodic=True) # This is needed to give the same weight to the first and last points as for the other ones. Needed if fourier transform will be applied.
    dz_1d  = pcs.generate_1d_diff(z_1d)

    # Generate a 3D mesh
    r, th, z = np.meshgrid(r_1d, th_1d, z_1d, indexing='ij')
    # Generate 3D differentials
    dr, dth, dz = np.meshgrid(dr_1d, dth_1d, dz_1d, indexing='ij')

    # Generate xy coordinates, which are needed for probes
    x = r*np.cos(th)
    y = r*np.sin(th)

    
    rotated_tensor = rotate_tensor(comm, [x, y, z], [np.copy(x), np.copy(y), np.copy(z)], tensor_field_names=['x', 'y', 'z'], rotation_matrix=cartesian_to_cylindrical_rotation_matrix)
    
    rr = rotated_tensor[0]
    thth = rotated_tensor[1]
    zz = rotated_tensor[2]

    # The component of the vector x,y,z in the r direction is the magnitude. And in theta direction is 0, since it is aligned.
    t1 = np.allclose(rr, np.sqrt(x**2 + y**2))
    t2 = np.allclose(thth, np.zeros_like(th))
    t3 = np.allclose(zz, z)

    passed = np.all([t1, t2, t3])

    print(passed)

    assert passed

test_cartesian_to_cylindrical()