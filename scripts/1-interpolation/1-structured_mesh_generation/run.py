# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import json
import h5py

# Get mpi info
comm = MPI.COMM_WORLD

# This example is designed to work in one rank only
if comm.Get_size() > 1:
    raise ValueError("This example is designed to run with one rank only")

import pysemtools.interpolation.pointclouds as pcs

def main():

    # Read the inputs file
    with open('inputs.json', 'r') as f:
        inputs = json.load(f)

    # Generate the bounding box of the points
    r_bbox = [inputs["dir_1"]["min_value"], inputs["dir_1"]["max_value"]]
    th_bbox = [inputs["dir_2"]["min_value"], inputs["dir_2"]["max_value"]]
    z_bbox = [inputs["dir_3"]["min_value"], inputs["dir_3"]["max_value"]]
    nx = inputs["dir_1"]["number_of_points"]
    ny = inputs["dir_2"]["number_of_points"]
    nz = inputs["dir_3"]["number_of_points"]

    # Generate the 1D mesh
    r_1d = pcs.generate_1d_arrays(r_bbox, nx, mode=inputs["dir_1"]["point_distribution"], gain=inputs["dir_1"]["point_distribution_gain"])
    th_1d = pcs.generate_1d_arrays(th_bbox, ny, mode=inputs["dir_2"]["point_distribution"], gain=inputs["dir_2"]["point_distribution_gain"])
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode=inputs["dir_3"]["point_distribution"], gain=inputs["dir_3"]["point_distribution_gain"])

    # Generate differetials (dr, dth, dz)
    dr_1d  = pcs.generate_1d_diff(r_1d, periodic=inputs["dir_1"]["periodic"])
    dth_1d = pcs.generate_1d_diff(th_1d, periodic=inputs["dir_2"]["periodic"]) # This is needed to give the same weight to the first and last points as for the other ones. Needed if fourier transform will be applied.
    dz_1d  = pcs.generate_1d_diff(z_1d, periodic=inputs["dir_3"]["periodic"])

    # Generate a 3D mesh
    z, th, r = np.meshgrid(z_1d, th_1d, r_1d, indexing='ij')
    # Generate 3D differentials
    dz, dth, dr = np.meshgrid(dz_1d, dth_1d, dr_1d, indexing='ij')

    # Generate xy coordinates, which are needed for probes
    x = r*np.cos(th)
    y = r*np.sin(th)

    # For volume
    B = r * dr * dth * dz
    # For area given each angle slice
    A = dr * dz

    with h5py.File(inputs["output_fname"], 'w') as f:

        # Create a header
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nz'] = nz

        # Include data sets
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('z', data=z)
        f.create_dataset('r', data=r)
        f.create_dataset('th', data=th)
        f.create_dataset('mass', data=B)
        f.create_dataset('mass_area', data=A)

if __name__ == "__main__":
    main()