import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys

# Get mpi info
comm = MPI.COMM_WORLD

# Hide the log for the notebook. Not recommended when running in clusters as it is better you see what happens
import os
os.environ["PYSEMTOOLS_HIDE_LOG"] = 'false'

# =========================
# Download the example data
# =========================

if comm.Get_rank() == 0:
    os.system("git clone https://github.com/adperezm/sem_data.git ../data/sem_data")
comm.Barrier()

# =========================
# Create a point cloud
# =========================

# Get the bounds of the box (Skip if you know them)
from pysemtools.io.wrappers import read_data
sem_mesh_fname = "../data/sem_data/instantaneous/channel_nelv_512/field0.f01601"
data = read_data(comm, fname=sem_mesh_fname, keys=["x", "y", "z"], parallel_io=True)
min_x = comm.allreduce(np.min(data["x"]), op=MPI.MIN)
max_x = comm.allreduce(np.max(data["x"]), op=MPI.MAX)
min_y = comm.allreduce(np.min(data["y"]), op=MPI.MIN)
max_y = comm.allreduce(np.max(data["y"]), op=MPI.MAX)
min_z = comm.allreduce(np.min(data["z"]), op=MPI.MIN)
max_z = comm.allreduce(np.max(data["z"]), op=MPI.MAX)

# Create a 3D mesh and write it down. This is done in rank 0 just to show
import pysemtools.interpolation.pointclouds as pcs

if comm.Get_rank() == 0:
    x_bbox = [min_x, max_x]
    y_bbox = [min_y, max_y]
    z_bbox = [min_z, max_z] # The spanwise size is this one for the data set for some reason
    nx = 8*8
    ny = 8*8
    nz = 8*8

    # Generate the 1D mesh
    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal", gain=1)

    # Generate differetials (dr, dth, dz)
    dx_1d  = pcs.generate_1d_diff(x_1d)
    dy_1d = pcs.generate_1d_diff(y_1d) # This is needed to give the same weight to the first and last points as for the other ones. Needed if fourier transform will be applied.
    dz_1d  = pcs.generate_1d_diff(z_1d, periodic=True)

    # Generate mesh
    x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    print(np.min(x), np.max(x))
    print(np.min(y), np.max(y))
    print(np.min(z), np.max(z))
    # Generate 3D differentials
    dx, dy, dz = np.meshgrid(dx_1d, dy_1d, dz_1d, indexing='ij')
    # Mass matrix
    B = dx*dy*dz
    print(np.sum(B))

    # Write the data
    fname = 'points.hdf5'
    with h5py.File(fname, 'w') as f:

        # Create a header
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nz'] = nz

        # Include data sets
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('z', data=z)
        f.create_dataset('mass', data=B)
    print("point data written")

# =========================
# Interpolate the data
# =========================
from pysemtools.interpolation.wrappers import interpolate_fields_from_disk

query_points_fname = "./points.hdf5"
sem_mesh_fname = "../data/sem_data/instantaneous/channel_nelv_512/field0.f01601"
sem_dtype = np.single
nsnapshots = 400
field_interpolation_dictionary = {}
field_interpolation_dictionary['input_type'] = "file_sequence"
field_interpolation_dictionary['file_sequence'] = [f"../data/sem_data/instantaneous/channel_nelv_512/field0.f{str(i).zfill(5)}" for i in range(1601, 1601+nsnapshots)]
field_interpolation_dictionary['fields_to_interpolate'] = ["all"]
interpolated_fields_output_fname = "interpolated_fields.hdf5"
interpolation_settings = {"find_points_max_iter" : 50, "find_points_iterative" : [True, 10]}
interpolate_fields_from_disk(comm, query_points_fname, [sem_mesh_fname, sem_dtype], field_interpolation_dictionary, interpolated_fields_output_fname=interpolated_fields_output_fname, **interpolation_settings)

# =========================
# Perform the POD
# =========================
file_sequence = [f"./interpolated_fields{str(1+i).zfill(5)}.hdf5" for i in range(0, nsnapshots)]
pod_fields = ["u", "v", "w"]
mesh_fname = "points.hdf5"
mass_matrix_fname = "points.hdf5"
mass_matrix_key = "mass"
k = len(file_sequence) # This is the number of modes to be kept / updated
p = len(file_sequence) # This is the batch size for updates, i.e., how mamny snapshots you use at the time. P = nsnapshots, then you just do normal svd, not updating one.
fft_axis = 2 # Since we are doing fft over the z axis.
distributed_axis = 1 # So we are distributing y planes along ranks

# Import the pysemtools routines
from pysemtools.rom.fft_pod_wrappers import pod_fourier_1_homogenous_direction, physical_space
from pysemtools.io.wrappers import read_data
pod, ioh, _3d_bm_shape, number_of_frequencies, N_samples = pod_fourier_1_homogenous_direction(comm, file_sequence, pod_fields, mass_matrix_fname, mass_matrix_key, k, p, fft_axis, distributed_axis=distributed_axis)

# =========================
# Write the data
# =========================
from pysemtools.rom.fft_pod_wrappers import write_3dfield_to_file
from pyevtk.hl import gridToVTK

out_wavenumbers = 10
out_modes = 1

msh_data = read_data(comm, fname=mesh_fname, keys=["x", "y", "z"], parallel_io=True, distributed_axis=distributed_axis)
x = msh_data["x"]
y = msh_data["y"]
z = msh_data["z"]

# Write out 5 modes for the first 3 wavenumbers
write_3dfield_to_file("pod.hdf5", x, y, z, pod, ioh, wavenumbers=[k for k in range(0, out_wavenumbers)], modes=[i for i in range(0,out_modes)], field_shape=_3d_bm_shape, fft_axis=fft_axis, field_names=pod_fields, N_samples=N_samples, distributed_axis=distributed_axis,comm = comm)
comm.Barrier()

# To visualize with VTK, I only know how to do it in one rank currently
if comm.Get_rank() == 0:

    msh_data = read_data(comm, fname=mesh_fname, keys=["x", "y", "z"], parallel_io=False,distributed_axis=distributed_axis)
    x = msh_data["x"]
    y = msh_data["y"]
    z = msh_data["z"]

    for wavenumbers in range(0,out_wavenumbers):
        for modes in range(0, out_modes):

            data = read_data(comm, fname=f"pod_kappa_{wavenumbers}_mode{modes}.hdf5", keys=["u", "v", "w"], parallel_io=False, distributed_axis=distributed_axis)

            # write to vtk
            gridToVTK( "mode_0_wavenumber_"+str(wavenumbers).zfill(5),  x, y, z, pointData=data)