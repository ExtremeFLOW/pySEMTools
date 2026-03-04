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

file_sequence = [f"/home/adperez/software/nov04_pyNekTools/examples/4-interpolation/interpolated_fields{str(1+i).zfill(5)}.hdf5" for i in range(0, 48)]
pod_fields = ["u", "v", "w"]
mesh_fname = "/home/adperez/software/nov04_pyNekTools/examples/4-interpolation/points.hdf5"
mass_matrix_fname = "/home/adperez/software/nov04_pyNekTools/examples/4-interpolation/points.hdf5"
mass_matrix_key = "mass"
k = len(file_sequence)
p = len(file_sequence)
fft_axis = 1 # 0 for x, 1 for y, 2 for z (Depends on how the mesh was created)
distributed_axis = 2
from pyevtk.hl import gridToVTK


# Import the pysemtools routines
from pysemtools.rom.fft_pod_wrappers import pod_fourier_1_homogenous_direction, physical_space
from pysemtools.io.wrappers import read_data
from pysemtools.rom.fft_pod_wrappers import write_3dfield_to_file

# Perform the POD with your input data
pod, ioh, _3d_bm_shape, number_of_frequencies, N_samples = pod_fourier_1_homogenous_direction(comm, file_sequence, pod_fields, mass_matrix_fname, mass_matrix_key, k, p, fft_axis, distributed_axis=distributed_axis)


msh_data = read_data(comm, fname=mesh_fname, keys=["x", "y", "z"], parallel_io=True, distributed_axis=distributed_axis)
x = msh_data["x"]
y = msh_data["y"]
z = msh_data["z"]

# Write out 5 modes for the first 3 wavenumbers
write_3dfield_to_file("pod.hdf5", x, y, z, pod, ioh, wavenumbers=[k for k in range(0, 3)], modes=[i for i in range(0,1)], field_shape=_3d_bm_shape, fft_axis=fft_axis, field_names=pod_fields, N_samples=N_samples, distributed_axis=distributed_axis,comm = comm)

comm.Barrier()

# To visualize with VTK, I only know how to do it in one rank currently
if comm.Get_rank() == 0:

    msh_data = read_data(comm, fname=mesh_fname, keys=["x", "y", "z"], parallel_io=False,distributed_axis=distributed_axis)
    x = msh_data["x"]
    y = msh_data["y"]
    z = msh_data["z"]

    for wavenumbers in range(0, 3):
        for modes in range(0, 1):

            data = read_data(comm, fname=f"pod_kappa_{wavenumbers}_mode{modes}.hdf5", keys=["u", "v", "w"], parallel_io=False, distributed_axis=distributed_axis)

            # write to vtk
            gridToVTK( "mode_0_wavenumber_"+str(wavenumbers).zfill(5),  x, y, z, pointData=data)