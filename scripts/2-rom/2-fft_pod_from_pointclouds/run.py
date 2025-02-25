# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import h5py
import json
import os

# Get mpi info
comm = MPI.COMM_WORLD

# Import the pysemtools routines
from pysemtools.rom.fft_pod_wrappers import pod_fourier_1_homogenous_direction, physical_space
from pysemtools.rom.fft_pod_wrappers import write_3dfield_to_file
from pysemtools.rom.fft_pod_wrappers import save_pod_state

if comm.Get_size() > 1:
    raise ValueError("This script is not designed to run in parallel (You can stream), since the fft is done in serial. In the future we can do this in parallel by doing it on planes of the 3D field.")

def main():

    # Read inputs
    with open("inputs.json", "r") as f:
        inputs = json.load(f)

    # Fix the file sequence
    path = os.path.dirname(inputs["file_sequence"]["fname"])
    if path == "": path = "."
    prefix = os.path.basename(inputs["file_sequence"]["fname"]).split(".")[0]
    extension = os.path.basename(inputs["file_sequence"]["fname"]).split(".")[1]
    min_index = inputs["file_sequence"]["first_index"]
    max_index = inputs["file_sequence"]["first_index"] + inputs["file_sequence"]["number_of_files"]
    file_sequence = []
    for i in range(min_index, max_index):
        suffix = str(i).zfill(inputs["file_sequence"]["fill_name_with_zeros"])
        file_sequence.append(f"{path}/{prefix}{suffix}.{extension}")

    pod_fields = inputs["pod_fields"]
    mass_matrix_fname = inputs["mass_matrix"]["fname"]
    mass_matrix_key = inputs["mass_matrix"]["key"]
    k = inputs["pod_modes_to_update"]
    p = inputs["batch_size"]
    fft_axis = inputs["fft_axis"]

    # Perform the POD with your input data
    pod, ioh, _3d_bm_shape, number_of_frequencies, N_samples = pod_fourier_1_homogenous_direction(comm, file_sequence, pod_fields, mass_matrix_fname, mass_matrix_key, k, p, fft_axis)

    # Save outputs
    mesh_fname = inputs["mass_matrix"]["fname"]
    # Load the mesh
    with h5py.File(mesh_fname, 'r') as f:
        x = f["x"][:]
        y = f["y"][:]
        z = f["z"][:]

    if inputs["outputs"]["write_modes"]["write"]:
    
        if inputs["outputs"]["write_modes"]["wavenumbers"] == "all":
            wavenumbers = [k for k in range(0, number_of_frequencies)]
        else:
            wavenumbers = inputs["outputs"]["write_modes"]["wavenumbers"]

        if inputs["outputs"]["write_modes"]["modes"] == "all":
            modes = [i for i in range(0, k)]
        else:
            modes = inputs["outputs"]["write_modes"]["modes"]
    
        write_3dfield_to_file(inputs["outputs"]["prefix"], x, y, z, pod, ioh, wavenumbers=wavenumbers, modes=modes, field_shape=_3d_bm_shape, fft_axis=fft_axis, field_names=pod_fields, N_samples=N_samples)
    
    if inputs["outputs"]["write_reconstruction"]["write"]:
    
        if inputs["outputs"]["write_reconstruction"]["wavenumbers"] == "all":
            wavenumbers = [k for k in range(0, number_of_frequencies)]
        else:
            wavenumbers = inputs["outputs"]["write_reconstruction"]["wavenumbers"]

        if inputs["outputs"]["write_reconstruction"]["modes"] == "all":
            modes = [i for i in range(0, k)]
        else:
            modes = inputs["outputs"]["write_reconstruction"]["modes"]

        if inputs["outputs"]["write_reconstruction"]["snapshots"] == "all":
            snapshots = [i for i in range(0, len(file_sequence))]
        else:
            snapshots = inputs["outputs"]["write_reconstruction"]["snapshots"]
        
        write_3dfield_to_file(inputs["outputs"]["prefix"], x, y, z, pod, ioh, wavenumbers=wavenumbers, modes=modes, field_shape=_3d_bm_shape, fft_axis=fft_axis, field_names=pod_fields, N_samples=N_samples, snapshots=snapshots)

    if inputs["outputs"]["write_state"]["write"]: 

        save_pod_state(inputs["outputs"]["write_state"]["name"], pod)

if __name__ == "__main__":
    main()