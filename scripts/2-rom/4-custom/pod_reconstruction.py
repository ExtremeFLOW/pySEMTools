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
import json

def add_settings_to_hdf5(h5group, settings_dict):
    """
    Recursively adds the key/value pairs from a settings dictionary to an HDF5 group.
    Dictionary values that are themselves dictionaries are added as subgroups;
    other values are stored as attributes.
    """
    for key, value in settings_dict.items():
        if isinstance(value, dict):
            subgroup = h5group.create_group(key)
            add_settings_to_hdf5(subgroup, value)
        else:
            h5group.attrs[key] = value

def load_hdf5_settings(group):
    """
    Recursively loads an HDF5 group into a dictionary.
    Attributes become key/value pairs and subgroups are loaded recursively.
    """
    settings = {}
    # Load attributes
    for key, value in group.attrs.items():
        settings[key] = value
    # Recursively load subgroups
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            settings[key] = load_hdf5_settings(item)
    return settings

def v_pre(field):
    return field * np.sqrt(1/2)

def v_post(field):
    return field / np.sqrt(1/2)

def t_pre(field):
    cp = 1
    gamma = 1.4
    cv = cp / gamma
    field[field < 0] = 0.0
    return np.sqrt(cv * field)

# Get mpi info
comm = MPI.COMM_WORLD

# Hide the log for the notebook. Not recommended when running in clusters as it is better you see what happens
import os
os.environ["PYSEMTOOLS_DEBUG"] = 'false'
os.environ["PYSEMTOOLS_HIDE_LOG"] = 'false'

# Import the pysemtools routines
from pysemtools.rom.fft_pod_wrappers import pod_fourier_1_homogenous_direction, physical_space, extended_pod_1_homogenous_direction, save_pod_state
from pysemtools.io.wrappers import read_data
from pysemtools.rom.fft_pod_wrappers import load_pod_state

distributed_axis = 0
mesh_fname = "../../1-interpolation/1-structured_mesh_generation/points.hdf5"

# =========================
# load pod
# =========================

pod, ioh, settings = load_pod_state(comm, "pod_state.hdf5", parallel_io=True, distributed_axis = distributed_axis)
number_of_frequencies = settings["number_of_frequencies"]
N_samples = settings["N_samples"]
pod_fields = settings["pod_fields"]
fft_axis = settings["fft_axis"]

# =========================
# Sort energetic modes
# =========================
singular_values = []
wavenumber = []
mode = []
for k in range(0, number_of_frequencies):
    nmodes = pod[k].d_1t.shape[0]
    for i in range(0, nmodes):
        singular_values.append(pod[k].d_1t[i])
        wavenumber.append(k)
        mode.append(i)

singular_values = np.array(singular_values)
all_wavenumbers = np.array(wavenumber)
all_modes = np.array(mode)
indices = np.argsort(singular_values)[::-1]

# =========================
# Write the data relevant data
# =========================
from pysemtools.rom.fft_pod_wrappers import write_3dfield_to_file
from pyevtk.hl import gridToVTK

out_modes = 20
msh_data = read_data(comm, fname=mesh_fname, keys=["x", "y", "z"], parallel_io=True, distributed_axis=distributed_axis)
x = msh_data["x"]
y = msh_data["y"]
z = msh_data["z"]

# Write out the modes that are requested
mode_index = {}
mode_index_json = {}
for i in range(0, out_modes):
    k = all_wavenumbers[indices[i]]
    mode = all_modes[indices[i]]

    # To be written to json
    mode_index_json[i] = {}
    mode_index_json[i]["wavenumber"] = str(k)
    mode_index_json[i]["mode"] = str(mode)
    mode_index_json[i]["singular_value"] = str(singular_values[indices[i]])

    # To be written to hdf5
    mode_index[i] = {}
    mode_index[i]["wavenumber"] = k
    mode_index[i]["mode"] = mode
    mode_index[i]["singular_value"] = singular_values[indices[i]]
    mode_index[i]["right_singular_vector_real"] = pod[k].vt_1t[mode, :].real
    mode_index[i]["right_singular_vector_imag"] = pod[k].vt_1t[mode, :].imag
    write_3dfield_to_file("pod.hdf5", x, y, z, pod, ioh, wavenumbers=[k], modes=[mode], field_shape= x.shape, fft_axis=fft_axis, field_names=pod_fields, N_samples=N_samples, distributed_axis=distributed_axis,comm = comm)


if comm.Get_size() > 1:
    print("to write to vtk you need to do it in serial")
    sys.exit(0)


nsnapshots = 400
wavenumber_rec = []
mode_rec = {}
for key in [0, 1, 2, 3]:
    kappa = mode_index[key]["wavenumber"]
    if kappa not in wavenumber_rec:
        wavenumber_rec.append(kappa)
        mode_rec[kappa] = [mode_index[key]["mode"]]
    else:
        mode_rec[kappa].append(mode_index[key]["mode"])

for i in range(0, nsnapshots):
    
    phys = physical_space(pod, ioh, wavenumbers=wavenumber_rec, modes=mode_rec, field_shape=x.shape, fft_axis=fft_axis, field_names=pod_fields, N_samples=N_samples, snapshots=[i])

    print(f"Writing snapshot {i} of {nsnapshots} to vtk") 
    gridToVTK( f"energetic_modes_snapshot_{str(i).zfill(5)}",  x, y, z, pointData=phys[i])

