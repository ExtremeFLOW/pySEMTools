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

def tensor_pre(x, y, z, rotation_function, rotation_matrix, cutoff_index, field_data, info):
    '''
    This function applies a rotation to the velocity and also mirrors the velocity and temeprature fields based on symetries.
    It is important to note that in this case, it is assumed that the fields have already been mirrored spatially through interpolation, thus only the values are changed.
    That has not been included here.
    '''

    # First rotate the u, v, w components
    info["logger"].write("info", "Rotating the velocity components")
    tensor = [field_data["u"], field_data["v"], field_data["w"]]   
    rotated_tensor = rotation_function(x, y, z, tensor, rotation_matrix)
    field_data["u"] = rotated_tensor[0].copy()
    field_data["v"] = rotated_tensor[1].copy()
    field_data["w"] = rotated_tensor[2].copy()

    # Now mirror the necesary components if the snapshot index is appropiate
    if info["fidx"] >= cutoff_index: # Since it starts at 0, the mirrored snapshots start at cutoff_index
        info["logger"].write("info", f"Snapshot {info['fidx']} is mirrored, applying mirroring operations")
        info["logger"].write("info", "Mirroring the v (azimuthal) and w (vertical) velocity components")
        field_data["v"] = (-field_data["v"]).copy()
        field_data["w"] = (-field_data["w"]).copy()

        info["logger"].write("info", "Mirroring the temperature")
        t = field_data["t"] * 2 # Now temp is in range [0, 2]
        t +=  -1 # Now temp is in range [-1,   1]
        t *=  -1 # Now temp is mirrored [ 1 , -1]
        t +=   1 # Now temp is in range [ 2 ,  0]
        field_data["t"] = (t / 2).copy()

    return field_data

# Get mpi info
comm = MPI.COMM_WORLD

# Hide the log for the notebook. Not recommended when running in clusters as it is better you see what happens
import os
os.environ["PYSEMTOOLS_DEBUG"] = 'false'
os.environ["PYSEMTOOLS_HIDE_LOG"] = 'false'

# =========================
# Perform the POD
# =========================
nsnapshots = 10
file_sequence = [f"../../1-interpolation/2-interpolation_file_sequence/interpolated_fields{str(1+i).zfill(5)}.hdf5" for i in range(0, nsnapshots)]
pod_fields = ["u", "v", "w", "t"]
preproc = [v_pre, v_pre, v_pre, t_pre]
postproc = [v_post, v_post, v_post, None]
mesh_fname = "../../1-interpolation/1-structured_mesh_generation/points.hdf5"
mass_matrix_fname = "../../1-interpolation/1-structured_mesh_generation/points.hdf5"
mass_matrix_key = "mass"
k = len(file_sequence) # This is the number of modes to be kept / updated
p = k # This is the number of modes to be kept / updated
fft_axis = 1 
distributed_axis = 0
verify_reconstruction = False

# Set up the preprocessing tensor operation
from pysemtools.io.wrappers import read_data
from pysemtools.postprocessing.rotations import rotate_rank1_tensor, cartesian_to_cylindrical_rotation_matrix
msh_data = read_data(comm, fname=mesh_fname, keys=["x", "y", "z"], parallel_io=True, distributed_axis=distributed_axis)
rotation_function = rotate_rank1_tensor
rotation_matrix = cartesian_to_cylindrical_rotation_matrix
cutoff_index = len(file_sequence) # The index of the snapshot where the mirroring starts
preproc_t = lambda field_data, info: tensor_pre(msh_data["x"], msh_data["y"], msh_data["z"], rotation_function, rotation_matrix, cutoff_index, field_data, info)
if (cutoff_index - nsnapshots) <= 0:
    preproc_t = None
# Import the pysemtools routines
from pysemtools.rom.fft_pod_wrappers import pod_fourier_1_homogenous_direction, physical_space, extended_pod_1_homogenous_direction, save_pod_state
pod, ioh, _3d_bm_shape, number_of_frequencies, N_samples = pod_fourier_1_homogenous_direction(comm, file_sequence, pod_fields, mass_matrix_fname, mass_matrix_key, k, p, fft_axis,
                                                                                            distributed_axis=distributed_axis, preprocessing_field_operation=preproc, postprocessing_field_operation=postproc,
                                                                                            preprocessing_tensor_operation=preproc_t)

# Make sure that the mirrored time coefficients are not included in the POD
for kappa in pod.keys():
    try:
        int(kappa)
    except:
        continue  # Skip non-integer keys
    pod[kappa].vt_1t = pod[kappa].vt_1t[:, :nsnapshots]

# =========================
# Perform extended POD
# =========================
file_sequence = file_sequence
extended_pod_fields = ["t", "p"]
extended_pod_1_homogenous_direction(comm, file_sequence, extended_pod_fields, mass_matrix_fname, mass_matrix_key, fft_axis, distributed_axis=distributed_axis, pod=pod, ioh=ioh)
# Define the output field names
pod_fields = ["u", "v", "w", "sqrt(cv*t)", "t" , "p"]

# =========================
# Verify reconstruction
# =========================
if verify_reconstruction:

    verification_fields_file = ["u", "v", "w", "t", "p"]
    verification_fields_pod = ["u", "v", "w", "t" , "p"]

    for i, file in enumerate(file_sequence):

        fld = read_data(comm, file, keys=verification_fields_file, parallel_io=True, distributed_axis=distributed_axis)
        phys = physical_space(pod, ioh, wavenumbers=[k for k in range(0, number_of_frequencies)], modes=[i for i in range(0, k)], field_shape=_3d_bm_shape, fft_axis=fft_axis, field_names=pod_fields, N_samples=N_samples, snapshots=[i])

        all_passed = []
        for j, field in enumerate(verification_fields_file):

            a = phys[i][verification_fields_pod[j]]
            b = fld[verification_fields_file[j]]

            passed = np.allclose(a,b, rtol=1e-5, atol=1e-7)
            all_passed.append(passed)
            if not passed:
                print(f"Reconstruction failed for snapshot {i} and field {field}")  
                break
        passed = np.all(all_passed)

        if comm.Get_rank() == 0:
            print(f"Reconstruction for snapshot {i} {'passed' if passed else 'failed'}")


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
    write_3dfield_to_file("pod.hdf5", x, y, z, pod, ioh, wavenumbers=[k], modes=[mode], field_shape=_3d_bm_shape, fft_axis=fft_axis, field_names=pod_fields, N_samples=N_samples, distributed_axis=distributed_axis,comm = comm)

if comm.Get_rank() == 0:
    import json
    with open("mode_index.json", "w") as f:
        json.dump(mode_index_json, f, indent=4)
    print("Wrote mode index to file")

    # Write the data to hdf5
    f = h5py.File("mode_index.hdf5", "w")
    for key in mode_index:
        group = f.create_group(f"{key}")
        add_settings_to_hdf5(group, mode_index[key])
    f.close()

comm.Barrier()

# To visualize with VTK, I only know how to do it in one rank currently
if comm.Get_rank() == 0:

    msh_data = read_data(comm, fname=mesh_fname, keys=["x", "y", "z"], parallel_io=False,distributed_axis=distributed_axis)
    x = msh_data["x"]
    y = msh_data["y"]
    z = msh_data["z"]

    for i in range(0, out_modes):
        wavenumbers = mode_index[i]["wavenumber"]
        modes = mode_index[i]["mode"]

        data = read_data(comm, fname=f"pod_kappa_{wavenumbers}_mode{modes}.hdf5", keys=pod_fields, parallel_io=False, distributed_axis=distributed_axis)

        # write to vtk
        print(f"Writing mode {i} to vtk")
        gridToVTK( "energetic_mode"+str(i).zfill(5),  x, y, z, pointData=data)

# =========================
# Save the POD state
# =========================
if comm.Get_rank() == 0: print("Saving the POD state to file")
save_pod_state(comm, "pod_state.hdf5", pod, ioh, pod_fields, fft_axis, N_samples, number_of_frequencies, parallel_io=True, distributed_axis=distributed_axis)


# =========================
# Load the POD state and verify
# =========================
comm.Barrier()
from pysemtools.rom.fft_pod_wrappers import load_pod_state
if 1 == 0:
    pod_r, ioh_r, settings_r = load_pod_state(comm, "pod_state.hdf5", parallel_io=True, distributed_axis=distributed_axis)
    for kappa in pod.keys():
        try:
            int(kappa)
        except:
            continue  # Skip non-integer keys

        pas = np.allclose(pod_r[kappa].u_1t, pod[kappa].u_1t, rtol=1e-5, atol=1e-7) 
        print(pas)

        pas = np.allclose(pod_r[kappa].vt_1t, pod[kappa].vt_1t, rtol=1e-5, atol=1e-7)
        print(pas)

        pas = np.allclose(pod_r[kappa].d_1t, pod[kappa].d_1t, rtol=1e-5, atol=1e-7)
        print(pas)
