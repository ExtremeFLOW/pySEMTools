# Import general modules
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import matplotlib.pyplot as plt

def _mirror_right(a, axis=0):
    """
    Mirror 'a' about its last sample along `axis` (one-sided).
    """
    a = np.asarray(a)
    N = a.shape[axis]
    if N < 2:
        raise ValueError("Need at least 2 samples along axis to mirror.")
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(N-2, 0, -1)  # N-2, ..., 1
    a_m = a[tuple(sl)]
    return np.concatenate([a, a_m], axis=axis)

def _hilbert_fft(a, axis=0):
    """
    Analytic signal via FFT along `axis`.
    """
    a = np.asarray(a)
    M = a.shape[axis]
    A = np.fft.fft(a, axis=axis)

    # Build the kernel in fourier space
    h = np.zeros(M, dtype=float)
    if M % 2 == 0:
        h[0] = 1.0
        h[M//2] = 1.0
        h[1:M//2] = 2.0
    else:
        h[0] = 1.0
        h[1:(M+1)//2] = 2.0

    shape = [1] * a.ndim
    shape[axis] = M
    H = h.reshape(shape)

    # Perform the convolution (multiplication in fourier space)
    z = np.fft.ifft(A * H, axis=axis)
    return z

def hilbert_axis0_mirror_right(a, axis=0, return_analytic=False):
    """
    1) Extend by mirroring around the last point (right side)
    2) Hilbert (via FFT) along `axis`
    3) Return only the first N samples (original length)

    If return_analytic=False: returns Hilbert transform (real, same dtype as input float)
    If return_analytic=True: returns analytic signal (complex)
    """
    a = np.asarray(a)
    N = a.shape[axis]

    a_ext = _mirror_right(a, axis=axis)      # length 2N-2
    z_ext = _hilbert_fft(a_ext, axis=axis)   # analytic signal on extended

    # slice back to original length
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(0, N)
    z = z_ext[tuple(sl)]

    if return_analytic:
        return z
    return np.imag(z)

if __name__ == "__main__":

    # Import MPI
    from mpi4py import MPI #equivalent to the use of MPI_init() in C
    # Import IO helper functions
    from pysemtools.io.utils import IoPathData, get_fld_from_ndarray
    # Import modules for reading and writing
    from pysemtools.io.ppymech import pynekread, pynekwrite
    # Data types
    from pysemtools.datatypes import FieldRegistry, Mesh, Coef
    # POD
    from pysemtools.rom import POD, IoHelp
    # Interpolation
    from pysemtools.interpolation import Probes
    from pysemtools.io.wrappers import read_data, write_data

    # Split communicator for MPI - MPMD
    worldcomm = MPI.COMM_WORLD
    worldrank = worldcomm.Get_rank()
    worldsize = worldcomm.Get_size()
    col = 1
    comm = worldcomm.Split(col,worldrank)
    rank = comm.Get_rank()
    size = comm.Get_size()

    #=========================================
    # Define inputs
    #=========================================

    # Open input file to see path
    f = open ("inputs.json", "r")
    params_file = json.loads(f.read())
    f.close()

    # Read the POD inputs
    pod_number_of_snapshots = params_file["number_of_snapshots"]
    pod_fields = params_file["fields"]
    number_of_pod_fields = len(pod_fields)
    pod_batch_size = params_file["batch_size"]
    pod_keep_modes = params_file["keep_modes"]
    pod_write_modes = params_file["write_modes"]
    hpod_flag = params_file["hpod"]
    dtype_string = params_file.get("dtype", "double")
    print(dtype_string)
    backend = params_file.get("backend", "numpy")
    if dtype_string == "single":
        dtype = np.float32
    else:
        dtype = np.float64
    if not hpod_flag:
        pod_dtype = dtype
    else:
        pod_dtype = np.complex128 if dtype == np.float64 else np.complex64

    # Read the interpolation inputs
    query_points_fname = params_file["interpolation"]["query_points_fname"]
    interpolated_fields_output_fname = params_file["interpolation"]["output_sequence_fname"]
    distributed_axis = params_file["interpolation"]["distributed_axis"]
    parallel_io = params_file["interpolation"]["parallel_io"]
    interpolation_settings = params_file["interpolation"].get('interpolation_settings', {})
    interpolation_settings['point_interpolator_type'] = interpolation_settings.get('point_interpolator_type', 'multiple_point_legendre_numpy')

    # Start time
    start_time = MPI.Wtime()

    #=========================================
    # Get the mesh
    #=========================================

    # Read the data paths from the input file
    mesh_data = IoPathData(params_file["IO"]["mesh_data"])
    field_data = IoPathData(params_file["IO"]["field_data"])

    # Initialize the mesh file
    path     = mesh_data.dataPath
    casename = mesh_data.casename
    index    = mesh_data.index
    fname    = path+casename+'0.f'+str(index).zfill(5)
    msh = Mesh(comm)
    pynekread(fname, comm, data_dtype=dtype, msh = msh)

    # Read the meshgrid data
    print("Reading query points from: ", query_points_fname)
    meshgrid = read_data(comm, fname=query_points_fname, keys=["x", "y", "z", "mass"], parallel_io=parallel_io, distributed_axis=distributed_axis, dtype=dtype)

    #=========================================
    # Initialize the interpolation
    #=========================================

    xyz = [meshgrid["x"].flatten(), meshgrid["y"].flatten() , meshgrid["z"].flatten() ]
    xyz = np.ascontiguousarray(np.array(xyz).T)

    # Initialize the probe object
    probes = Probes(comm, probes=xyz, msh = msh, output_fname=interpolated_fields_output_fname, **interpolation_settings)

    #=========================================
    # Initialize the POD
    #=========================================

    # Instance the POD object
    pod = POD(comm, number_of_modes_to_update = pod_keep_modes, global_updates = True, auto_expand = False, bckend = backend)

    # Initialize coef to get the mass matrix
    bm = meshgrid["mass"]

    # Instance io helper that will serve as buffer for the snapshots
    ioh = IoHelp(comm, number_of_fields = number_of_pod_fields, batch_size = pod_batch_size, field_size = bm.size, field_data_type=pod_dtype, mass_matrix_data_type=pod_dtype)

    # Put the mass matrix in the appropiate format (long 1d array)
    mass_list = []
    for i in range(0, number_of_pod_fields):
        if not hpod_flag:
            mass_list.append(np.copy(np.sqrt(bm)))
        else:
            mass_list.append(np.sqrt(bm)+np.sqrt(bm)*0j)
    ioh.copy_fieldlist_to_xi(mass_list)
    ioh.bm1sqrt[:,:] = np.copy(ioh.xi[:,:])

    #=========================================
    # Perform the streaming of data
    #=========================================

    j = 0
    while j < pod_number_of_snapshots:

        # Recieve the data from fortran
        path     = field_data.dataPath
        casename = field_data.casename
        index    = field_data.index
        fname=path+casename+'0.f'+str(index + j).zfill(5)
        fld = FieldRegistry(comm)
        pynekread(fname, comm, data_dtype=dtype, fld = fld) 

        # Interpolate
        field_list = [fld.registry[key] for key in pod_fields]
        field_names = pod_fields
        probes.interpolate_from_field_list(fld.t, field_list, comm, write_data=False, field_names=field_names)

        # Reshape the data
        output_data = {}
        for i, key in enumerate(field_names):
            output_data[key] = probes.interpolated_fields[:, i+1].reshape(bm.shape).astype(dtype)
        print(output_data['u'].shape, output_data['v'].dtype) 

        output_data_h = {}
        if hpod_flag:
            # Hilber transform
            for i, key in enumerate(field_names):
                output_data_h[key] = hilbert_axis0_mirror_right(output_data[key], axis=0, return_analytic=True).astype(pod_dtype)
            output_data = output_data_h
            print(output_data['u'].shape, output_data['v'].dtype) 

        # Put the snapshot data into a column array
        ioh.copy_fieldlist_to_xi([output_data[key] for key in field_names])

        # Load the column array into the buffer
        ioh.load_buffer(scale_snapshot = True)

        # Update POD modes
        if ioh.update_from_buffer:
            pod.update(comm, buff = ioh.buff[:,:(ioh.buffer_index)])

        j += 1

    #=========================================
    # Perform post-stream operations
    #=========================================

    # Check if there is information in the buffer that should be taken in case the loop exit without flushing
    if ioh.buffer_index > ioh.buffer_max_index:
        ioh.log.write("info","All snapshots where properly included in the updates")
    else: 
        ioh.log.write("warning","Last loaded snapshot to buffer was: "+repr(ioh.buffer_index-1))
        ioh.log.write("warning","The buffer updates when it is full to position: "+repr(ioh.buffer_max_index))
        ioh.log.write("warning","Data must be updated now to not lose anything,  Performing an update with data in buffer ")
        pod.update(comm, buff = ioh.buff[:,:(ioh.buffer_index)])

    # Scale back the modes
    pod.scale_modes(comm, bm1sqrt = ioh.bm1sqrt, op = "div")

    # Rotate local modes back to global, This only enters in effect if global_update = false
    pod.rotate_local_modes_to_global(comm)

    #=========================================
    # Write out the modes
    #=========================================

    # Write the data out
    output_index = 1
    for j in range(0, pod_write_modes):

        if (j+1) < pod.u_1t.shape[1]:

            ## Split the snapshots into the proper fields
            output_data = {}
            field_list1d = ioh.split_narray_to_1dfields(pod.u_1t[:,j])
            output_data = {key: field.reshape(bm.shape) for key, field in zip(field_names, field_list1d)}
            print(output_data['u'].shape, output_data['u'].dtype)

            if not hpod_flag:
                #write the data
                prefix = interpolated_fields_output_fname.split(".")[0]
                suffix = interpolated_fields_output_fname.split(".")[1]
                out_fname = f"{prefix}{str(output_index).zfill(5)}.{suffix}"
                write_data(comm, fname=out_fname, data_dict = output_data, parallel_io=parallel_io, distributed_axis=distributed_axis, msh = [meshgrid[key] for key in ["x", "y", "z"]], write_mesh = True, uniform_shape = True)
                output_index += 1
                # Clear the fields
            else:
                #write the data

                # Real part
                prefix = interpolated_fields_output_fname.split(".")[0]
                suffix = interpolated_fields_output_fname.split(".")[1]
                out_fname = f"{prefix}{str(output_index).zfill(5)}.{suffix}"
                _output_data = {key: np.real(output_data[key]) for key in field_names}
                write_data(comm, fname=out_fname, data_dict = _output_data, parallel_io=parallel_io, distributed_axis=distributed_axis, msh = [meshgrid[key] for key in ["x", "y", "z"]], write_mesh = True, uniform_shape = True)
                output_index += 1
                
                # Imaginary part
                prefix = interpolated_fields_output_fname.split(".")[0]
                suffix = interpolated_fields_output_fname.split(".")[1]
                out_fname = f"{prefix}{str(output_index).zfill(5)}.{suffix}"
                _output_data = {key: np.imag(output_data[key]) for key in field_names}
                write_data(comm, fname=out_fname, data_dict = _output_data, parallel_io=parallel_io, distributed_axis=distributed_axis, msh = [meshgrid[key] for key in ["x", "y", "z"]], write_mesh = True, uniform_shape = True)
                output_index += 1
            
            fld.clear()

    #=========================================
    # Write out singular values and right 
    # singular vectors
    #=========================================

    # Write the singular values and vectors
    if comm.Get_rank() == 0:
        np.save("singular_values", pod.d_1t)
        print("Wrote signular values")
        np.save("right_singular_vectors", pod.vt_1t)
        print("Wrote right signular values")

    # End time
    end_time = MPI.Wtime()
    # Print the time
    if comm.Get_rank() == 0:
        print("Time to complete: ", end_time - start_time)
