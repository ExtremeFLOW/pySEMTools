""" Contains functions to wrap the ROM types to easily post process data """

from .pod import POD
from typing import Union, Callable
from ..io.wrappers import read_data, write_data
from .io_help import IoHelp
import numpy as np
import h5py
import os
import json
from ..monitoring.logger import Logger
from pyevtk.hl import gridToVTK

def get_wavenumber_slice(kappa, fft_axis):
    """
    Get the correct slice of a 3d field that has experienced an fft in the fft_axis

    Parameters
    ----------
    kappa : int
        Wavenumber to get the slice for
    fft_axis : int
        Axis where the fft was performed
    """
    if fft_axis == 0:
        return (kappa, slice(None), slice(None))
    elif fft_axis == 1:
        return (slice(None), kappa, slice(None))
    elif fft_axis == 2:
        return (slice(None), slice(None), kappa)


def get_mass_slice(fft_axis):
    """
    Get the correct slice of a 3d field that will experience fft in the fft_axis

    This is particularly necessary for the mass matrix to be applied to the frequencies individually.

    Parameters
    ----------
    fft_axis : int
        Axis where the fft will be performed
    """

    # Have a slice of the axis to perform the fft
    if fft_axis == 0:
        mass_slice = (0, slice(None), slice(None))
    elif fft_axis == 1:
        mass_slice = (slice(None), 0, slice(None))
    elif fft_axis == 2:
        mass_slice = (slice(None), slice(None), 0)
    return mass_slice


def get_2d_slice_shape(fft_axis, field_shape):
    """
    Get the shape of the 2d slice of a 3d field that has experienced an fft in the fft_axis

    Parameters
    ----------
    fft_axis : int
        Axis where the fft was performed
    field_shape : tuple
        Shape of the field in physical space
    """

    if fft_axis == 0:
        return (field_shape[1], field_shape[2])
    elif fft_axis == 1:
        return (field_shape[0], field_shape[2])
    elif fft_axis == 2:
        return (field_shape[0], field_shape[1])


def fourier_normalization(N_samples):
    """
    Get the value that will be used to normalize the fourier coefficientds after fft

    Parameters
    ----------
    N_samples : int
        Number of samples used to get the fft
    """
    return np.sqrt(N_samples)


def degenerate_scaling(kappa):
    """
    Get the scaling factor for the degenerate wavenumbers.

    This alludes to wavebnumbers were we only calculate things once but that because symetries we have to multiply by 2 or more.

    Parameters
    ----------
    kappa : int
        Wavenumber to get the scaling for
    """

    if kappa == 0:
        scaling = 1
    else:
        scaling = 2
    return np.sqrt(scaling)


def physical_space(
    pod: dict[int, POD],
    ioh: dict[int, IoHelp],
    wavenumbers: list[int],
    modes: Union[list[int], dict[int, list[int]]],
    field_shape: tuple,
    fft_axis: int,
    field_names: list[str],
    N_samples: int,
    snapshots: list[int] = None,
):
    """
    Function to transform modes or snapshots from the POD objects into the physical space

    This will either produce a set of specified modes for specified wavenumbers in physical space.

    Or it will use the specified modes in the specified wavenumbers to reconstruct the specified snapshots in physical space.

    Parameters
    ----------
    pod : dict[int, POD]
        Dictionary of POD object with the modes to transform to physical space
        the int key is the wavenumber
    ioh : dict[int, IoHelp]
        Dictionary of IoHelp object, which has some functionalities to split fields
        the int key is the wavenumber
    wavenumbers : list[int]
        List of wavenumbers to use in the operations
    modes : int
        list of the modes to use in the operations.
        if snapshot is not given, the modes will be transformed to physical space and returned.
        if snapshot is given, the modes will be used to reconstruct the snapshots and returned.
    field_shape : tuple
        Shape of the field in physical space
    fft_axis : int
        Axis where the fft was performed
    field_names : list[str]
        List of field names to put in the output dictionary
    N_samples : int
        Number of samples in the fft
    snapshots : list[int], optional
        List of snapshots to transform to physical space, by default None
        If this option is given, then the return will be a list of snapshots in physical space
        using the snapshot indices for the reconstruction.
        Be mindfull that the snapshot indices should be in the range of the snapshots used to create the POD objects.

    Returns
    -------
    """

    # To reconstruct snapshots
    if isinstance(snapshots, list):

        physical_fields = {}

        # Reconstruct the fourier coefficients per wavenumber with the given snapshots and modes
        fourier_reconstruction = {}
        for kappa in wavenumbers:

            # If modes is a dict, we take the modes for the wavenumber
            if isinstance(modes, dict):
                modes_ = modes[kappa]
            else:
                modes_ = modes

            fourier_reconstruction[kappa] = (
                pod[kappa].u_1t[:, modes_].reshape(-1, len(modes_))
                @ np.diag(pod[kappa].d_1t[modes_])
                @ pod[kappa]
                .vt_1t[np.ix_(modes_, snapshots)]
                .reshape(len(modes_), len(snapshots))
            )

        # Go thorugh the wavenumbers in the list and put the modes in the physical space
        for snap_id, snapshot in enumerate(snapshots):

            physical_fields[snapshot] = {}

            # Create a buffer to zero out all the other wavenumber contributions
            fourier_field_3d = [
                np.zeros(field_shape, dtype=pod[0].u_1t.dtype)
                for i in range(0, len(field_names))
            ]

            # Fill the fourier fields with the contributions of the wavenumbers
            for kappa in wavenumbers:

                ## Split the 1d snapshot into a list with the fields you want
                field_list1d = ioh[kappa].split_narray_to_1dfields(
                    fourier_reconstruction[kappa][:, snap_id]
                )
                ## Reshape the obtained 1d fields to be 2d
                _2d_field_shape = get_2d_slice_shape(fft_axis, field_shape)
                field_list_2d = [
                    field.reshape(_2d_field_shape) for field in field_list1d
                ]

                # Get the proper data slice for positive and negative wavenumber
                positive_wavenumber_slice = get_wavenumber_slice(kappa, fft_axis)
                negative_wavenumber_slice = get_wavenumber_slice(-kappa, fft_axis)

                for i, field_name in enumerate(field_names):

                    # Fill the buffer with the proper wavenumber contribution
                    fourier_field_3d[i][positive_wavenumber_slice] = field_list_2d[i]
                    if kappa != 0:
                        fourier_field_3d[i][negative_wavenumber_slice] = np.conj(
                            field_list_2d[i]
                        )

            for i, field_name in enumerate(field_names):

                # Perform the inverse fft
                physical_field_3d = np.fft.ifft(
                    fourier_field_3d[i] * fourier_normalization(N_samples),
                    axis=fft_axis,
                )  # Rescale the coefficients

                # Save the field in the dictionary (only the real part)
                physical_fields[snapshot][field_name] = np.copy(physical_field_3d.real)

    # To obtain only the modes
    else:

        # Go thorugh the wavenumbers in the list and put the modes in the physical space
        physical_fields = {}
        for kappa in wavenumbers:

            # Create the physical space dictionary
            physical_fields[kappa] = {}

            for mode in modes:

                # Add the mode to the dictionary
                physical_fields[kappa][mode] = {}

                ## Split the 1d snapshot into a list with the fields you want
                field_list1d = ioh[kappa].split_narray_to_1dfields(
                    pod[kappa].u_1t[:, mode]
                )
                ## Reshape the obtained 1d fields to be 2d
                _2d_field_shape = get_2d_slice_shape(fft_axis, field_shape)
                field_list_2d = [
                    field.reshape(_2d_field_shape) for field in field_list1d
                ]

                # Get the proper data slice for positive and negative wavenumber
                positive_wavenumber_slice = get_wavenumber_slice(kappa, fft_axis)
                negative_wavenumber_slice = get_wavenumber_slice(-kappa, fft_axis)

                for i, field_name in enumerate(field_names):

                    # Create a buffer to zero out all the other wavenumber contributions
                    fourier_field_3d = np.zeros(
                        field_shape, dtype=pod[kappa].u_1t.dtype
                    )

                    # Fill the buffer with the proper wavenumber contribution
                    fourier_field_3d[positive_wavenumber_slice] = field_list_2d[i]
                    if kappa != 0:
                        fourier_field_3d[negative_wavenumber_slice] = np.conj(
                            field_list_2d[i]
                        )

                    # Perform the inverse fft
                    physical_field_3d = np.fft.ifft(
                        fourier_field_3d * fourier_normalization(N_samples),
                        axis=fft_axis,
                    )  # Rescale the coefficients

                    # Save the field in the dictionary (only the real part)
                    physical_fields[kappa][mode][field_name] = np.copy(
                        physical_field_3d.real
                    )

    return physical_fields

def extended_pod_1_homogenous_direction(
    comm,
    file_sequence: list[str],
    extended_pod_fields: list[str],
    mass_matrix_fname: str,
    mass_matrix_key: str,
    fft_axis: int,
    distributed_axis: int = None,
    pod: dict[int, POD] = None,
    ioh: dict[int, IoHelp] = None,
    ): 

    """
    Perform extended pod from given modes. The data is read from disk.
    """
    
    log = Logger(comm=comm, module_name="extended_pod")
    log.write("info", "Starting extended POD in homogenous direction")
    for field in extended_pod_fields:
        log.write("info", "Extended POD field: " + field)
 
    number_of_pod_fields = len(extended_pod_fields)
    
    # Read the mass matrix simply to set up sizes
    if distributed_axis is not None:
        parallel_io = True
    else:
        parallel_io = False
        distributed_axis = 0

    dat = read_data(comm, fname= mass_matrix_fname, keys=[mass_matrix_key], parallel_io=parallel_io, distributed_axis=distributed_axis)
    bm = dat[mass_matrix_key]
    bm[np.where(bm == 0)] = 1e-14
    field_3d_shape = bm.shape

    # Obtain the number of frequencies you will obtain
    N_samples = bm.shape[fft_axis]
    number_of_frequencies = N_samples // 2 + 1
    # Choose the proper mass matrix slice
    bm = bm[get_mass_slice(fft_axis)]

    extended_ioh = {"wavenumber": "buffers"}
    extended_modes = {"wavenumber": "mode_array"}

    # Initialize the buffers and objects for each wavenumber
    for kappa in range(0, number_of_frequencies):

        # Instance io helper that will serve as buffer for the snapshots
        extended_ioh[kappa] = IoHelp(
            comm,
            number_of_fields=number_of_pod_fields,
            batch_size=1,
            field_size=bm.size,
            mass_matrix_data_type=bm.dtype,
            field_data_type=np.complex128,
            module_name="extended_buffer_kappa" + str(kappa),
        )

        extended_modes[kappa] = np.zeros((bm.size*number_of_pod_fields, pod[kappa].u_1t.shape[1]), dtype=np.complex128)
    
    # ============
    # Main program
    # ============
    # Extended POD modes
    # ============

    j = 0
    while j < len(file_sequence):

        # Load the snapshot data
        fname = file_sequence[j]

        log.write("info", "Processing snapshot: " + fname)

        fld_data_ = read_data(comm, fname=fname, keys=extended_pod_fields, parallel_io=parallel_io, distributed_axis=distributed_axis)
        fld_data = [fld_data_[field] for field in extended_pod_fields]

        # Perform the fft
        for i in range(0, number_of_pod_fields):
            fld_data[i] = np.fft.fft(
                fld_data[i], axis=fft_axis
            ) / fourier_normalization(N_samples)

        # For each wavenumber, load buffers and update if needed
        for kappa in range(0, number_of_frequencies):

            # Get the proper slice for the wavenumber
            positive_wavenumber_slice = get_wavenumber_slice(kappa, fft_axis)

            # Get the wavenumber data
            wavenumber_data = []
            for i in range(0, number_of_pod_fields):
                wavenumber_data.append(
                    fld_data[i][positive_wavenumber_slice] * degenerate_scaling(kappa)
                )  # Here add contributions from negative wavenumbers

            # Put the fourier snapshot data into a column array
            extended_ioh[kappa].copy_fieldlist_to_xi(wavenumber_data)

            # Project the snapshot into the known right singular vectors
            extended_modes[kappa] = extended_modes[kappa] + extended_ioh[kappa].xi @ (np.conj(pod[kappa].vt_1t.T))[j, :].reshape(1, -1)

        j += 1
    
    # ============
    # Main program
    # ============
    # rscale modes
    # ============

    log.write("info", "Processing snapshots: Done")
    log.write("info", "Performing necesary rescalings")

    # Use the singular values
    for kappa in range(0, number_of_frequencies):
        extended_modes[kappa] /= pod[kappa].d_1t.reshape(1, -1)

        extended_modes[kappa] /= degenerate_scaling(kappa)

    # Extend the POD object with the data
    log.write("info", "Extending POD objects with the extended modes")
    for kappa in range(0, number_of_frequencies):
        pod[kappa].u_1t = np.concatenate((pod[kappa].u_1t, extended_modes[kappa]), axis=0)

    log.write("info", "Extended POD in homogenous direction done")

def pod_fourier_1_homogenous_direction(
    comm,
    file_sequence: list[str],
    pod_fields: list[str],
    mass_matrix_fname: str,
    mass_matrix_key: str,
    k: int,
    p: int,
    fft_axis: int,
    distributed_axis: int = None,
    preprocessing_field_operation: list[Callable] = None,
    postprocessing_field_operation: list[Callable] = None,
    preprocessing_tensor_operation: Callable = None,
) -> tuple:
    """
    Perform POD on a sequence of snapshot while applying fft in an homogenous direction of choice.

    This will not work on the spectral element mesh, therefore we have the mass matrix as a parameter.

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator
    file_sequence : list[str]
        List of file names containing the snapshots.
    pod_fields : list[str]
        List of fields to perform the POD on.
        They should currespond to the name of the fields in the input file.
        currently only hdf5 input file supported.
    mass_matrix_fname : str
        Name of the file containing the mass matrix.
        currently only hdf5 input file supported.
    mass_matrix_key : str
        Key of the mass matrix in the hdf5 file.
    k : int
        Number of modes to update.
        set to len(file_sequence) to update all modes.
    p : int
        Number of snapshots to load at once
        set to len(file_sequence) perform the process without updating.
    fft_axis : int
        Axis to perform the fft on.
        0 for x, 1 for y, 2 for z. (Although this depends on how the mesh was created)
    distributed_axis : int, optional
        Axis to distribute the data over.
        If None, the data will not be distributed.
    preprocessing_field_operation : list[Callable], optional
        List of functions to apply to the fields before the fft.
        Each function should take a field as input and return the processed field.
    postprocessing_field_operation : list[Callable], optional
        List of functions to apply to the modes after the fft.
        Each function should take a field as input and return the processed field.
    preprocessing_tensor_operation : Callable, optional
        Dictionary of functions and settings to apply to the fields before the fft.
        The inputs passed to this function should have the following:
        - `fld_data_`: The data expected to be read from the file. It is a dictionary.
        - `pod_info`: A dictionary that will have the info from the read snapshot and POD settings.
        This function can, thus, be used for anything that needs to be done before the fft,
        and it is applied just after reading the snapshot data.
        
    Returns
    -------
    tuple
        A tuple containing:
        - POD object
        - IoHelp object
        - Shape of the 3d field
        - Number of frequencies
        - Number of samples used (points in the fft_axis)
    """

    # ============
    # Main program
    # ============
    # Initialize
    # ============

    number_of_pod_fields = len(pod_fields)

    if preprocessing_field_operation is not None:
        if len(preprocessing_field_operation) != number_of_pod_fields:
            raise ValueError(
                "The preprocessing_field_operation list must have the same length as the pod_fields list"
            )
    
    if postprocessing_field_operation is not None:
        if len(postprocessing_field_operation) != number_of_pod_fields:
            raise ValueError(
                "The postprocessing_field_operation list must have the same length as the pod_fields list"
            )

    # Load the mass matrix
    #with h5py.File(mass_matrix_fname, "r") as f:
    #    bm = f[mass_matrix_key][:]
    if distributed_axis is not None:
        parallel_io = True
    else:
        parallel_io = False
        distributed_axis = 0

    dat = read_data(comm, fname= mass_matrix_fname, keys=[mass_matrix_key], parallel_io=parallel_io, distributed_axis=distributed_axis)
    bm = dat[mass_matrix_key]
    bm[np.where(bm == 0)] = 1e-14
    field_3d_shape = bm.shape

    # Obtain the number of frequencies you will obtain
    N_samples = bm.shape[fft_axis]
    number_of_frequencies = N_samples // 2 + 1
    # Choose the proper mass matrix slice
    bm = bm[get_mass_slice(fft_axis)]

    ioh = {"wavenumber": "buffers"}
    pod = {"wavenumber": "POD object"}

    # Initialize the buffers and objects for each wavenumber
    for kappa in range(0, number_of_frequencies):

        # Instance io helper that will serve as buffer for the snapshots
        ioh[kappa] = IoHelp(
            comm,
            number_of_fields=number_of_pod_fields,
            batch_size=p,
            field_size=bm.size,
            mass_matrix_data_type=bm.dtype,
            field_data_type=np.complex128,
            module_name="buffer_kappa" + str(kappa),
        )

        # Put the mass matrix in the appropiate format (long 1d array)
        mass_list = []
        for i in range(0, number_of_pod_fields):
            mass_list.append(np.copy(np.sqrt(bm)))
        ioh[kappa].copy_fieldlist_to_xi(mass_list)
        ioh[kappa].bm1sqrt[:, :] = np.copy(ioh[kappa].xi[:, :])

        # Instance the POD object
        pod[kappa] = POD(
            comm, number_of_modes_to_update=k, global_updates=True, auto_expand=False
        )

    # ============
    # Main program
    # ============
    # Update modes
    # ============

    j = 0
    while j < len(file_sequence):

        # Load the snapshot data
        fname = file_sequence[j]
        #with h5py.File(fname, "r") as f:
        #   fld_data = []
        #    for field in pod_fields:
        #        fld_data.append(f[field][:])
        fld_data_ = read_data(comm, fname=fname, keys=pod_fields, parallel_io=parallel_io, distributed_axis=distributed_axis)

        # Set up the info of the POD for the current snapshot
        pod_info = dict(pod_fields=pod_fields, field_shape=field_3d_shape, fft_axis=fft_axis, N_samples=N_samples, distributed_axis=distributed_axis, comm=comm, bm=bm, fname=fname, fidx=j, logger=ioh[0].log)

        # Apply the preprocessing tensor operations if needed:
        if preprocessing_tensor_operation is not None:
            fld_data_ = preprocessing_tensor_operation(fld_data_, pod_info)

        # Put the fields in a list
        fld_data = [fld_data_[field] for field in pod_fields]

        # Apply the preprocessing field operations if needed:
        if preprocessing_field_operation is not None:
            for i in range(0, number_of_pod_fields):
                if preprocessing_field_operation[i] is not None:
                    fld_data[i] = preprocessing_field_operation[i](fld_data[i])

        # Perform the fft
        for i in range(0, number_of_pod_fields):
            fld_data[i] = np.fft.fft(
                fld_data[i], axis=fft_axis
            ) / fourier_normalization(N_samples)

        # For each wavenumber, load buffers and update if needed
        for kappa in range(0, number_of_frequencies):

            # Get the proper slice for the wavenumber
            positive_wavenumber_slice = get_wavenumber_slice(kappa, fft_axis)

            # Get the wavenumber data
            wavenumber_data = []
            for i in range(0, number_of_pod_fields):
                wavenumber_data.append(
                    fld_data[i][positive_wavenumber_slice] * degenerate_scaling(kappa)
                )  # Here add contributions from negative wavenumbers

            # Put the fourier snapshot data into a column array
            ioh[kappa].copy_fieldlist_to_xi(wavenumber_data)

            # Load the column array into the buffer
            ioh[kappa].load_buffer(scale_snapshot=True)

            # Update POD modes
            if ioh[kappa].update_from_buffer:
                pod[kappa].update(
                    comm, buff=ioh[kappa].buff[:, : (ioh[kappa].buffer_index)]
                )

        j += 1

    # ============
    # Main program
    # ============
    # rscale modes
    # ============

    for kappa in range(0, number_of_frequencies):
        # Check if there is information in the buffer that should be taken in case the loop exit without flushing
        if ioh[kappa].buffer_index > ioh[kappa].buffer_max_index:
            ioh[kappa].log.write(
                "info", "All snapshots where properly included in the updates"
            )
        else:
            ioh[kappa].log.write(
                "warning",
                "Last loaded snapshot to buffer was: "
                + repr(ioh[kappa].buffer_index - 1),
            )
            ioh[kappa].log.write(
                "warning",
                "The buffer updates when it is full to position: "
                + repr(ioh[kappa].buffer_max_index),
            )
            ioh[kappa].log.write(
                "warning",
                "Data must be updated now to not lose anything,  Performing an update with data in buffer ",
            )
            pod[kappa].update(
                comm, buff=ioh[kappa].buff[:, : (ioh[kappa].buffer_index)]
            )
        
        # Apply the postprocessing field operations if needed:
        if postprocessing_field_operation is not None:
            for i in range(0, number_of_pod_fields):
                if postprocessing_field_operation[i] is not None:
                    mode_size = bm.size
                    start = i * mode_size
                    end = (i + 1) * mode_size
                    # Apply the postprocessing operation to the modes
                    pod[kappa].u_1t[start:end, :] = postprocessing_field_operation[i](pod[kappa].u_1t[start:end, :])

        # Scale back the modes (with the mass matrix)
        pod[kappa].scale_modes(comm, bm1sqrt=ioh[kappa].bm1sqrt, op="div")

        # Scale back the modes (with wavenumbers and degeneracy)
        pod[kappa].u_1t = pod[kappa].u_1t / degenerate_scaling(kappa)

        # Rotate local modes back to global, This only enters in effect if global_update = false
        pod[kappa].rotate_local_modes_to_global(comm)

    return pod, ioh, field_3d_shape, number_of_frequencies, N_samples


def write_3dfield_to_file(
    fname: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pod: dict[int, POD],
    ioh: dict[int, IoHelp],
    wavenumbers: list[int],
    modes: list[int],
    field_shape: tuple,
    fft_axis: int,
    field_names: list[str],
    N_samples: int,
    snapshots: list[int] = None,
    distributed_axis: int = None,
    comm = None,
):
    """
    Write 3D fields.

    Parameters
    ----------
    fname : str
        Name of the file to write the data to
    x : np.ndarray
        X coordinates of the field
    y : np.ndarray
        Y coordinates of the field
    z : np.ndarray
        Z coordinates of the field
    pod : dict[int, POD]
        Dictionary of POD object with the modes to transform to physical space
        the int key is the wavenumber
    ioh : dict[int, IoHelp]
        Dictionary of IoHelp object, which has some functionalities to split fields
        the int key is the wavenumber
    wavenumbers : list[int]
        List of wavenumbers to use in the operations
    modes : int
        list of the modes to use in the operations.
        if snapshot is not given, the modes will be transformed to physical space and returned.
        if snapshot is given, the modes will be used to reconstruct the snapshots and returned.
    field_shape : tuple
        Shape of the field in physical space
    fft_axis : int
        Axis where the fft was performed
    field_names : list[str]
        List of field names to put in the output dictionary
    N_samples : int
        Number of samples in the fft
    snapshots : list[int], optional
        List of snapshots to transform to physical space, by default None
        If this option is given, then the return will be a list of snapshots in physical space
        using the snapshot indices for the reconstruction.
        Be mindfull that the snapshot indices should be in the range of the snapshots used to create the POD objects.

    Returns
    -------
    None
    """
    
    if distributed_axis is not None:
        parallel_io = True
    else:
        parallel_io = False
        distributed_axis = 0

    # Always iterate over the wavenumbers or snapshots to not be too harsh on memory
    # Write a reconstruction to vtk
    if isinstance(snapshots, list):

        for snapshot in snapshots:
            # Fetch the data for this mode and wavenumber
            reconstruction_dict = physical_space(
                pod,
                ioh,
                wavenumbers,
                modes,
                field_shape,
                fft_axis,
                field_names,
                N_samples,
                snapshots=[snapshot],
            )

            # Write 3d_field
            sufix = f"reconstructed_data_{snapshot}"

            # Check the extension and path of the file
            ## Path
            path = os.path.dirname(fname)
            if path == "":
                path = "."
            ## prefix
            prefix = os.path.basename(fname).split(".")[0]
            ## Extension
            extension = os.path.basename(fname).split(".")[1]

            if (extension == "vtk") or (extension == "vts"):
                outname = f"{path}/{prefix}_{sufix}"
                print(f"Writing {outname}")
                for field in reconstruction_dict[snapshot].keys():
                    reconstruction_dict[snapshot][field] = np.ascontiguousarray(reconstruction_dict[snapshot][field])
                gridToVTK(outname, x, y, z, pointData=reconstruction_dict[snapshot])

    # Write modes to vtk
    else:

        for kappa in wavenumbers:
            for mode in modes:

                # Fetch the data for this mode and wavenumber
                mode_dict = physical_space(
                    pod,
                    ioh,
                    [kappa],
                    [mode],
                    field_shape,
                    fft_axis,
                    field_names,
                    N_samples,
                    snapshots,
                )

                # Write 3D field
                sufix = f"kappa_{kappa}_mode{mode}"

                # Check the extension and path of the file
                ## Path
                path = os.path.dirname(fname)
                if path == "":
                    path = "."
                ## prefix
                prefix = os.path.basename(fname).split(".")[0]
                ## Extension
                extension = os.path.basename(fname).split(".")[1]

                if (extension == "vtk") or (extension == "vts"):
                    outname = f"{path}/{prefix}_{sufix}.vtk"
                    print(f"Writing {outname}") 
                    for field in mode_dict[kappa][mode].keys():
                        mode_dict[kappa][mode][field] = np.ascontiguousarray(mode_dict[kappa][mode][field])
                    gridToVTK(outname, x, y, z, pointData=mode_dict[kappa][mode])
                elif extension == "hdf5":
                    outname = f"{path}/{prefix}_{sufix}.hdf5"
                    print(f"Writing {outname}")
                    write_data(comm, fname=outname, data_dict = mode_dict[kappa][mode], parallel_io=parallel_io, distributed_axis=distributed_axis) 


def save_pod_state(comm, fname: str, 
                   pod: dict[int, POD], 
                   ioh: dict[int, IoHelp], 
                   pod_fields: list[str], 
                   fft_axis: int, 
                   N_samples: int,
                   number_of_frequencies: int, 
                   parallel_io: bool = False, 
                   distributed_axis: int = 0):
    """
    Save the POD object dictionary to a file. From this, one can produce more analysis.

    Parameters
    ----------
    fname : str
        Name of the file to save the data to
    pod : dict[int, POD]
        Dictionary of POD object with the modes to transform to physical space
        the int key is the wavenumber
    parallel_io : bool, optional
        If True, the data will be written in parallel, by default False
    distributed_axis : int, optional
        Axis where the data is distributed, by default 0
        This is only used if parallel_io is True.

    """
    
    log = Logger(comm=comm, module_name="pod-savestate")

    path = os.path.dirname(fname)
    if path == "":
        path = "."
    prefix = os.path.basename(fname).split(".")[0]
    extension = os.path.basename(fname).split(".")[1]

    # Save the modes
    log.write("info", "Saving POD modes to file")
    mode_data = {}
    for kappa in pod.keys():
        try:
            int(kappa)
        except:
            continue

        mode_per_field = ioh[kappa].split_narray_to_1dfields(pod[kappa].u_1t)
        for f, field in enumerate(pod_fields):
            mode_data[f"field_{field}_wavenumber_{kappa}"] = mode_per_field[f].copy()

    write_data(comm, fname=f"{prefix}_modes.{extension}", data_dict = mode_data, parallel_io=parallel_io, distributed_axis=distributed_axis) 

    log.write("info", "Saving POD singular values and right singular vectors to file")
    
    comm.Barrier()  # Ensure all processes are synchronized before saving
    if comm.Get_rank() == 0:

        # Save settings that were used to create the objects
        # Write the data to hdf5
        settings = {}
        settings["pod_fields"] = pod_fields
        settings["fft_axis"] = fft_axis
        settings["N_samples"] = N_samples
        settings["number_of_frequencies"] = number_of_frequencies
        wavenumbers = []
        for kappa in pod.keys():
            try:
                int(kappa)
            except:
                continue
            wavenumbers.append(int(kappa))
        settings["wavenumbers"] = wavenumbers
        f = h5py.File(f"{prefix}_settings.{extension}", "w")
        f.attrs['settings'] = json.dumps(settings) 
        f.close()

        # Save singular values
        singular_value_data = {}
        for kappa in pod.keys():
            try:
                int(kappa)
            except:
                continue
            # Save the POD object
            singular_value_data[f"{kappa}"] = pod[kappa].d_1t
        write_data(comm, fname=f"{prefix}_singular_values.{extension}", data_dict = singular_value_data, parallel_io=False)

        # Save right singular vectors
        right_singular_vectors_data = {}
        for kappa in pod.keys():
            try:
                int(kappa)
            except:
                continue
            # Save the POD object
            right_singular_vectors_data[f"{kappa}"] = pod[kappa].vt_1t
        write_data(comm, fname=f"{prefix}_right_singular_vectors.{extension}", data_dict = right_singular_vectors_data, parallel_io=False)

    return


def load_pod_state(comm, fname, parallel_io: bool = False, distributed_axis: int = 0):
    
    path = os.path.dirname(fname)
    if path == "":
        path = "."
    prefix = os.path.basename(fname).split(".")[0]
    extension = os.path.basename(fname).split(".")[1]

    # First load the settings
    with h5py.File(f"{prefix}_settings.{extension}") as f:
        settings = json.loads(f.attrs['settings'])  

    pod_fields = settings["pod_fields"]
    fft_axis = settings["fft_axis"]
    N_samples = settings["N_samples"]
    number_of_frequencies = settings["number_of_frequencies"]
    wavenumbers = settings["wavenumbers"]

    # Load the modes
    pod = {}
    ioh = {}
    for wavenumber in wavenumbers:

        # Determine the keys to read
        wavenumber_keys = []
        for pod_field in pod_fields:
            wavenumber_keys.append(f"field_{pod_field}_wavenumber_{wavenumber}")
        
        # Read the data from the file
        wavenumber_data = read_data(comm, f"{prefix}_modes.{extension}", wavenumber_keys, parallel_io=parallel_io, distributed_axis=distributed_axis, dtype=np.complex128)

        # Get the field size
        field_size = wavenumber_data[wavenumber_keys[0]].shape[0]
        number_of_modes = wavenumber_data[wavenumber_keys[0]].shape[1]

        # Initialize the POD object
        pod[int(wavenumber)] = POD(
            comm, number_of_modes_to_update=number_of_modes, global_updates=True, auto_expand=False
        )
        # Concatenate the modes into one array
        pod[int(wavenumber)].u_1t = np.concatenate(
            [wavenumber_data[key].copy() for key in wavenumber_data.keys()], axis=0
        )

        # Initialize the IoHelp object
        ioh[int(wavenumber)] = IoHelp(
            comm,
            number_of_fields=len(pod_fields),
            batch_size=1,
            field_size=field_size,
            mass_matrix_data_type=np.float64,  # Assuming mass matrix is float64
            field_data_type=np.complex128,
            module_name="buffer_kappa" + str(wavenumber),
        )

        # Read the singular values
        singular_values = read_data(comm, f"{prefix}_singular_values.{extension}", [str(wavenumber)], parallel_io=False, dtype=np.float64)
        pod[int(wavenumber)].d_1t = singular_values[str(wavenumber)].copy()

        # Read the right singular vectors
        right_singular_vectors = read_data(comm, f"{prefix}_right_singular_vectors.{extension}", [str(wavenumber)], parallel_io=False, dtype=np.complex128)
        pod[int(wavenumber)].vt_1t = right_singular_vectors[str(wavenumber)].copy()

    return pod, ioh, settings 