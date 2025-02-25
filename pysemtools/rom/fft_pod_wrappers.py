""" Contains functions to wrap the ROM types to easily post process data """

from .pod import POD
from .io_help import IoHelp
import numpy as np
import h5py
import os
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
    modes: list[int],
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
            fourier_reconstruction[kappa] = (
                pod[kappa].u_1t[:, modes].reshape(-1, len(modes))
                @ np.diag(pod[kappa].d_1t[modes])
                @ pod[kappa]
                .vt_1t[np.ix_(modes, snapshots)]
                .reshape(len(modes), len(snapshots))
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


def pod_fourier_1_homogenous_direction(
    comm,
    file_sequence: list[str],
    pod_fields: list[str],
    mass_matrix_fname: str,
    mass_matrix_key: str,
    k: int,
    p: int,
    fft_axis: int,
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

    # Load the mass matrix
    with h5py.File(mass_matrix_fname, "r") as f:
        bm = f[mass_matrix_key][:]
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
        with h5py.File(fname, "r") as f:
            fld_data = []
            for field in pod_fields:
                fld_data.append(f[field][:])

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
                sufix = f"kappa_{kappa}_mode{mode}.vtk"

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
                    gridToVTK(outname, x, y, z, pointData=mode_dict[kappa][mode])


def save_pod_state(fname: str, pod: dict[int, POD]):
    """
    Save the POD object dictionary to a file. From this, one can produce more analysis.

    Parameters
    ----------
    fname : str
        Name of the file to save the data to
    pod : dict[int, POD]
        Dictionary of POD object with the modes to transform to physical space
        the int key is the wavenumber
    """

    path = os.path.dirname(fname)
    if path == "":
        path = "."
    prefix = os.path.basename(fname).split(".")[0]
    extension = os.path.basename(fname).split(".")[1]

    f = h5py.File(f"{prefix}_modes.{extension}", "w")
    for kappa in pod.keys():
        try:
            int(kappa)
        except:
            continue
        # Save the POD object
        f.create_dataset(f"wavenumber_{kappa}", data=pod[kappa].u_1t)
    f.close()

    f = h5py.File(f"{prefix}_singlular_values.{extension}", "w")
    for kappa in pod.keys():
        try:
            int(kappa)
        except:
            continue
        # Save the POD object
        f.create_dataset(f"wavenumber_{kappa}", data=pod[kappa].d_1t)
    f.close()

    f = h5py.File(f"{prefix}_right_singular_vectors.{extension}", "w")
    for kappa in pod.keys():
        try:
            int(kappa)
        except:
            continue
        # Save the POD object
        f.create_dataset(f"wavenumber_{kappa}", data=pod[kappa].vt_1t)
    f.close()

    return
