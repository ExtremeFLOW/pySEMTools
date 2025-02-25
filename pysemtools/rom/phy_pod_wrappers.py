""" Contains functions to wrap the ROM types to easily post process data """

from .pod import POD
from .io_help import IoHelp
import numpy as np
import h5py
import os
from pyevtk.hl import gridToVTK
from ..io.wrappers import read_data
from ..datatypes.msh import Mesh
from ..datatypes.coef import Coef
from typing import Union

def pod_from_files(
    comm,
    file_sequence: list[str],
    pod_fields: list[str],
    mass_matrix_fname: str,
    mass_matrix_key: Union[str, list[str]],
    k: int,
    p: int,
) -> tuple:
    """
    Perform POD on a sequence of snapshot while applying fft in an homogenous direction of choice.

    This can be used on the spectral element mesh. In that case, indicate ["x", "y", "z"] as the mass_matrix_key.

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

    Returns
    -------
    tuple
        A tuple containing:
        - POD object
        - IoHelp object
        - Shape of the 3d field
    """

    number_of_pod_fields = len(pod_fields)

    # Load the mass matrix
    ## If it is in a file with a key
    if isinstance(mass_matrix_key, str):
        mass_matrix_key = [mass_matrix_key]
        mass_data = read_data(comm, mass_matrix_fname, mass_matrix_key)
        bm = mass_data[mass_matrix_key[0]]
        del mass_data
    ## If it should be created from a SEM mesh
    elif isinstance(mass_matrix_key, list):
        if mass_matrix_key != ["x", "y", "z"]:
            raise ValueError("Only ['x', 'y', 'z'] is supported for the spectral element mesh")
        
        mass_data = read_data(comm, mass_matrix_fname, mass_matrix_key)
        msh = Mesh(comm, x=mass_data['x'], y=mass_data['y'], z=mass_data['z'], create_connectivity=False)
        coef = Coef(msh, comm)
        bm = np.copy(coef.B)
        del mass_data, msh, coef

    # Remove zero entries from the mass matrix
    bm[np.where(bm == 0)] = 1e-12
    field_3d_shape = bm.shape

    # Instance io helper that will serve as buffer for the snapshots
    ioh = IoHelp(
        comm, number_of_fields=number_of_pod_fields, batch_size=p, field_size=bm.size
    )

    # Put the mass matrix in the appropiate format (long 1d array)
    mass_list = []
    for i in range(0, number_of_pod_fields):
        mass_list.append(np.copy(np.sqrt(bm)))
    ioh.copy_fieldlist_to_xi(mass_list)
    ioh.bm1sqrt[:, :] = np.copy(ioh.xi[:, :])

    # Instance the POD object
    pod = POD(comm, number_of_modes_to_update=k, global_updates=True, auto_expand=False)

    # Perform reading and updates
    j = 0
    while j < len(file_sequence):

        # Load the snapshot data
        fname = file_sequence[j] 
        data = read_data(comm, fname, pod_fields, dtype = np.single)
        fld_data = []
        for field_name in pod_fields:
            fld_data.append(data[field_name])    

        # Put the snapshot data into a column array
        ioh.copy_fieldlist_to_xi(fld_data)

        # Load the column array into the buffer
        ioh.load_buffer(scale_snapshot=True)

        # Update POD modes
        if ioh.update_from_buffer:
            pod.update(comm, buff=ioh.buff[:, : (ioh.buffer_index)])

        j += 1

    # Check if there is information in the buffer that should be taken in case the loop exit without flushing
    if ioh.buffer_index > ioh.buffer_max_index:
        ioh.log.write("info", "All snapshots where properly included in the updates")
    else:
        ioh.log.write(
            "warning",
            "Last loaded snapshot to buffer was: " + repr(ioh.buffer_index - 1),
        )
        ioh.log.write(
            "warning",
            "The buffer updates when it is full to position: "
            + repr(ioh.buffer_max_index),
        )
        ioh.log.write(
            "warning",
            "Data must be updated now to not lose anything,  Performing an update with data in buffer ",
        )
        pod.update(comm, buff=ioh.buff[:, : (ioh.buffer_index)])

    # Scale back the modes
    pod.scale_modes(comm, bm1sqrt=ioh.bm1sqrt, op="div")

    # Rotate local modes back to global, This only enters in effect if global_update = false
    pod.rotate_local_modes_to_global(comm)

    return pod, ioh, field_3d_shape


def physical_space(
    pod: POD,
    ioh: IoHelp,
    modes: list[int],
    field_shape: tuple,
    field_names: list[str],
    snapshots: list[int] = None,
):
    """
    Function to transform modes or snapshots from the POD objects into the physical space

    This will either produce a set of specified modes in physical space.

    Or it will use the specified modes to reconstruct the specified snapshots in physical space.

    Parameters
    ----------
    pod : POD
        POD object with the modes to transform to physical space
    ioh : IoHelp
        IoHelp object, which has some functionalities to split fields
    modes : list[int]
        list of the modes to use in the operations.
        if snapshot is not given, the modes will be transformed to physical space and returned.
        if snapshot is given, the modes will be used to reconstruct the snapshots and returned.
    field_shape : tuple
        Shape of the field in physical space
    field_names : list[str]
        List of field names to put in the output dictionary
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

        # Reconstruct the snapshots
        reconstruction = (
            pod.u_1t[:, modes].reshape(-1, len(modes))
            @ np.diag(pod.d_1t[modes])
            @ pod.vt_1t[np.ix_(modes, snapshots)].reshape(len(modes), len(snapshots))
        )

        # Go thorugh the wavenumbers in the list and put the modes in the physical space
        for snap_id, snapshot in enumerate(snapshots):

            physical_fields[snapshot] = {}

            ## Split the 1d snapshot into a list with the fields you want
            field_list1d = ioh.split_narray_to_1dfields(reconstruction[:, snap_id])
            ## Reshape the obtained 1d fields to be in their appropiate shape
            field_list_xd = [field.reshape(field_shape) for field in field_list1d]

            for i, field_name in enumerate(field_names):

                # Save the field in the dictionary
                physical_fields[snapshot][field_name] = np.copy(field_list_xd[i])

    # To obtain only the modes
    else:

        # Go thorugh the wavenumbers in the list and put the modes in the physical space
        physical_fields = {}

        for mode in modes:

            # Add the mode to the dictionary
            physical_fields[mode] = {}

            ## Split the 1d snapshot into a list with the fields you want
            field_list1d = ioh.split_narray_to_1dfields(pod.u_1t[:, mode])
            ## Reshape the obtained 1d fields to be in their appropiate shape
            field_list_xd = [field.reshape(field_shape) for field in field_list1d]

            for i, field_name in enumerate(field_names):

                # Save the field in the dictionary (only the real part)
                physical_fields[mode][field_name] = np.copy(field_list_xd[i])

    return physical_fields


def write_3dfield_to_file(
    fname: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pod: POD,
    ioh: IoHelp,
    modes: list[int],
    field_shape: tuple,
    field_names: list[str],
    snapshots: list[int] = None,
):
    """
    Write 3D fields

    Parameters
    ----------
    fname : str
        Name of the file to write the data to.
    x : np.ndarray
        x coordinates of the grid
    y : np.ndarray
        y coordinates of the grid
    z : np.ndarray
        z coordinates of the grid
    pod : POD
        POD object with the modes to transform to physical space
    ioh : IoHelp
        IoHelp object, which has some functionalities to split fields
    modes : list[int]
        list of the modes to use in the operations.
        if snapshot is not given, the modes will be transformed to physical space and returned.
        if snapshot is given, the modes will be used to reconstruct the snapshots and returned.
    field_shape : tuple
        Shape of the field in physical space
    field_names : list[str]
        List of field names to put in the output dictionary
    snapshots : list[int], optional
        List of snapshots to transform to physical space, by default None
        If this option is given, then the return will be a list of snapshots in physical space
        using the snapshot indices for the reconstruction.
        Be mindfull that the snapshot indices should be in the range of the snapshots used to create the POD objects.

    Returns
    -------
    """

    # Always iterate over the wavenumbers or snapshots to not be too harsh on memory
    # Write a reconstruction to vtk
    if isinstance(snapshots, list):

        for snapshot in snapshots:
            # Fetch the data for this mode and wavenumber
            reconstruction_dict = physical_space(
                pod, ioh, modes, field_shape, field_names, snapshots=[snapshot]
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

        for mode in modes:

            # Fetch the data for this mode and wavenumber
            mode_dict = physical_space(
                pod, ioh, [mode], field_shape, field_names, snapshots
            )

            # Write 3D field
            sufix = f"mode{mode}.vtk"

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
                gridToVTK(outname, x, y, z, pointData=mode_dict[mode])


def save_pod_state(fname: str, pod: POD):
    """
    Save the POD object to a file

    Parameters
    ----------
    fname : str
        Name of the file to save the POD object to.
    pod : POD
        POD object to save

    Returns
    -------
    """

    path = os.path.dirname(fname)
    if path == "":
        path = "."
    prefix = os.path.basename(fname).split(".")[0]
    extension = os.path.basename(fname).split(".")[1]

    f = h5py.File(f"{prefix}_modes.{extension}", "w")
    f.create_dataset(f"modes", data=pod.u_1t)
    f.close()

    f = h5py.File(f"{prefix}_singlular_values.{extension}", "w")
    f.create_dataset(f"singular_values", data=pod.d_1t)
    f.close()

    f = h5py.File(f"{prefix}_right_singular_vectors.{extension}", "w")
    f.create_dataset(f"right_singular_vectors", data=pod.vt_1t)
    f.close()

    return
