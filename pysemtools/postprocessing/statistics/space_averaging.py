import json
import numpy as np
import os

from ...monitoring.logger import Logger
from ...datatypes.msh import Mesh
from ...datatypes.field import Field, FieldRegistry
from ...datatypes.coef import Coef
from ...io.ppymech.neksuite import preadnek, pwritenek, pynekread, pynekwrite
from ...comm.router import Router
from ...interpolation.point_interpolator.single_point_helper_functions import GLL_pwts


def space_average_field_files(
    comm,
    file_sequence = "",
    field_index_name="",
    output_folder="./",
    mesh_index="",
    dtype=np.single,
    rel_tol=0.01,
    output_word_size=4,
    write_mesh=True,
    homogeneous_dir="z",
    output_in_3d=False,
):
    """
    Average field files in the given dimension.

    Used to average files from a field index.

    Parameters
    ----------
    comm : MPI.COMM
        MPI communicator.

    file_sequence : list
        List of file names to be averaged in the homogeneous direction.
    field_index_name : str
        Index file that contains the information of the field files to be averaged.
        Relative or absule path to the file should be provided.
    output_folder : str
        Output folder where the averaged field files will be saved.
        By default, the same folder as the field_index_name will be used.
        If a path is provided, that one will be used for all outputs.
    mesh_index : int
        Index of the mesh file in the field_index_name.
        If not provided, the mesh index will be assumed to be 0.
        In other words, we assume that the first file in the index contain the mesh information.
        If one does not use a field_index but instead a file sequence, then the mesh should be in the first file.
    dtype : np.dtype
        Precision of the data once is read in memory. Precision in file does not matter.
        default is np.single.
    rel_tol : float
        Relative tolerance to consider when dividing the files into batches.
        A 1% (1e-2) tolerance is used by default.
    output_word_size : int
        Word size of the output files. Default is 4 bytes, i.e. single precision.
        The other option is 8 bytes, i.e. double precision.
    write_mesh : bool
        Write the mesh in the output files. Default is True.
        If False, the mesh will anyway be written in the first file.
        of the outputs. I.e., batch 0.
    homogeneous_dir : str
        Direction to average the field. Default is "z".
        Options are "x", "y", or "z".
    output_in_3d : bool
        If the output should be in 3D. Default is False.
        The 3D output will contain only 1 element in the homogenous direction.
        This is useful if one wants to interpolate later, as those routines require 3d inputs.

    Notes
    -----
    This function will output the results in single precision and will put the mesh in all outputs by default.
    Future implementations might include this as an option.

    Returns
    -------
    None
        Files are written to disk.
    """

    rt = Router(comm)
    logger = Logger(comm=comm, module_name="space_average_field_files")

    if field_index_name != "":
        logger.write("info", f"Averaging field files in index: {field_index_name}")

        # Read the mesh
        if mesh_index == "":
            logger.write("warning", "Mesh index not provided")
            logger.write(
                "warning",
                "Provide the mesh_index keyword with an index to a file that contiains the mesh",
            )

            for i, key in enumerate(file_index.keys()):
                try:
                    int_key = int(key)
                except ValueError:
                    continue
                mesh_index = key
                logger.write(
                    "warning", f"we assume that the mesh index correspond to {mesh_index}"
                )
                break
    
        # Read the json index file
        with open(field_index_name, "r") as f:
            file_index = json.load(f)

        logger.write("info", f"Reading mesh from file {mesh_index} in {field_index_name}")
        logger.write("info", f"Reading mesh in precision:  {dtype}")
    
        msh_fname = file_index[str(mesh_index)]["path"]

        file_list = list(file_index.keys())

    elif file_sequence != "":
        logger.write("info", f"Averaging field files in sequence: {file_sequence}")
        logger.write("info", f"Reading the mesh from the first file in the sequence")

        msh_fname = file_sequence[0]

        file_list = file_sequence

    else:
        logger.write("error", "No field files provided")
        logger.write("error", "Please provide either a field index or a file sequence")
        return 
    
    
    logger.write("info", f"Output files will be saved in: {output_folder}")

    msh = Mesh(comm, create_connectivity=False)
    pynekread(msh_fname, comm, data_dtype=dtype, msh=msh)

    # Create the coefficient object
    logger.write("info", f"Initializing coefficients")
    coef = Coef(msh, comm, get_area=False)

    if homogeneous_dir == "z":
        direction = 1
    elif homogeneous_dir == "y":
        direction = 2
    elif homogeneous_dir == "x":
        direction = 3
    else:
        logger.write("error", "Direction not recognized")
        logger.write("error", "Please provide a valid direction (x, y, or z)")
        return

    logger.write("info", f"Averaging in the direction: {homogeneous_dir}")
    if homogeneous_dir == "z":
        xx = msh.x
        yy = msh.y
        direction = 1
        ax = (2, 3)
    elif homogeneous_dir == "y":
        xx = msh.x
        yy = msh.z
        direction = 2
        ax = (1, 3)
    elif homogeneous_dir == "x":
        xx = msh.y
        yy = msh.z
        direction = 3
        ax = (1, 2)
    else:
        logger.write("error", "Direction not implemented")
        return

    logger.write("info", f"Identifying unique cross-sections in each rank separately")
    # Find the centroid of all elements:
    # First assumption here. We assume that all these values will not change in the z direction
    # So we pick the first one
    x_bar = (
        np.min(xx, axis=ax)[:, 0]
        + (np.max(xx, axis=ax)[:, 0] - np.min(xx, axis=ax)[:, 0]) / 2
    )
    y_bar = (
        np.min(yy, axis=ax)[:, 0]
        + (np.max(yy, axis=ax)[:, 0] - np.min(yy, axis=ax)[:, 0]) / 2
    )
    centroids = np.zeros((msh.nelv, 2), dtype=dtype)
    centroids[:, 0] = x_bar
    centroids[:, 1] = y_bar

    # Find the unique centroids in the rank
    rank_unique_centroids = np.unique(centroids, axis=0)

    # Get el owners
    el_owner = get_el_owner(
        logger=logger,
        rank_unique_centroids=rank_unique_centroids,
        rt=rt,
        dtype=dtype,
        rel_tol=rel_tol,
    )

    # Map centroids to unique centroids in this rank
    logger.write(
        "info",
        f"Mapping the indices of the original elements to those of the unique elements in the specified direction",
    )
    elem_to_unique_map = np.zeros((msh.nelv), dtype=np.int64)
    for i in range(0, msh.nelv):
        elem_to_unique_map[i] = np.where(
            np.all(
                np.isclose(rank_unique_centroids, centroids[i, :], rtol=rel_tol), axis=1
            )
        )[0][0]

    logger.write("info", f"Getting 2D slices")
    # Create a 2D sem arrays to hold averages
    if homogeneous_dir == "z":
        slice_shape = (rank_unique_centroids.shape[0], 1, msh.ly, msh.lx)
        slice_shape_e = (-1, 1, msh.ly, msh.lx)
    elif homogeneous_dir == "y":
        slice_shape = (rank_unique_centroids.shape[0], 1, msh.lz, msh.lx)
        slice_shape_e = (-1, 1, msh.lz, msh.lx)
    elif homogeneous_dir == "x":
        slice_shape = (rank_unique_centroids.shape[0], 1, msh.lz, msh.ly)
        slice_shape_e = (-1, 1, msh.lz, msh.ly)
    x_2d = np.zeros(slice_shape, dtype=dtype)
    y_2d = np.zeros(slice_shape, dtype=dtype)

    # Average the fileds to test
    rank_weights = average_field_in_dir_local(
        avrg_field=x_2d,
        field=xx,
        coef=coef,
        elem_to_unique_map=elem_to_unique_map,
        direction=direction,
    )
    global_average, elements_i_own = average_slice_global(
        avrg_field=x_2d, el_owner=el_owner, router=rt, rank_weights=rank_weights
    )

    _ = average_field_in_dir_local(
        avrg_field=y_2d,
        field=yy,
        coef=coef,
        elem_to_unique_map=elem_to_unique_map,
        direction=direction,
    )
    _ = average_slice_global(
        avrg_field=y_2d, el_owner=el_owner, router=rt, rank_weights=rank_weights
    )

    logger.write("info", f"Verifying averaging")

    if global_average:
        passed = True
        for e in elements_i_own:

            if direction == 1:
                t1 = np.allclose(x_2d[elem_to_unique_map[e], :, :, :], xx[e, 0, :, :])
                t2 = np.allclose(y_2d[elem_to_unique_map[e], :, :, :], yy[e, 0, :, :])
            elif direction == 2:
                t1 = np.allclose(x_2d[elem_to_unique_map[e], :, :, :], xx[e, :, 0, :])
                t2 = np.allclose(y_2d[elem_to_unique_map[e], :, :, :], yy[e, :, 0, :])
            elif direction == 3:
                t1 = np.allclose(x_2d[elem_to_unique_map[e], :, :, :], xx[e, :, :, 0])
                t2 = np.allclose(y_2d[elem_to_unique_map[e], :, :, :], yy[e, :, :, 0])

            passed = np.all([t1, t2])

            if not passed:
                break

        if passed:
            logger.write("info", f"Averaging test passed: {passed}")
        else:
            logger.write("error", f"Averaging test passed: {passed}")

    # Create 2d mesh object
    logger.write("info", f"Creating 2D mesh object on ranks that have them")
    comm_2d = comm.Split(color=1 if global_average else 0, key=comm.Get_rank())
    if global_average:
        x_2d_msh = x_2d[elements_i_own, :, :, :].reshape(slice_shape_e)
        y_2d_msh = y_2d[elements_i_own, :, :, :].reshape(slice_shape_e)
        z_2d_msh = np.zeros_like(x_2d_msh)

        if output_in_3d:

            # Get the quadrature nodes
            nn = x_2d_msh.shape[-1]
            x_, _ = GLL_pwts(nn)
            x_gll_ = np.flip(x_)

            x_2d_msh = np.tile(x_2d_msh, (1, nn, 1, 1))
            y_2d_msh = np.tile(y_2d_msh, (1, nn, 1, 1))
            z_2d_msh = np.tile(z_2d_msh, (1, nn, 1, 1))
            z_2d_msh[:, :, :, :] = x_gll_.reshape((1, nn, 1, 1))

        msh_2d = Mesh(
            comm_2d, create_connectivity=False, x=x_2d_msh, y=y_2d_msh, z=z_2d_msh
        )

        fld_2d = FieldRegistry(comm_2d)
        fld_2d.add_field(comm_2d, field_name="x", field=x_2d_msh, dtype=dtype)
        fld_2d.add_field(comm_2d, field_name="y", field=y_2d_msh, dtype=dtype)
        fld_2d.add_field(
            comm_2d,
            field_name="rank",
            field=np.ones_like(x_2d_msh) * comm.rank,
            dtype=dtype,
        )
        pynekwrite(
            output_folder + "2dcoordinates0.f00000",
            comm_2d,
            msh=msh_2d,
            fld=fld_2d,
            wdsz=output_word_size,
            write_mesh=write_mesh,
        )
        fld_2d.clear()

    if comm.Get_rank() == 0:
        print(
            "================================================================================================="
        )

    # Create the field that will hold the 3D data that is read
    fld = FieldRegistry(comm)

    logger.write(
        "info", f"Starting to go through the files in the index: {field_index_name}"
    )
    logger.tic()
    write_coords = True
    for i, file in enumerate(file_list):

        # Mark if the z velocity is in the file 
        z_velocity_last = False

        if field_index_name != "":
            try:
                int_key = int(file)
            except ValueError:
                continue

            out_fname = (
                file_index[file]["fname"].split(".")[0][:-1]
                + "0.f"
                + file_index[file]["fname"].split(".")[1][1:]
            )
            out_fname = f"{homogeneous_dir}_avg_{out_fname}"

            # read the file
            logger.write("info", f"Reading file {file_index[file]['path']}")
            pynekread(file_index[file]["path"], comm, data_dtype=dtype, fld=fld)
        elif file_sequence != "":
            
            in_fname = os.path.basename(file)
            out_fname = (
                in_fname.split(".")[0][:-1] + "0.f" + in_fname.split(".")[1][1:]
            )
            out_fname = f"{homogeneous_dir}_avg_{out_fname}"

            # read the file
            logger.write("info", f"Reading file {file}")
            pynekread(file, comm, data_dtype=dtype, fld=fld)
        else:
            logger.write("error", "No field files provided")
            logger.write("error", "Please provide either a field index or a file sequence")
            return

        logger.write("info", f"Averaging the fields in the specified direction")
        for field_key in fld.registry.keys():

            field_2d = np.zeros_like(x_2d)

            _ = average_field_in_dir_local(
                avrg_field=field_2d,
                field=fld.registry[field_key],
                coef=coef,
                elem_to_unique_map=elem_to_unique_map,
                direction=direction,
            )
            _ = average_slice_global(
                avrg_field=field_2d,
                el_owner=el_owner,
                router=rt,
                rank_weights=rank_weights,
            )

            if global_average:

                add_field = field_2d[elements_i_own, :, :, :].reshape(slice_shape_e)

                if output_in_3d:

                    add_field = np.tile(add_field, (1, nn, 1, 1))

                if (field_key == "w") and not output_in_3d  :
                    z_velocity_name = f"z_velocity"
                    z_velocity_field = add_field.copy()
                    z_velocity_last = True
                    logger.write("info", f"z velocity field found in the file and output in 3D is not requested. Saving z velocity as the last scalar")
                    continue
                else:
                    add_field_name = f"{field_key}"
                
                fld_2d.add_field(
                    comm_2d,
                    field_name=add_field_name,
                    field=add_field.copy(),
                    dtype=dtype,
                )

        # Add the z velocity as the last scalar
        if z_velocity_last:
            fld_2d.add_field(
                comm_2d,
                field_name=z_velocity_name,
                field=z_velocity_field,
                dtype=dtype,
            )

        logger.write(
            "info", f"Averaging finished, writing to file {output_folder}{out_fname}"
        )
        if global_average:
            pynekwrite(
                output_folder + out_fname,
                comm_2d,
                msh=msh_2d,
                fld=fld_2d,
                wdsz=output_word_size,
                write_mesh=write_coords,
            )
            # Clear the file for next iteration
            fld_2d.clear()
            # After the first file has been written, allow the option to not write mesh again by using the actual input by the user.
            write_coords = write_mesh

    logger.write("info", f"Run finished")
    logger.toc()


def average_field_in_dir_local(
    avrg_field=None, field=None, coef=None, elem_to_unique_map=None, direction=1
):
    """
    Average the field in the specified direction.

    Parameters
    ----------
    avrg_field : np.ndarray
        Array to store the averaged field.

    field : np.ndarray
        Field to be averaged.

    coef : Coef
        Coefficient object.

    elem_to_unique_map : np.ndarray
        Map from elements to unique elements.

    dir : int
        Direction to average the field. Default is 1. Options are 0, 1, 2.
        Note that the sum will always be done in the 0 direction as well, i.e.
        The dimension of the elements.
        So the sums are always done in the axis (0. dir).

    Notes
    -----
    Some casting to float64 is done to avoid problems in the sums.
    Even the mean functions were producing somewhat unexcpected results in single.
    Note that the "output" is in the buffer arrray provided as an input

    Returns
    -------
    rank_weights : np.ndarray
        Array of shape (nelv, 1, ly, lx) with the weights used to average
    """
    rank_weights = np.zeros(
        (avrg_field.shape), dtype=np.float64
    )  # double to avoid precision errors

    for i in range(0, avrg_field.shape[0]):

        # Averaging weights
        if direction == 1:
            b_shape = (1, 1, coef.B.shape[2], coef.B.shape[3])
        elif direction == 2:
            b_shape = (1, coef.B.shape[1], 1, coef.B.shape[3])
        elif direction == 3:
            b_shape = (1, coef.B.shape[1], coef.B.shape[2], 1)

        b = np.sum(
            np.float64(coef.B[np.where(elem_to_unique_map == i)[0], :, :, :]),
            axis=(0, direction),
        ).reshape(b_shape)
        weights = coef.B[elem_to_unique_map == i, :, :, :] / b

        # Do the weighted sum over the specified direction to get the average
        avrg_field[i, 0, :, :] = np.sum(
            np.float64(field[elem_to_unique_map == i, :, :, :]) * (weights),
            axis=(0, direction),
        )
        rank_weights[i, 0, :, :] = b.reshape(
            (int(np.sqrt(b.size)), int(np.sqrt(b.size)))
        )

    return rank_weights


def average_slice_global(
    avrg_field=None, el_owner=None, router=None, rank_weights=None
):
    """
    Average a 2D slice of a domain globally

    Parameters
    ----------
    avrg_field : np.ndarray
        Array to store the averaged field.
        We expect it to a be an array of sixe (nelv, 1, ly, lx)
        in this case nelv is the number of unique 2d elements in the rank.

    el_owner : np.ndarray
        Array of shape (nelv, 2) where the first column is the rank owner of the element
        and the second column is the element id in the owner rank.
        Typically, this is determined in another function.

    router : Router
        Router object to send and receive data.

    rank_weights : np.ndarray
        Array of shape (nelv, 1, ly, lx) with the weights used to average the field.

    Returns
    -------
    globally_averaged : bool
        If the field was globally averaged or not.

    elements_i_own : list
        List of sliced elements that the rank owns.
    """

    # Identify the destinations
    destinations = list(np.unique(el_owner[:, 0]))

    # Split the data that should be sent
    location_data = []
    field_data = []
    weigth_data = []
    for dest_ind, dest in enumerate(destinations):
        location_data.append(el_owner[np.where(el_owner[:, 0] == dest)[0], 1])
        field_data.append(avrg_field[np.where(el_owner[:, 0] == dest)[0], :, :, :])
        weigth_data.append(rank_weights[np.where(el_owner[:, 0] == dest)[0], :, :, :])

    # First send the locations
    dtype = el_owner.dtype
    sources, unique_locations = router.all_to_all(
        destination=destinations, data=location_data, dtype=dtype
    )

    # Then the data
    dtype = avrg_field.dtype
    sources, unique_field_data = router.all_to_all(
        destination=destinations, data=field_data, dtype=dtype
    )

    # Then the rank weights
    dtype = rank_weights.dtype
    sources, unique_rank_weights = router.all_to_all(
        destination=destinations, data=weigth_data, dtype=dtype
    )

    for i in range(0, len(unique_field_data)):
        unique_field_data[i] = unique_field_data[i].reshape(
            (-1, avrg_field.shape[1], avrg_field.shape[2], avrg_field.shape[3])
        )
        unique_rank_weights[i] = unique_rank_weights[i].reshape(
            (-1, avrg_field.shape[1], avrg_field.shape[2], avrg_field.shape[3])
        )

    globally_averaged = False
    elements_i_own = []

    if len(sources) > 0:

        for e in range(0, avrg_field.shape[0]):

            # Consider using list comprehension here
            elem_data = []
            elem_weights = []
            for ind, location in enumerate(unique_locations):
                if np.any(location == e):
                    elem_data.append(
                        np.float64(unique_field_data[ind][np.where(location == e)])
                    )
                    elem_weights.append(
                        unique_rank_weights[ind][np.where(location == e)]
                    )

            elem_data = np.array(elem_data)
            elem_weights = np.array(elem_weights)

            # If I do not own this element, then just continue
            if elem_data.size == 0:
                continue

            num = np.sum(elem_data * elem_weights, axis=(0, 1))
            den = np.sum(elem_weights, axis=(0, 1))

            avrg_field[e, :, :, :] = num / den

            elements_i_own.append(e)

        globally_averaged = True

    return globally_averaged, elements_i_own


def get_el_owner(
    logger=None, rank_unique_centroids=None, rt=None, dtype=np.single, rel_tol=0.01
):

    logger.write("info", f"Identifying which rank will take charge of the 2D elements")
    logger.write("warning", f"This might be slow...")
    logger.write("warning", f"It might be slower with many ranks...")
    logger.write(
        "warning", f"Consider using only as many as necesary for data to fit in memory"
    )
    logger.tic()
    # Go over all unique centroids in the domain per rank and find which ...
    # ... rank should take care of averaging the global data and writing out.

    ## Doing with this loop for memory efficiency, rather than creating a big a array
    ## Consider doing this in bigger batches, rather than point per point.
    ### Buffer to store each centroid
    centroid = np.zeros((1, 2), dtype=dtype)
    ### Buffer to store the rank and element owner of the centroid
    el_owner = np.zeros((rank_unique_centroids.shape[0], 2), dtype=np.int64)
    ### In this case, we send our centroid to all ranks
    destination = [rank for rank in range(0, rt.comm.Get_size())]
    ### Identify how many unique elements the rank that has the most have.
    tmp, _ = rt.all_gather(rank_unique_centroids.shape[0], dtype=np.int32)
    max_unique = np.max(tmp)

    ### Repeat the process for all the unique centroid in the rank
    for i in range(0, max_unique):
        ### Use this if statement since some ranks might have more unique centroid than me
        if i < rank_unique_centroids.shape[0]:
            centroid[0, 0] = rank_unique_centroids[i, 0]
            centroid[0, 1] = rank_unique_centroids[i, 1]
        else:
            centroid[0, 0] = -1e7
            centroid[0, 1] = -1e7

        ### send my centroid to all ranks and recieve the centroids from all others
        sources, recvbf = rt.all_to_all(
            destination=destination, data=centroid, dtype=np.single
        )

        ### Prepare a response array list indicating if I have the centroid that the rank sent.
        response = []
        for j in range(0, len(recvbf)):

            ### The buffers are always flattened. We know it is a 2D array
            recvbf[j] = recvbf[j].reshape((-1, 2))

            ### Flag == 0 if I do not have the centroid
            ### Flag == 1 if I have the centroid with a relative tolerance
            flag = np.zeros((2), dtype=np.int32)
            if np.any(
                np.all(
                    np.isclose(rank_unique_centroids, recvbf[j], rtol=rel_tol), axis=1
                )
            ):
                flag[0] = 1
                ### El id of the unique element that holds it
                flag[1] = np.where(
                    np.all(
                        np.isclose(rank_unique_centroids, recvbf[j], rtol=rel_tol),
                        axis=1,
                    )
                )[0][0]

            ### Append the flag to the response list
            response.append(flag)

        ### Now that the rank has checked if it has the centorid, we answer back to the original owners.
        _, recvbf = rt.all_to_all(destination=sources, data=response, dtype=np.int32)

        ### The rank owner of the element will be the lowest index rank that owns it.
        ### It will take care of averaging the data from all ranks and writing out.
        for j in range(0, rt.comm.Get_size()):
            if recvbf[j][0] == 1:
                el_owner[i, 0] = j
                el_owner[i, 1] = recvbf[j][1]
                break
    logger.write("info", f"done!")
    logger.toc()

    return el_owner
