import json
import numpy as np
import os

from ...monitoring.logger import Logger
from ...datatypes.msh import Mesh
from ...datatypes.field import Field
from ...io.ppymech.neksuite import preadnek, pwritenek, pynekread, pynekwrite


def average_field_files(
    comm,
    field_index_name="",
    output_folder="./",
    output_name="",
    output_batch_t_len=50000,
    mesh_index="",
    dtype=np.single,
    rel_tol=0.05,
    output_word_size=4,
    write_mesh=True,
):
    """
    Average field files in batches of a given time interval.

    Used to average files from a field index.

    Parameters
    ----------
    comm : MPI.COMM
        MPI communicator.

    field_index_name : str
        Index file that contains the information of the field files to be averaged.
        Relative or absule path to the file should be provided.
    output_folder : str
        Output folder where the averaged field files will be saved.
        By default, the same folder as the field_index_name will be used.
        If a path is provided, that one will be used for all outputs.
    output_batch_t_len : float
        Time interval in time units that should be included in each batch.
        If you want to average all the fields in the index, set this value to a large number.
        Particularly, a number larger than the sum of all the time intervals in the stat files.
    mesh_index : int
        Index of the mesh file in the field_index_name.
        If not provided, the mesh index will be assumed to be 0.
        In other words, we assume that the first file in the index contain the mesh information.
    dtype : np.dtype
        Precision of the data once is read in memory. Precision in file does not matter.
        default is np.single.
    rel_tol : float
        Relative tolerance to consider when dividing the files into batches.
        A 5% (5e-2) tolerance is used by default.
    output_word_size : int
        Word size of the output files. Default is 4 bytes, i.e. single precision.
        The other option is 8 bytes, i.e. double precision.
    write_mesh : bool
        Write the mesh in the output files. Default is True.
        If False, the mesh will anyway be written in the first file.
        of the outputs. I.e., batch 0.

    Notes
    -----
    This function will output the results in single precision and will put the mesh in all outputs by default.
    Future implementations might include this as an option.

    Returns
    -------
    None
        Files are written to disk.
    """

    logger = Logger(comm=comm, module_name="average_field_files")

    logger.write("info", f"Averaging field files in index: {field_index_name}")
    logger.write("info", f"Output files will be saved in: {output_folder}")

    # Read the json index file
    with open(field_index_name, "r") as f:
        file_index = json.load(f)

    logger.write(
        "info", f"Files will be averaged in batches of {output_batch_t_len} time units"
    )
    logger.write("info", f"Start dividing files into batches")

    batches = {}
    # Loop over the files and group them into batches
    batch_number = 0
    batches[batch_number] = {}
    file_in_batch_number = 0
    current_batch_t_len = 0.0
    # for i in range(0, len(file_index)-1):
    for i, key in enumerate(file_index.keys()):
        try:
            int_key = int(key)
        except ValueError:
            continue

        file_t_interval = file_index[key]["time_interval"]

        # Add to the current batch
        if current_batch_t_len + file_t_interval <= output_batch_t_len * (
            1 + rel_tol
        ):  # Add a 5% tolerance
            create_new_batch = False
        else:
            batches[batch_number]["averaging_time"] = current_batch_t_len
            create_new_batch = True

        if create_new_batch:
            batch_number += 1
            batches[batch_number] = {}
            current_batch_t_len = 0.0
            file_in_batch_number = 0

        batches[batch_number][file_in_batch_number] = file_index[key]
        current_batch_t_len += file_t_interval
        file_in_batch_number += 1

    # Check if averaging time is in the dictionary
    if batches[batch_number].get("averaging_time") is None:
        batches[batch_number]["averaging_time"] = current_batch_t_len

    logger.write(
        "info",
        f"Finished dividing files into batches.",
    )

    logger.write(
        "info",
        f"Writing {output_folder}batches_{os.path.basename(field_index_name)} batch index",
    )
    logger.tic()
    if comm.Get_rank() == 0:
        with open(
            output_folder + "batches_" + os.path.basename(field_index_name), "w"
        ) as outfile:
            outfile.write(json.dumps(batches, indent=4))
    comm.Barrier()
    logger.toc()

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

    logger.write("info", f"Reading mesh from file {mesh_index} in {field_index_name}")
    logger.write("info", f"Reading mesh in precision:  {dtype}")

    msh_fname = file_index[str(mesh_index)]["path"]
    msh = Mesh(comm, create_connectivity=False)
    pynekread(msh_fname, comm, data_dtype=dtype, msh=msh)

    if comm.Get_rank() == 0:
        print(
            "================================================================================================="
        )

    for batch in batches.keys():

        logger.tic()
        logger.write("info", f"Averaging files in batch {batch}")

        batch_time = 0.0
        batch_mean_field = Field(comm)
        work_field = Field(comm)

        for i, file in enumerate(
            key for key in batches[batch].keys() if isinstance(key, (int))
        ):

            if i == 0:
                if output_name == "":
                    out_fname = (
                        batches[batch][file]["fname"].split(".")[0][:-1]
                        + "0.f"
                        + str(batch).zfill(5)
                    )
                
                    out_fname = "batch_" + out_fname
                else:
                    prefix = os.path.basename(output_name).split(".")[0]
                    out_fname = (
                        prefix
                        + "0.f"
                        + str(batch).zfill(5)
                    )                

            fname = batches[batch][file]["path"]
            file_dt = batches[batch][file]["time_interval"]

            logger.write(
                "info",
                f"Processing file: {os.path.basename(fname)}, with time interval: {file_dt}",
            )

            if i == 0:
                pynekread(fname, comm, data_dtype=dtype, fld=batch_mean_field)

                logger.write("info", f"Multiplying fields by dt = {file_dt}")

                # Multiply the fields by the time interval
                for field in batch_mean_field.fields.keys():
                    for qoi in range(0, len(batch_mean_field.fields[field])):
                        batch_mean_field.fields[field][qoi] = (
                            batch_mean_field.fields[field][qoi] * file_dt
                        )

            else:
                work_field.clear()
                pynekread(fname, comm, data_dtype=dtype, fld=work_field)

                logger.write("info", f"Multiplying fields by dt = {file_dt}")

                # Multiply the fields by the time interval
                for field in batch_mean_field.fields.keys():
                    for qoi in range(0, len(batch_mean_field.fields[field])):
                        batch_mean_field.fields[field][qoi] += (
                            work_field.fields[field][qoi] * file_dt
                        )

            batch_time += file_dt

            logger.write("info", f"Current averaging time of batch = {batch_time}")

        logger.write("info", f"Reading files in batch {batch} finished")
        logger.write("info", f"Dividing fields by total time in batch={batch_time}")

        # Divide by the total time
        for field in batch_mean_field.fields.keys():
            for qoi in range(0, len(batch_mean_field.fields[field])):
                batch_mean_field.fields[field][qoi] = (
                    batch_mean_field.fields[field][qoi] / batch_time
                )

        # Write fields
        logger.write("info", f"Writing averaged fields in batch {batch} to file")

        # Set the time to be the cummulative time of the batch
        batch_mean_field.t = batch_time
        if batch == 0:
            write_coords = True
        else:
            write_coords = write_mesh
        pynekwrite(
            output_folder + out_fname,
            comm,
            fld=batch_mean_field,
            msh=msh,
            wdsz=output_word_size,
            istep=batch,
            write_mesh=write_coords,
        )

        logger.write("info", f"Processing of batch : {batch} done.")
        logger.toc()

        if comm.Get_rank() == 0:
            print(
                "================================================================================================="
            )
