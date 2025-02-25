"""Functions for indexing of files"""

import re
import json
import os
import numpy as np
from ..monitoring.logger import Logger
import glob
from pymech.neksuite.field import read_header
import numpy as np
from ..datatypes.field import FieldRegistry
from ..io.ppymech.neksuite import pynekread
from mpi4py import MPI

def index_files_from_log(comm, logpath="", logname="", progress_reports=50):
    """
    Idenx files based on the outputs of a neko log file.

    Index files based on a neko log file.

    Parameters
    ----------
    comm : MPI.COMM
        MPI communicator

    logpath : str
        Path to the log file. Optional. If not provided, the current working directory is used.
    logname : str
        Name of the log file
    progress_reports : int
        Number of progress reports (Default value = 50).

    Returns
    -------
    None
        An index file is written for each output type found in the log.

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from pysemtools.postprocessing.file_indexing import index_files_from_log
    >>> comm = MPI.COMM_WORLD
    >>> index_files_from_log(comm, logpath="path/to/logfile/", logname="logfile.log", progress_reports=50)
    """

    logger = Logger(comm=comm, module_name="file_index_from_log")

    if logpath == "":
        path = os.getcwd() + "/"
    else:
        path = logpath

    logger.write("info", f"Reading file: {path + logname}")
    logger.tic()
    logfile = open(path + logname, "r")
    log = logfile.readlines()
    number_of_lines = len(log)
    logfile.close()
    logger.toc()

    # Search patterns
    file_patterns = {"File name": re.compile(r"File name\s*:\s*(?P<value>.+)")}
    t_pattern = re.compile(r"t =\s*(?P<t_value>[+-]?\d*\.\d+E[+-]?\d+).*")
    output_number_pattern = re.compile(r"Output number\s*:\s*(?P<value>.+)")
    writing_time_pattern = re.compile(
        r"Writing at time:\s*(?P<writing_time>\d+\.\d+).*"
    )

    added_files = []

    logger.write("info", "Start reading the lines")
    report_interval = number_of_lines // progress_reports
    read_first_time_step = False
    for i, line in enumerate(log):

        if np.mod(i, report_interval) == 0:
            if comm.Get_rank() == 0:
                print(
                    "========================================================================================="
                )
            logger.write(
                "info",
                f"read up to line: {i}, out of {number_of_lines} <-> {i/number_of_lines*100}%",
            )
            if comm.Get_rank() == 0:
                print(
                    "========================================================================================="
                )

        if not read_first_time_step:

            for key, pattern in file_patterns.items():
                match = pattern.search(line)
                if match:
                    add_file = match.group("value").strip()
                    if add_file not in added_files:
                        added_files.append(add_file)

        # Find the first time step
        if "Time-step" in line and not read_first_time_step:

            match = t_pattern.search(log[i - 2])
            if match:
                start_time = float(match.group("t_value"))

            logger.write("info", f"Run start time: {start_time}")

            read_first_time_step = True

            files = {}
            files_found = {}
            files_index = {}
            files_last_sample = {}
            for file in added_files:
                files[file] = dict()
                files[file]["simulation_start_time"] = start_time

            for file in added_files:
                files_found[file] = False

            for file in added_files:
                files_index[file] = 0

            for file in added_files:
                files_last_sample[file] = start_time

            lines_to_check = 2 * len(added_files) + 2

        if "Writer output" in line and read_first_time_step:

            if comm.Get_rank() == 0:
                print(
                    "========================================================================================="
                )

            for file in added_files:
                files_found[file] = False

            writer_output_block = log[i : i + lines_to_check]

            # Check if this output block contains another output
            for j, next_line in enumerate(writer_output_block):
                if "Writer output" in next_line and j > 0:
                    k = j
                    break
                else:
                    k = 10000

            for j, next_line in enumerate(writer_output_block):

                if j >= k:
                    break

                for file in added_files:
                    if " " + file in next_line:
                        files_found[file] = True

                        match = output_number_pattern.search(writer_output_block[j + 1])
                        if match:
                            output_number = int(match.group("value"))
                            file_name = (
                                file.split(".")[0] + "0.f" + str(output_number).zfill(5)
                            )

                        files[file][files_index[file]] = dict()
                        files[file][files_index[file]]["fname"] = file_name

                        # Check if the file exists in this folder
                        file_exists = os.path.exists(path + file_name)
                        if file_exists:
                            files[file][files_index[file]]["path"] = os.path.abspath(
                                path + file_name
                            )
                        else:
                            files[file][files_index[file]][
                                "path"
                            ] = "file_not_in_folder"

                if "Writing at time" in next_line:

                    match = writing_time_pattern.search(next_line)
                    if match:
                        file_time = float(match.group("writing_time"))

            for file in added_files:
                if files_found[file]:
                    files[file][files_index[file]]["time"] = file_time
                    files[file][files_index[file]]["time_previous_output"] = (
                        files_last_sample[file]
                    )
                    files[file][files_index[file]]["time_interval"] = (
                        file_time - files_last_sample[file]
                    )

                    files_last_sample[file] = file_time

                    logger.write("info", f"{file} found")
                    for key in files[file][files_index[file]].keys():
                        logger.write(
                            "info", f"{key}: {files[file][files_index[file]][key]}"
                        )
                    files_index[file] += 1

    if comm.Get_rank() == 0:
        print(
            "========================================================================================="
        )
        print(
            "========================================================================================="
        )

    logger.write("info", "Check finished")

    for file in added_files:
        logger.write("info", f"Writing {file} file index")
        logger.tic()
        with open(path + os.path.basename(file).split('.')[0] + "_index.json", "w") as outfile:
            outfile.write(json.dumps(files[file], indent=4))
        logger.toc()

    del logger


def index_files_from_folder(
    comm,
    folder_path="",
    run_start_time=0,
    stat_start_time=0,
    output_folder="",
    file_type="",
    include_time_interval: bool = True,
    include_file_contents: bool = False,
):
    """
    Index files based on a folder.

    Index all field files in a folder.

    Parameters
    ----------
    comm : MPI.COMM
        mpi communicator.

    folder_path : str
        Path to the folder. Optional. If not provided, the current working directory is used.
    run_start_time : float
        Start time of the simulation (Default value = 0). This is used to calculate the intervals.
        Intervals that use this are any field that does not contain "stat" or "mean" in the name.
    stat_start_time : float
        Start time of the statistics (Default value = 0). This is used to calculate the intervals.
        Intervals that use this are any field that contains "stat" or "mean" in the name.
    output_folder : str
        Path to the output folder. Optional. If not provided, the same folder as the input folder is used.
    file_type: list
        A list with type of file that should be indexed. If empty, all files are indexed.
        An example is: ["field.fld", "stats.fld"]

    Returns
    -------
    None
        An index file is written for each output type found in the folder.

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from pysemtools.postprocessing.file_indexing import index_files_from_folder
    >>> comm = MPI.COMM_WORLD
    >>> index_files_from_folder(comm, folder_path="path/to/folder/", run_start_time=0, stat_start_time=0, output_folder = "")
    """

    if folder_path == "":
        folder_path = os.getcwd() + "/"

    if folder_path[-1] != "/":
        folder_path = folder_path + "/"

    logger = Logger(comm=comm, module_name="file_index_from_folder")

    logger.write("info", f"Reading folder: {folder_path}")

    # Get all files in the folder that are fields
    files_in_folder = sorted(glob.glob(folder_path + "*0.f*"))

    if len(files_in_folder) < 1:
        logger.write("warning", "No *0.f* files in the folder")

    added_files = []
    for i in range(0, len(files_in_folder)):
        files_in_folder[i] = os.path.basename(files_in_folder[i])

        ftype = files_in_folder[i].split(".")[0][:-1] + ".fld"

        # Detetmine if only some indices should be check
        if not isinstance(file_type, list):
            if ftype not in added_files:
                added_files.append(ftype)
        else:
            if ftype not in added_files and ftype in file_type:
                added_files.append(ftype)

    for ftype in added_files:
        logger.write("info", f"Found files with {ftype} pattern")

    if isinstance(file_type, list) and len(added_files) < 1:
        logger.write("warning", f"No files with pattern {file_type} found")

    # Do a test to see if the file type exist and if one wants to overwrite it
    remove = []
    for ftype in added_files:
        index_fname = folder_path + os.path.basename(ftype).split('.')[0] + "_index.json"
        file_exists = os.path.exists(index_fname)
        if file_exists:
            logger.write("warning", f"File {index_fname} exists. Overwrite?")
            overwrite = input("input: [yes/no] ")
            if overwrite == "no":
                remove.append(ftype)

    added_files = [nm for nm in added_files if nm not in remove]

    for ftype in added_files:
        logger.write("info", f"Writing index for {ftype} pattern")

    if comm.Get_rank() == 0:
        print(
            "========================================================================================="
        )

    files = {}
    files_index = {}
    files_last_sample = {}
    for ftype in added_files:
        files[ftype] = dict()
        if include_time_interval:
            if "stat" in ftype or "mean" in ftype:
                files[ftype]["statistics_start_time"] = stat_start_time
            else:
                files[ftype]["simulation_start_time"] = run_start_time
        files_index[ftype] = 0

    for i, file_in_folder in enumerate(files_in_folder):

        ftype = files_in_folder[i].split(".")[0][:-1] + ".fld"

        if ftype not in added_files:
            continue

        logger.write("debug", f"Indexing file: {file_in_folder}")

        files[ftype][files_index[ftype]] = dict()
        files[ftype][files_index[ftype]]["fname"] = file_in_folder
        files[ftype][files_index[ftype]]["path"] = os.path.abspath(
            folder_path + file_in_folder
        )

        # Determine the time from the header
        header = read_header(files[ftype][files_index[ftype]]["path"])
        current_time = header.time

        files[ftype][files_index[ftype]]["time"] = current_time

        if files_index[ftype] == 0:
            if include_time_interval:
                if "stat" in ftype or "mean" in ftype:
                    files[ftype][files_index[ftype]][
                        "time_previous_output"
                    ] = stat_start_time
                else:
                    files[ftype][files_index[ftype]][
                        "time_previous_output"
                    ] = run_start_time
        else:
            if include_time_interval:
                files[ftype][files_index[ftype]]["time_previous_output"] = files[ftype][
                    files_index[ftype] - 1
                ]["time"]

        if include_time_interval:
            files[ftype][files_index[ftype]]["time_interval"] = (
                current_time - files[ftype][files_index[ftype]]["time_previous_output"]
            )
        
        if include_file_contents:
            files[ftype][files_index[ftype]]["file_contents"] = {"mesh_fields" : header.nb_vars[0],
                                                                 "velocity_fields" : header.nb_vars[1],
                                                                 "pressure_fields" : header.nb_vars[2],
                                                                 "temperature_fields" : header.nb_vars[3],
                                                                 "scalar_fields" : header.nb_vars[4]}

        for key in files[ftype][files_index[ftype]].keys():
            logger.write("debug", f"{key}: {files[ftype][files_index[ftype]][key]}")

        files_index[ftype] += 1

        #if comm.Get_rank() == 0:
        #    print(
        #        "========================================================================================="
        #    )

    logger.write("info", "Check finished")

    if output_folder == "":
        output_folder = folder_path

    for file in added_files:
        logger.write("info", f"Writing {file} file index")
        logger.tic()
        if comm.Get_rank() == 0:
            with open(output_folder + os.path.basename(file).split(".")[0] + "_index.json", "w") as outfile:
                outfile.write(json.dumps(files[file], indent=4))
        comm.Barrier()
        logger.toc()

    del logger


def merge_index_files(comm, index_list="", output_fname="", sort_by_time=False):
    """
    Merge index files into one.

    Merge index files in multiple locations into one to consolidate information.

    Parameters
    ----------
    comm : MPI.COMM
        MPI communicator.
    index_list : list
        List of index files to merge.
        Providee a list with relative or absolute paths to the index files.
        Generally, provide the indices in the order you want them to merged.
        If you want to sort them by time, keep checking the options.
    output_fname : str
        Name of the output file.
        Include also the path. If not provided, the current working directory is used.
        And a default name "consolidated_index.json" is used.
    sort_by_time : bool
        Sort the index files by time. Default is False.

    Returns
    -------
    None
        A merged index file is written.
    """

    if output_fname == "":
        output_fname = os.getcwd() + "/consolidated_index.json"

    if not isinstance(index_list, list):
        raise ValueError("index_list must be a list")

    if len(index_list) == 0:
        raise ValueError("index_list must contain at least one index file")

    logger = Logger(comm=comm, module_name="merge_index_files")

    logger.write("info", f"Merging index files: {index_list}")

    consolidated_index = {}
    consolidated_index["simulation_start_time"] = 1e12
    consolidated_key = 0

    for index_file in index_list:

        try:
            with open(index_file, "r") as infile:
                index = json.load(infile)
        except FileNotFoundError:
            logger.write(
                "warning",
                f"Expected file {index_file} but it does not exist. skipping it",
            )
            continue

        logger.write("info", f"Reading index file: {index_file}")

        for key in index.keys():

            if key == "simulation_start_time":
                if index[key] < consolidated_index["simulation_start_time"]:
                    consolidated_index["simulation_start_time"] = index[key]
                continue

            elif index[key]["path"] != "file_not_in_folder":
                consolidated_index[consolidated_key] = index[key]
                consolidated_key += 1

    if sort_by_time:
        logger.write("info", "Sorting index files by time")

        unsorted_key = []
        time = []
        for key in consolidated_index.keys():
            try:
                int_key = int(key)
            except ValueError:
                continue

            unsorted_key.append(int(key))
            time.append(consolidated_index[key]["time"])

        unsorted_key = np.array(unsorted_key)

        sorted_indices = np.argsort(time)
        sorted_key = unsorted_key[sorted_indices]

        sorted_consolidated_index = {}
        sorted_consolidated_key = 0
        sorted_consolidated_index["simulation_start_time"] = consolidated_index[
            "simulation_start_time"
        ]

        for key in sorted_key:
            sorted_consolidated_index[sorted_consolidated_key] = consolidated_index[key]
            sorted_consolidated_key += 1

        consolidated_index = sorted_consolidated_index

    logger.write("info", f"Writing consolidated index file: {output_fname}")

    write_index = True
    file_exists = os.path.exists(output_fname)
    if file_exists:
        logger.write("warning", f"File {output_fname} exists. Overwrite?")
        overwrite = input("[yes/no] ")

        if overwrite == "no":
            write_index = False
            logger.write("warning", f"Skipping writing index {output_fname}")

    if write_index:
        with open(output_fname, "w") as outfile:
            outfile.write(json.dumps(consolidated_index, indent=4))

    del logger

def inspect_field_files(
    comm,
    input_name="",
    output_name="./",
):
    """
    Check the maximun, minimun and average value of the fields in the index.


    Parameters
    ----------
    comm : MPI.COMM
        MPI communicator.
    field_index_name : str
        Name of the field index file.
    output_name : str
        Name of the output file. Optional. If not provided, the name of the input file name is used
    """

    logger = Logger(comm=comm, module_name="inspect_field_files")
    input_path_ = os.path.dirname(input_name)
    input_name_ = os.path.basename(input_name)
    output_path_ = os.path.dirname(output_name)
    if output_path_ == "":
        output_path_ = os.getcwd()
    output_name_ = os.path.basename(output_name)
    if output_name_ == "":
        output_name_ = f"check_{input_name_}"
    

    logger.write("info", f"Reading index file: {input_name}")
    with open(input_name, "r") as infile:
        index = json.load(infile)

    logger.write("info", f"Checking fields in index file: {input_name}")
    for index_key in index.keys():
        try: 
            int_index = int(index_key)
        except ValueError:
            continue

        logger.write("info", f"Checking field in index position: {index_key}")
        logger.tic()

        fname = index[index_key]["path"]
        logger.write("info", f"Reading field: {fname}")

        fld = FieldRegistry(comm)
        pynekread(fname, comm, data_dtype=np.single, fld=fld)

        # Check maximun, minimun and average of the fields
        for field_key in fld.registry.keys():
            field = fld.registry[field_key]
            logger.write("info", f"Checking field: {field_key}")
            max_field = np.ones((1), dtype = np.double) * np.max(field)
            min_field = np.ones((1), dtype = np.double) * np.min(field)
            avg_field = np.ones((1), dtype = np.double) * np.mean(field)

            global_max_field = comm.allreduce(max_field, op=MPI.MAX)
            global_min_field = comm.allreduce(min_field, op=MPI.MIN)
            global_avg_field = comm.allreduce(avg_field, op=MPI.SUM) / comm.Get_size()

            logger.write("info", f"Maximun value of field: {global_max_field}")
            logger.write("info", f"Minimun value of field: {global_min_field}")
            logger.write("info", f"Unweighted Average value of field: {global_avg_field}")

            index[index_key][field_key] = {"max": global_max_field[0], "min": global_min_field[0], "unweigthed_avg": global_avg_field[0]}

        fld.clear()

        logger.write("info", f"Field in index position: {index_key} checked/finalized")
        logger.toc()

    logger.write("info", f"Writing field index: {output_path_}/{output_name_}")
    
    if comm.Get_rank() == 0:
        with open(f"{output_path_}/{output_name_}", "w") as outfile:
            outfile.write(json.dumps(index, indent=4))