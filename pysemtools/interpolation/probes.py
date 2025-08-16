"""'Contains the probes class"""

import json
import csv

try:
    import h5py
except ImportError:
    print(f"h5py not found. HDF5 files will not be supported")
import numpy as np
import os
from ..io.ppymech.neksuite import preadnek, pynekread
from .interpolator import Interpolator
from ..monitoring.logger import Logger
from typing import Union
from ..datatypes.msh import Mesh
from .utils import transform_from_array_to_list, transform_from_list_to_array

NoneType = type(None)


class Probes:
    """
    Class to interpolate fields at probes from a SEM mesh.

    Main interpolation class. This works in parallel.

    If the points to interpolate are available only at rank 0, make sure to only pass them at that rank and set the others to None.
    In that case, the points will be scattered to all ranks.

    If the points are passed in all ranks, then they will be considered as different points to be interpolated and the work to interpolate will be multiplied.
    If you are doing this, make sure that the points that each rank pass are different. Otherwise simply pass None in all ranks but 0.
    
    See example below to observe how to avoid unnecessary replication of data in all ranks when passing probes as argument if the points are the same in all ranks.

    If reading probes from file, they will be read on rank 0 and scattered unless parallel hdf5 is used, in which case the probes will be read in all ranks. (In development)

    Parameters
    ----------
    comm : MPI communicator
        MPI communicator.
    output_fname : str
        Output file name. Default is "./interpolated_fields.csv".
        Note that you can change the file extension to .hdf5 to write in hdf5 format. by using the name "interpolated_fields.hdf5".
    probes : Union[np.ndarray, str]
        Probes coordinates. If a string, it is assumed to be a file name.
    msh : Union[Mesh, str, list]
        Mesh data. If a string, it is assumed to be a file name. If it is a list
        the first entry is the file name and the second is the dtype of the data.
        if it is a Mesh object, the x, y, z coordinates are taken from the object.
    write_coords : bool
        If True, the coordinates of the probes are written to a file. Default is True.
    progress_bar : bool
        If True, a progress bar is shown. Default is False.
    point_interpolator_type : str
        Type of point interpolator. Default is single_point_legendre.
        options are: single_point_legendre, single_point_lagrange,
        multiple_point_legendre_numpy, multiple_point_legendre_torch.
    max_pts : int, optional
        Maximum number of points to interpolate. Default is 128. Used if multiple point interpolator is selected.
    find_points_iterative : list
        List with two elements. First element is a boolean that indicates if the search is iterative.
        Second element is the maximum number of candidate ranks to send the data. This affects memory. Default is [False, 5000].
    find_points_comm_pattern : str
        Communication pattern for finding points. Default is point_to_point.
        options are: point_to_point, collective.
    elem_percent_expansion : float
        Percentage expansion of the element bounding box. Default is 0.01.
    global_tree_type : str
        How is the global tree constructed to determine rank candidates for the probes.
        Only really used if using tree structures to determine candidates. Default is rank_bbox.
        options are: rank_bbox, domain_binning.
    global_tree_nbins : int
        Number of bins in the global tree. Only used if the global tree is domain_binning.
        Default is 1024.
    use_autograd : bool
        If True, autograd is used. Default is False.
    find_points_tol : float
        The tolerance to use when finding points. Default is np.finfo(np.double).eps * 10.
    find_points_max_iter : int
        The maximum number of iterations to use when finding points. Default is
        50.

    Attributes
    ----------
    probes : ndarray
        2D array of probe coordinates. shape = (n_probes, 3).
    interpolated_fields : ndarray
        2D array of interpolated fields at probes. shape = (n_probes, n_fields + 1).
        The first column is always time, the rest are the interpolated fields.

    Notes
    -----
    A sample input file can be found in the examples folder of the main repository,
    However, the file is not used in said example.

    Examples
    --------
    1. Initialize from file:

    >>> from mpi4py import MPI
    >>> from pysemtools.interpolation.probes import Probes
    >>> comm = MPI.COMM_WORLD
    >>> probes = Probes(comm, filename="path/to/params.json")

    2. Initialize from code, passing everything as arguments.
    Assume msh is created. One must then create the probe data in
    rank 0. A dummy probe_data must be created in all other ranks

    >>> from mpi4py import MPI
    >>> from pysemtools.interpolation.probes import Probes
    >>> comm = MPI.COMM_WORLD
    >>> if comm.Get_rank() == 0:
    >>>     probes_data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    >>> else:
    >>>    probes_data = None
    >>> probes = Probes(comm, probes=probes_data, msh=msh)

    Note that probes is initialized in all ranks, but the probe_data containing
    the coordinates are only relevant in rank 0. They are scattered internally.
    """

    def __init__(
        self,
        comm,
        output_fname: str = "./interpolated_fields.csv",
        probes: Union[np.ndarray, str] = None,
        msh=Union[Mesh, str, list],
        write_coords: bool = True,
        progress_bar: bool = False,
        point_interpolator_type: str = "single_point_legendre",
        max_pts: int = 128,
        find_points_iterative: list = [False, 5000],
        find_points_comm_pattern: str = "point_to_point",
        elem_percent_expansion: float = 0.01,
        global_tree_type: str = "rank_bbox",
        global_tree_nbins: int = 1024,
        use_autograd: bool = False,
        find_points_tol: float = np.finfo(np.double).eps * 10,
        find_points_max_iter: int = 50,
        local_data_structure: str = "kdtree",
        use_oriented_bbox: bool = False,
    ):

        rank = comm.Get_rank()
        self.log = Logger(comm=comm, module_name="Probes")

        self.log.tic()
        self.log.write("info", "Initializing Probes object:")
        self.log.write("info", f" ======= Settings =======")
        self.log.write("info", f"output_fname: {output_fname}")
        self.log.write("info", f"write_coords: {write_coords}")
        self.log.write("info", f"progress_bar: {progress_bar}")
        self.log.write("info", f"point_interpolator_type: {point_interpolator_type}")
        self.log.write("info", f"max_pts: {max_pts}")
        self.log.write("info", f"find_points_iterative: {find_points_iterative}")
        self.log.write("info", f"find_points_comm_pattern: {find_points_comm_pattern}")
        self.log.write("info", f"elem_percent_expansion: {elem_percent_expansion}")
        self.log.write("info", f"global_tree_type: {global_tree_type}")
        self.log.write("info", f"global_tree_nbins: {global_tree_nbins}")
        self.log.write("info", f"use_autograd: {use_autograd}")
        self.log.write("info", f"find_points_tol: {find_points_tol}")
        self.log.write("info", f"find_points_max_iter: {find_points_max_iter}")
        self.log.write("info", f"local_data_structure: {local_data_structure}")
        self.log.write("info", f"use_oriented_bbox: {use_oriented_bbox}")
        self.log.write("info", f" ========================")

        # Assign probes
        self.log.sync_tic()
        self.data_read_from_structured_mesh = False
        if isinstance(probes, np.ndarray) or isinstance(probes, NoneType):
            self.log.write("info", "Probes provided as keyword argument")
            self.probes = probes
        elif isinstance(probes, str):
            self.log.write("info", f"Reading probes from {probes}")
            self.probes = read_probes(self, comm, probes)
        else:
            print(
                "ERROR: Probes must be provided as a string, numpy array or None if the probes are not distributed"
            )
            comm.Abort(1)
        self.log.sync_toc(message="Query points (probes) read")

        # Check if the probes are distributed
        self.distributed_probes = False
        if isinstance(probes, np.ndarray):
            this_rank_has_probes_flag = 1
        else:
            this_rank_has_probes_flag = 0

        flag_from_all_ranks = np.zeros((comm.Get_size()), dtype=np.int64)
        flag_from_this_rank = np.ones((1), dtype=np.int64) * this_rank_has_probes_flag
        comm.Allgather(flag_from_this_rank, flag_from_all_ranks)

        if np.all(flag_from_all_ranks) == 1 and comm.Get_size() > 1:
            self.distributed_probes = True

        self.log.write(
            "info",
            f"Input probes are distributed: {self.distributed_probes}",
        )
        if self.distributed_probes:
            self.log.write(
                "warning",
                "Probes are distributed, all the points input to all ranks will be processed",
            )
            self.log.write(
                "warning",
                "If you are passing the same points in all ranks, you are replicating data AND interpolating the same point multiple times",
            )
            self.log.write(
                "warning",
                "If all ranks have the same points, then pass probes=None in all ranks but 0",
            )

        # Assign mesh data
        if isinstance(msh, Mesh):
            self.log.write("info", "Mesh provided as keyword argument")
            if msh.bckend != 'numpy':
                raise ValueError("Only supported Mesh backend at the moment is numpy")
            self.x = msh.x
            self.y = msh.y
            self.z = msh.z
        elif isinstance(msh, str):
            self.log.write("info", f"Reading mesh from {msh}")
            self.x, self.y, self.z = read_mesh(comm, msh)
        elif isinstance(msh, list):
            fname = msh[0]
            if len(msh) == 1:
                dtype = np.single
            else:
                dtype = msh[1]
            self.log.write("info", f"Reading mesh from {fname} with dtype {dtype}")
            self.x, self.y, self.z = read_mesh(comm, fname, dtype=dtype)
        else:
            raise ValueError("msh must be provided as argument")

        # Initialize the interpolator
        self.log.write("info", "Initializing interpolator")
        self.log.sync_tic()
        self.itp = Interpolator(
            self.x,
            self.y,
            self.z,
            self.probes,
            comm,
            progress_bar,
            point_interpolator_type=point_interpolator_type,
            max_pts=max_pts,
            use_autograd=use_autograd,
            local_data_structure=local_data_structure,
        )

        # Set up the global tree
        self.log.write("info", "Setting up global tree")
        self.itp.set_up_global_tree(
            comm,
            find_points_comm_pattern=find_points_comm_pattern,
            global_tree_type=global_tree_type,
            global_tree_nbins=global_tree_nbins,
        )
        self.log.sync_toc(message="Interpolator initialized in all ranks")

        # Scatter the probes to all ranks
        if self.distributed_probes:
            self.log.write("info", "Assigning input probes to be a probe partition")
            self.itp.assign_local_probe_partitions()
        else:
            self.log.write("info", "Scattering probes to all ranks")
            self.itp.scatter_probes_from_io_rank(0, comm)

        # Find where the point in each rank should be
        self.log.write("info", "Finding points")
        self.log.sync_tic()
        self.itp.find_points(
            comm,
            find_points_iterative=find_points_iterative,
            find_points_comm_pattern=find_points_comm_pattern,
            elem_percent_expansion=elem_percent_expansion,
            tol=find_points_tol,
            max_iter=find_points_max_iter,
            local_data_structure=local_data_structure,
            use_oriented_bbox=use_oriented_bbox
        )
        self.log.sync_toc(message="Finding points finalized in all ranks")

        # Send points to the owners
        if self.distributed_probes:
            self.log.write("info", "Redistributing probes to found owners")
            self.itp.redistribute_probes_to_owners()
        else:
            # Gather probes to rank 0 again
            self.log.write("info", "Gathering probes to rank 0 after search")
            self.itp.gather_probes_to_io_rank(0, comm)

            # Redistribute the points
            self.log.write("info", "Redistributing probes to found owners")
            self.itp.redistribute_probes_to_owners_from_io_rank(0, comm)

        self.output_fname = output_fname
        if write_coords:

            # Write the coordinates to a file
            if not self.distributed_probes:
                if comm.Get_rank() == 0:
                    write_coordinates(self, parallel=False)
            else:
                write_coordinates(self, parallel=True)

            # Write the points with warnings
            if not self.distributed_probes:
                if comm.Get_rank() == 0:
                    write_warnings(self, parallel=False)
            else:
                write_warnings(self, parallel=True)
        elif (not self.distributed_probes) and (not write_coords):        
            if comm.Get_rank() == 0:
                nfound = len(np.where(self.itp.err_code == 1)[0])
                nnotfound = len(np.where(self.itp.err_code == 0)[0])
                nwarning = len(np.where(self.itp.err_code == -10)[0])
                self.log.write(
                    "info",
                    f"Found {nfound} points, {nnotfound} not found, {nwarning} with warnings",
                )
        elif self.distributed_probes and (not write_coords):
            nfound = len(np.where(self.itp.err_code == 1)[0])
            nnotfound = len(np.where(self.itp.err_code == 0)[0])
            nwarning = len(np.where(self.itp.err_code == -10)[0])
            self.log.write(
                "info_all",
                f"Found {nfound} points, {nnotfound} not found, {nwarning} with warnings",
            )

        ## init dummy variables
        self.fld_data = None
        self.list_of_fields = None
        self.list_of_qoi = None
        self.number_of_files = None
        self.number_of_fields = None
        self.my_interpolated_fields = None
        self.interpolated_fields = None
        self.written_file_counter = 0

        self.log.write("info", "Probes object initialized")
        self.log.toc()

    def interpolate_from_field_list(
        self, t, field_list, comm, write_data=True, field_names: list[str] = None
    ):
        """
        Interpolate the probes from a list of fields.

        This method interpolates from a list of fields (ndarrays of shape (nelv, lz, ly, lx)).

        Parameters
        ----------
        t : float
            Time of the field data.

        field_list : list
            List of fields to interpolate. Each field is an ndarray of shape (nelv, lz, ly, lx).

        comm : Comm
            MPI communicator.

        write_data : bool
            If True, the interpolated data is written to a file. Default is True.

        field_names : list
            List of names of the interpolated fields. Useful when writing to file. Default is None.

        Examples
        --------
        This method can be used to interpolate fields from a list of fields. If you have
        previosly obtained a set of fields u,v,w as ndarrays of shape (nelv, lz, ly, lx), you can:

        >>> probes.interpolate_from_field_list(t, [u,v,w], comm)

        The results are stored in probes.interpolated_fields attribute.
        Remember: the first column of this attribute is always the time t given.
        """

        self.log.write("info", "Interpolating fields from field list")
        self.log.sync_tic()

        self.number_of_fields = len(field_list)

        # If the probes are distributed, each rank interpolate the point that it owns physically,
        # and the sends the data to the rank were the probe was given as an input
        if self.distributed_probes:

            # Allocate buffer that keeps the interpolated fields of the points that were input in this rank
            self.interpolated_fields = np.zeros(
                (self.probes.shape[0], self.number_of_fields + 1), dtype=np.double
            )

            # Allocate buffer to keeps the interpolated fields of the points that were sent by other ranks
            my_interpolated_fields = []
            for i in range(0, len(self.itp.my_probes)):
                my_interpolated_fields.append(
                    np.zeros(
                        (self.itp.my_probes[i].shape[0], self.number_of_fields + 1),
                        dtype=np.double,
                    )
                )
                my_interpolated_fields[i][:, 0] = t

            # Interpolate each of the fields in the list for all sources
            for i in range(0, self.number_of_fields):

                field = field_list[i]

                self.log.write("info", f"Interpolating field {i}")

                interpolated_fields_from_sources = self.itp.interpolate_field_from_rst(
                    field
                )[:]

                # Put this interpolated field on the right place in the buffer for each rank that sent me data
                for j in range(0, len(self.itp.my_probes)):
                    my_interpolated_fields[j][:, i + 1] = (
                        interpolated_fields_from_sources[j]
                    )

            self.log.sync_toc(message="Interpolation: point evaluation done")
            self.log.sync_tic(id=1)

            # Send the data back to the ranks that sent me the probes
            sources, interpolated_data = self.itp.rt.send_recv(
                destination=self.itp.my_sources,
                data=my_interpolated_fields,
                dtype=np.double,
                tag = 1,
            )
            # reshape the data
            for i in range(0, len(sources)):
                interpolated_data[i] = interpolated_data[i].reshape(
                    -1, self.number_of_fields + 1
                )

            for i in range(0, len(sources)):
                self.interpolated_fields[
                    self.itp.local_probe_index_sent_to_destination[i]
                ] = interpolated_data[list(sources).index(self.itp.destinations[i])][:]
            
            self.log.sync_toc(id = 1, message="Interpolation: point redistribution done")
            self.log.sync_toc(message="Finished interpolation and redistribution in all ranks", time_message="Aggregated time: ")

        # If the probes were given in rank 0, then each rank interpolates the points that it owns physically
        # and then send them to rank 0 to be processed further
        else:

            # Allocate interpolated fields
            self.my_interpolated_fields = np.zeros(
                (self.itp.my_probes.shape[0], self.number_of_fields + 1),
                dtype=np.double,
            )
            if comm.Get_rank() == 0:
                self.interpolated_fields = np.zeros(
                    (self.probes.shape[0], self.number_of_fields + 1), dtype=np.double
                )
            else:
                self.interpolated_fields = None

            # Set the time
            self.my_interpolated_fields[:, 0] = t

            i = 0
            for field in field_list:

                self.log.write("info", f"Interpolating field {i}")
                self.my_interpolated_fields[:, i + 1] = (
                    self.itp.interpolate_field_from_rst(field)[:]
                )

                i += 1

            self.log.sync_toc(message="Interpolation: point evaluation done")
            self.log.sync_tic(id=1)

            # Gather in rank zero for processing
            root = 0
            sendbuf = self.my_interpolated_fields.reshape(
                (self.my_interpolated_fields.size)
            )
            recvbuf, _ = self.itp.rt.gather_in_root(sendbuf, root, np.double)

            if not isinstance(recvbuf, NoneType):
                tmp = recvbuf.reshape(
                    (
                        int(recvbuf.size / (self.number_of_fields + 1)),
                        self.number_of_fields + 1,
                    )
                )
                # IMPORTANT - After gathering, remember to sort to the way that it should
                #  be in rank zero to write values out
                # (See the scattering routine if you forget this)
                # The reason for this is that to scatter we need the data to be contigous.
                # You sort to make sure that the data from each rank is grouped.
                self.interpolated_fields[self.itp.sort_by_rank] = tmp
            
            self.log.sync_toc(id = 1, message="Interpolation: point redistribution done")
            self.log.sync_toc(message="Finished interpolation and redistribution in all ranks", time_message="Aggregated time: ")

        # Write the data
        self.log.sync_tic()
        if write_data:
            # Define the name of the interpolated fields
            if field_names is None:
                self.field_names = [f"field_{i}" for i in range(self.number_of_fields)]
            else:
                self.field_names = field_names

            self.written_file_counter += 1

            if not self.distributed_probes:
                if comm.Get_rank() == 0:
                    write_interpolated_data(self, parallel=False)
            else:
                write_interpolated_data(self, parallel=True)
        self.log.sync_toc(message="Finished writing interpolated fields")


def write_coordinates(self, parallel=False):

    # Write the coordinates
    if self.output_fname.split(".")[-1] == "csv":
        write_coordinates_csv(self, parallel)
    elif self.output_fname.split(".")[-1] == "hdf5":
        write_coordinates_hdf5(self, parallel)
    else:
        raise ValueError("Output file must be a csv or hdf5 file")


def write_interpolated_data(self, parallel=False):

    if self.output_fname.split(".")[-1] == "csv":
        write_interpolated_data_csv(self, parallel)
    elif self.output_fname.split(".")[-1] == "hdf5":
        write_interpolated_data_hdf5(self, parallel)
    else:
        raise ValueError("Output file must be a csv or hdf5 file")


def write_warnings(self, parallel=False):

    # Write out a file with the points with warnings
    indices = np.where(self.itp.err_code != 1)[0]

    if len(indices) == 0:
        self.log.write("info", "All points were found without warnings")
        return

    # Write the points with warnings in a json file
    point_warning = {}
    # point_warning["mesh_file_path"] = msh_fld_fname
    point_warning["output_file_path"] = self.output_fname
    i = 0
    for point in indices:
        point_warning[i] = {}
        point_warning[i]["id"] = int(point)
        point_warning[i]["xyz"] = [
            float(self.itp.probes[point, 0]),
            float(self.itp.probes[point, 1]),
            float(self.itp.probes[point, 2]),
        ]
        point_warning[i]["rst"] = [
            float(self.itp.probes_rst[point, 0]),
            float(self.itp.probes_rst[point, 1]),
            float(self.itp.probes_rst[point, 2]),
        ]
        # point_warning[i]["local_el_owner"] = int(self.itp.el_owner[point])
        point_warning[i]["global_el_owner"] = int(self.itp.glb_el_owner[point])
        point_warning[i]["error_code"] = int(self.itp.err_code[point])
        point_warning[i]["test_pattern"] = float(self.itp.test_pattern[point])

        i += 1
    path = os.path.dirname(self.output_fname)
    if path == "":
        path = "."
    fname = os.path.basename(self.output_fname)
    fname = fname.split(".")[0]

    if not parallel:
        params_file_str = json.dumps(point_warning, indent=4)
        json_output_fname = f"{path}/warning_points_{fname}.json"
    else:
        json_output_fname = (
            f"{path}/warning_points_rank_{self.itp.rt.comm.Get_rank()}_{fname}.json"
        )
        params_file_str = json.dumps(point_warning, indent=4)

    self.log.write(
        "info",
        "Writing points with warnings to {}".format(json_output_fname),
    )
    with open(json_output_fname, "w") as outfile:
        outfile.write(params_file_str)

    nfound = len(np.where(self.itp.err_code == 1)[0])
    nnotfound = len(np.where(self.itp.err_code == 0)[0])
    nwarning = len(np.where(self.itp.err_code == -10)[0])
    self.log.write(
        "info",
        f"Found {nfound} points, {nnotfound} not found, {nwarning} with warnings",
    )

    if nwarning > 0:
        self.log.write(
            "warning",
            "There are points with warnings. Check the warning file to see them (error code -10)",
        )
        self.log.write(
            "warning",
            "There are points with warnings. If test pattern is small, you can trust the interpolation",
        )

    if nnotfound > 0:
        self.log.write(
            "error",
            "Some points were not found. Check the warning file to see them (error code 0)",
        )
        self.log.write(
            "error",
            "Some points were not found. The result from their interpolation will be 0",
        )


def write_coordinates_csv(self, parallel=True):

    # Write the coordinates
    ## Set up the header
    field_type_list = None
    field_index_list = None

    ## Create the header
    if isinstance(field_type_list, NoneType) or isinstance(field_index_list, NoneType):
        header = [self.probes.shape[0], 0, 0]
    else:
        header = [self.probes.shape[0], len(field_type_list)]
        for i in range(len(field_type_list)):
            header.append(f"{field_type_list[i]}{field_index_list[i]}")

    ## Write the coordinates
    self.log.write("info", "Writing probe coordinates to {}".format(self.output_fname))

    if not parallel:
        write_csv(self.output_fname, self.probes, "w", header=header)
    else:
        path = os.path.dirname(self.output_fname)
        fname = os.path.basename(self.output_fname)
        write_csv(
            f"{path}/rank_{self.itp.rt.comm.Get_rank()}_{fname}",
            self.itp.probes,
            "w",
            header=header,
        )


def write_coordinates_hdf5(self, parallel=True):

    ## Write the coordinates
    self.log.write("info", "Writing probe coordinates to {}".format(self.output_fname))

    path = os.path.dirname(self.output_fname)
    if path == "":
        path = "."
    fname = os.path.basename(self.output_fname)
    fname = f"coordinates_{fname}"
    if not parallel:
        with h5py.File(f"{path}/{fname}", "w") as f:
            if self.data_read_from_structured_mesh:
                coord_list = transform_from_list_to_array(
                    self.input_nx, self.input_ny, self.input_nz, self.probes
                )
                f.create_dataset("x", data=coord_list[0])
                f.create_dataset("y", data=coord_list[1])
                f.create_dataset("z", data=coord_list[2])
            else:
                f.create_dataset("xyz", data=self.probes)
    else:
        with h5py.File(f"{path}/rank_{self.itp.rt.comm.Get_rank()}_{fname}", "w") as f:
            if self.data_read_from_structured_mesh:
                coord_list = transform_from_list_to_array(
                    self.input_nx, self.input_ny, self.input_nz, self.itp.probes
                )
                f.create_dataset("x", data=coord_list[0])
                f.create_dataset("y", data=coord_list[1])
                f.create_dataset("z", data=coord_list[2])
            else:
                f.create_dataset("xyz", data=self.itp.probes)


def write_interpolated_data_csv(self, parallel=True):

    if not parallel:

        self.log.write("info", f"Writing interpolated fields to {self.output_fname}")
        write_csv(self.output_fname, self.interpolated_fields, "a")

    else:
        path = os.path.dirname(self.output_fname)
        fname = os.path.basename(self.output_fname)
        write_csv(
            f"{path}/rank_{self.itp.rt.comm.Get_rank()}_{fname}",
            self.interpolated_fields,
            "a",
        )


def write_interpolated_data_hdf5(self, parallel=True):

    path = os.path.dirname(self.output_fname)
    if path == "":
        path = "."
    fname = os.path.basename(self.output_fname)
    fname = fname.split(".")[0]
    if not parallel:
        fname = f"{fname}{str(self.written_file_counter).zfill(5)}.hdf5"
    else:
        fname = f"rank_{self.itp.rt.comm.Get_rank()}_{fname}{str(self.written_file_counter).zfill(5)}.hdf5"

    self.log.write("info", f"Writing interpolated fields to {path}/{fname}")

    with h5py.File(f"{path}/{fname}", "w") as f:

        if self.data_read_from_structured_mesh:

            field_list = transform_from_list_to_array(
                self.input_nx, self.input_ny, self.input_nz, self.interpolated_fields
            )

            for i in range(len(field_list)):
                if i == 0:
                    f.attrs["time"] = self.interpolated_fields[0, i]
                else:
                    f.create_dataset(f"{self.field_names[i-1]}", data=field_list[i])
        else:
            for i in range(self.interpolated_fields.shape[1]):
                if i == 0:
                    f.attrs["time"] = self.interpolated_fields[0, i]
                else:
                    f.create_dataset(
                        f"{self.field_names[i-1]}", data=self.interpolated_fields[:, i]
                    )


def write_csv(fname, data, mode, header=None):
    """write point positions to the file

    Parameters
    ----------
    fname :

    data :

    mode :


    Returns
    -------

    """

    # string = "writing .csv file as " + fname
    # print(string)

    # open file
    outfile = open(fname, mode)

    writer = csv.writer(outfile)

    if not isinstance(header, NoneType):
        writer.writerow(header)

    for il in range(data.shape[0]):
        data_pos = data[il, :]
        writer.writerow(data_pos)


def read_probes(self, comm, fname):

    if fname.split(".")[-1] == "csv":
        # For csv files, read in rank 0 only
        if comm.Get_rank() == 0:
            probes = read_probes_csv(self, fname)
        else:
            probes = None
    elif fname.split(".")[-1] == "hdf5":
        # For hdf5 initially add support for reading in rank 0 only
        if comm.Get_rank() == 0:
            probes = read_probes_hdf5(self, fname)
        else:
            probes = None
    else:
        raise ValueError("Probes must be a csv or hdf5 file")

    return probes


def read_probes_csv(self, fname):
    file = open(fname)
    probes = np.array(list(csv.reader(file)), dtype=np.double)
    return probes


def read_probes_hdf5(self, fname):
    with h5py.File(fname, "r") as f:

        # Check if there is a key that indicates which are the probes
        probe_list_key = f.attrs.get("probe_list_key", "xyz")

        # Check if the data exists
        if probe_list_key in f:
            probes = f[probe_list_key][:]
        else:
            self.log.write("warning", f"Warning: {probe_list_key} not found in {fname}")
            self.log.write("warning", f"Attemping to use x,y,z coordinates")
            try:
                x = f["x"][:]
                y = f["y"][:]
                z = f["z"][:]
            except KeyError:
                raise KeyError("x, y, z coordinates not found in the file")
            nx = x.shape[0]
            ny = x.shape[1]
            nz = x.shape[2]
            probes = transform_from_array_to_list(nx, ny, nz, [x, y, z])

            # Store information on how the data was read to be able to use it when writing
            self.data_read_from_structured_mesh = True
            self.input_nx = nx
            self.input_ny = ny
            self.input_nz = nz

    return probes


def read_mesh(comm, fname, dtype=np.single):

    msh = Mesh(comm, create_connectivity=False)
    pynekread(fname, comm, data_dtype=dtype, msh=msh)

    return msh.x, msh.y, msh.z
