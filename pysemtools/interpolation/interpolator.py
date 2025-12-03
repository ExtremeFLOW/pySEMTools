""" Contains the interpolator class"""

import os
import sys
from mpi4py import MPI  # for the timer
from itertools import combinations
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from .point_interpolator.point_interpolator_factory import get_point_interpolator
from ..monitoring.logger import Logger
from ..comm.router import Router
from collections import Counter as collections_counter
import threading
import time
import ctypes
import gc

# check if we have rtree installed
try :
    from rtree import index as rtree_index
except ImportError:
    rtree_index = None

NoneType = type(None)

DEBUG = os.getenv("PYSEMTOOLS_DEBUG", "False").lower() in ("true", "1", "t")
INTERPOLATION_LOG_TIME = int(os.getenv("PYSEMTOOLS_INTERPOLATION_LOG_TIME", "60"))
INTERPOLATION_MAX_MESSAGE_SIZE = int(os.getenv("PYSEMTOOLS_INTERPOLATION_MAX_MESSAGE_SIZE", int(100))) # 100 MB seems to be fine
INTERPOLATION_MAX_CANDIDATE_IN_ITERATION = int(os.getenv("PYSEMTOOLS_INTERPOLATION_MAX_CANDIDATE_IN_ITERATION", np.iinfo(np.int32).max)) #256 is a good balance. It is very high for speed
INTERPOLATION_DTYPE = np.double if os.getenv("PYSEMTOOLS_INTERPOLATION_DTYPE", "double").lower() == "double" else np.single

class RMAWindow:
    '''
    Wrapper class to handle one-sided communication
    '''

    def __init__(self,
                 comm: MPI.Comm,
                 window_members: dict = None):
        
        self.comm = comm
        self.rank = comm.Get_rank()
        self._mpi_ptr = None
        self._lock = threading.Lock()

        # Update the submember info and get the total window size
        self.size_bytes = 0
        for member in window_members.keys():
            window_members[member]["dtype"] = np.dtype(window_members[member]["dtype"])
            window_members[member]["window_size"] = int(window_members[member]["window_size"])
            window_members[member]["itemsize"] = window_members[member]["dtype"].itemsize
            window_members[member]["size_bytes"] = window_members[member]["itemsize"] * window_members[member]["window_size"]
            window_members[member]["start_offset"] = np.copy(self.size_bytes)
            window_members[member]["fill_value"] = window_members[member].get("fill_value", None)
            self.size_bytes += window_members[member]["size_bytes"]
 
        # Allocate memory via MPI - Trying to avoid any memory address replication from numpy/python
        self._mpi_ptr = MPI.Alloc_mem(self.size_bytes, MPI.INFO_NULL)

        # Wrap pointer into ctypes buffer so it can be viewed as numpy
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mpi_ptr))
        buf_type = ctypes.c_char * self.size_bytes
        raw_buf = buf_type.from_address(addr)
        self.byte_arr = np.frombuffer(raw_buf, dtype=np.uint8)

        # Create MPI window over the allocated memory
        self.win = MPI.Win.Create(self._mpi_ptr, 1, MPI.INFO_NULL, self.comm)

        # Initialize each sub_window member:
        ## Set some default names just to help linter
        self.search_done: RMASubWindow | None = None
        self.find_busy: RMASubWindow | None = None
        self.find_done: RMASubWindow | None = None
        self.find_n_not_found: RMASubWindow | None = None
        self.find_p_probes : RMASubWindow | None = None
        self.find_p_info: RMASubWindow | None = None
        self.find_test_pattern: RMASubWindow | None = None
        self.verify_busy: RMASubWindow | None = None
        self.verify_done: RMASubWindow | None = None
        self.verify_n_not_found: RMASubWindow | None = None
        self.verify_p_probes : RMASubWindow | None = None
        self.verify_p_info: RMASubWindow | None = None
        self.verify_test_pattern: RMASubWindow | None = None 
        ## Initialize
        for member in window_members.keys():
            self.__setattr__(member, RMASubWindow(self, **window_members[member]))

class RMASubWindow:

    def __init__(self, parent_window: RMAWindow, dtype: np.dtype = None, window_size: int = None, itemsize: int = None, size_bytes: int = None, start_offset: int = None, fill_value=None):

        # Store my own data
        self.dtype = dtype
        self.window_size = window_size
        self.itemsize = itemsize
        self.size_bytes = size_bytes
        self.start_offset = start_offset
        self.fill_value = fill_value
        self.rank = parent_window.rank
        self.comm = parent_window.comm
        
        # Get my buffer slice from the parent window    
        start = self.start_offset
        end = start + self.size_bytes
        self.buff = parent_window.byte_arr[start:end].view(self.dtype)
        if self.fill_value is not None:
            self.buff[:] = self.fill_value

        # Store a reference to the parent window proper
        self.win = parent_window.win

        # Store a lock object
        self._lock = threading.Lock()

    def compare_and_swap(self, value_to_check = None, value_to_put = None, dest: int = None):
        '''Wrap around compare and swap to do atomic operations on the buffer.
        
        Parameters:
        -----------
        value_to_check : any
        value_to_put : any
        dest : int

        Returns:
        --------
        result : any
            The value that was in the buffer before the operation.
    
        '''

        expected = np.array([value_to_check], dtype=self.buff.dtype)
        origin = np.array([value_to_put], dtype=self.buff.dtype)
        result = np.empty(1, dtype=self.buff.dtype)

        self.win.Lock(dest, MPI.LOCK_EXCLUSIVE)
        self.win.Compare_and_swap(origin, expected, result, dest, target_disp=self.start_offset)
        self.win.Flush(dest)
        self.win.Unlock(dest)

        return result[0]

    def put(self, dest: int = None, data = None, dtype: np.dtype =None, displacement: int = 0, lock: bool = True, unlock: bool = True, flush: bool = True, mask: np.ndarray = None):

        '''Put data into the buffer at the specified displacement.
        
        Parameters:
        -----------
        dest : int
            The destination rank to put the data into.
        data : np.ndarray or any
            The data to put into the buffer. If not an array, it will be converted to one based on the dtype.
        dtype : np.dtype
            The data type of the data to put into the buffer. Required if data is not an array.
        displacement : int
            The displacement in the buffer where the data should be put.

        Returns:
        --------
        None
            
        '''

        if not isinstance(data, np.ndarray):
            if dtype is None:
                raise TypeError("If passing an individual value, dtype must be specified")
            else:
                data = np.array([data], dtype=dtype)

        if data.dtype != self.buff.dtype:
            raise ValueError("Data to put and local buffer must have the same item size (same dtype)")

        # Send data if no mask is given 
        mask_size = mask.size if mask is not None else 0
        if mask is None or mask_size == 0:
            byte_displacement = displacement * self.itemsize + self.start_offset
            byte_count = data.size * self.itemsize
            if lock: self.win.Lock(dest, MPI.LOCK_EXCLUSIVE)
            self.win.Put([data.view(np.uint8), MPI.BYTE], dest, [byte_displacement, byte_count, MPI.BYTE]) # Just send bytes
            if flush: self.win.Flush(dest)
            if unlock: self.win.Unlock(dest)

        # Send masked data if needed without duplicating data
        elif isinstance(mask, np.ndarray):

            # Determine the masked indices
            idx = np.flatnonzero(mask)
            strides = np.array(data.strides, dtype = np.intp)
            row_size = idx.shape[0]
            
            # Perform a chunked send over the mask to avoid memory issues
            if lock: self.win.Lock(dest, MPI.LOCK_EXCLUSIVE)

            max_message_size = INTERPOLATION_MAX_MESSAGE_SIZE * 1024 * 1024 # 100MB
            bytes_per_row = strides[0]
            row_chunk_size = max_message_size // strides[0]
            byte_displacement = displacement * self.itemsize + self.start_offset
            for row_chunk in range(0, row_size, row_chunk_size):
                row_ids = idx[row_chunk:row_chunk + row_chunk_size]
                row_displacements = row_ids * strides[0]
                # Create a data type with len(row_displacements) entries long and bytes_per_row in each entry. The data is taken from the original input at the given displacement
                row_dtype =  MPI.BYTE.Create_hindexed_block(blocklength=bytes_per_row, displacements=row_displacements)
                row_dtype.Commit()
                # Send
                byte_count = row_ids.size * bytes_per_row
                data_ = data.view()
                try:
                    data_.shape = data.size
                except AttributeError:
                    raise AttributeError("Error while reshaping the data to send. Make sure your probes are C contiguous")
                self.win.Put([data_.view(np.uint8), 1, row_dtype], dest, [byte_displacement, byte_count, MPI.BYTE]) # Just send bytes
                row_dtype.Free()
                byte_displacement += byte_count

            if flush: self.win.Flush(dest)
            if unlock: self.win.Unlock(dest)
    
    def put_sequence(self, dest: int = None, data : list = None, dtype: np.dtype =None, displacement: int = 0, lock: bool = True, unlock: bool = True, flush: bool = True, mask: np.ndarray = None):

        for idx in range(0, len(data)): 
            self.put(dest, data[idx], dtype=dtype, displacement=displacement, lock=lock, unlock=unlock, flush=flush, mask=mask)
            if mask is None:
                displacement += int(data[idx].size)
            else:
                displacement += int(np.flatnonzero(mask).size * data[idx].strides[0] / data[idx].itemsize) # This way since we only mask rows

    def get(self, source: int = None, displacement: int = 0, counts: int = None, lock: bool = True, unlock: bool = True, flush: bool = True):

        '''Get data from the buffer at the specified displacement.

        Parameters:
        -----------
        source : int
            The source rank to get the data from.
        displacement : int
            The displacement in the buffer where the data should be gotten.
        counts : int
            The number of elements to get from the buffer.

        Returns:
        --------
        data : np.ndarray
            The data gotten from the buffer.

        '''

        if counts is None:
            counts = self.buff.size - displacement
        elif counts == -1:
            counts = self.buff.size - displacement
        elif counts > self.buff.size - displacement:
            raise ValueError("Counts cannot be greater than the size of the buffer minus the displacement")


        byte_displacement = displacement * self.itemsize + self.start_offset
        byte_count = counts * self.itemsize
        data = np.empty(byte_count, dtype=np.uint8)
        if lock: self.win.Lock(source, MPI.LOCK_EXCLUSIVE)
        self.win.Get([data, MPI.BYTE], source, [byte_displacement, byte_count, MPI.BYTE])
        if flush: self.win.Flush(source)
        if unlock: self.win.Unlock(source)
        return data.view(self.dtype)

class Interpolator:
    """Class that interpolates data from a SEM mesh into a series of points"""

    def __init__(
        self,
        x,
        y,
        z,
        probes,
        comm,
        progress_bar=False,
        point_interpolator_type="single_point_legendre",
        max_pts=128,
        max_elems=1,
        use_autograd=False,
        local_data_structure="kdtree",
    ):
        self.log = Logger(comm=comm, module_name="Interpolator")
        self.log.write("info", "Initializing Interpolator object")
        self.log.tic()

        # Instance communication object
        self.rt = Router(comm)

        self.x = x
        self.y = y
        self.z = z
        if probes is not None:
            if probes.dtype == INTERPOLATION_DTYPE:
                self.probes = probes
            else:
                self.probes = probes.astype(INTERPOLATION_DTYPE)
        else:
            self.probes = None

        self.point_interpolator_type = point_interpolator_type
        self.max_pts = max_pts
        self.max_elems = max_elems

        self.local_data_structure = local_data_structure

        self.interpolator_dtype = INTERPOLATION_DTYPE

        # Determine which point interpolator to use
        self.log.write(
            "info",
            "Initializing point interpolator: {}".format(point_interpolator_type),
        )
        self.ei = get_point_interpolator(
            point_interpolator_type,
            x.shape[1],
            max_pts=max_pts,
            max_elems=max_elems,
            use_autograd=use_autograd,
        )

        # Determine which buffer to use
        self.log.write("info", "Allocating buffers in point interpolator")
        self.r = self.ei.alloc_result_buffer(dtype="double")
        self.s = self.ei.alloc_result_buffer(dtype="double")
        self.t = self.ei.alloc_result_buffer(dtype="double")
        self.test_interp = self.ei.alloc_result_buffer(dtype="double")


        # Print what you are using
        try:
            dev = self.r.device
        except AttributeError:
            dev = "cpu"

        self.log.write("info", f"Using device: {dev}")

        self.progress_bar = progress_bar

        # Find the element offset of each rank so you can store the global element number
        nelv = self.x.shape[0]
        sendbuf = np.ones((1), np.int64) * nelv
        recvbuf = np.zeros((1), np.int64)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - nelv

        # define dummy varaibles
        self.probes_rst = None
        self.el_owner = None
        self.glb_el_owner = None
        self.rank_owner = None
        self.err_code = None
        self.test_pattern = None
        self.probe_partition_sendcount = None
        self.probe_coord_partition_sendcount = None
        self.probe_partition = None
        self.probe_rst_partition = None
        self.el_owner_partition = None
        self.glb_el_owner_partition = None
        self.rank_owner_partition = None
        self.err_code_partition = None
        self.test_pattern_partition = None
        self.rank = None
        self.my_probes = None
        self.my_probes_rst = None
        self.my_el_owner = None
        self.my_rank_owner = None
        self.my_err_code = None
        self.my_bbox = None
        self.my_bbox_centroids = None
        self.my_bbox_maxdist = None
        self.my_tree = None
        self.ranks_ive_checked = None
        self.size = None
        self.nelv = None
        self.sendcounts = None
        self.sort_by_rank = None
        self.global_tree = None
        self.global_tree_type = None
        self.search_radious = None
        self.bin_to_rank_map = None

        self.log.write("info", "Interpolator initialized")
        self.log.toc()

    def set_up_global_tree(
        self,
        comm,
        find_points_comm_pattern="point_to_point",
        global_tree_type="rank_bbox",
        global_tree_nbins=None,
    ):

        if find_points_comm_pattern == "broadcast":
            self.log.write(
                "info", "Communication pattern selected does not need global tree"
            )

        elif (find_points_comm_pattern == "point_to_point") or (find_points_comm_pattern == "collective") or (find_points_comm_pattern == "rma"):
            self.global_tree_type = global_tree_type
            self.log.write("info", f"Using global_tree of type: {global_tree_type}")
            if global_tree_type == "rank_bbox":
                self.set_up_global_tree_rank_bbox_(
                    comm, global_tree_nbins=global_tree_nbins
                )
            elif global_tree_type == "domain_binning":
                self.set_up_global_tree_domain_binning_(
                    comm, global_tree_nbins=global_tree_nbins
                )

    def set_up_global_tree_rank_bbox_(self, comm, global_tree_nbins=None):

        self.log.tic()

        size = comm.Get_size()

        # Find the bounding box of the rank to create a global but "sparse" kdtree
        self.log.write("info", "Finding bounding boxes for each rank")
        rank_bbox = np.zeros((1, 6), dtype=INTERPOLATION_DTYPE)
        rank_bbox[0, 0] = np.min(self.x)
        rank_bbox[0, 1] = np.max(self.x)
        rank_bbox[0, 2] = np.min(self.y)
        rank_bbox[0, 3] = np.max(self.y)
        rank_bbox[0, 4] = np.min(self.z)
        rank_bbox[0, 5] = np.max(self.z)

        # Gather the bounding boxes in all ranks
        self.global_bbox = np.zeros((size * 6), dtype=INTERPOLATION_DTYPE)
        comm.Allgather(
            [rank_bbox.flatten(), MPI.DOUBLE], [self.global_bbox, MPI.DOUBLE]
        )
        self.global_bbox = self.global_bbox.reshape((size, 6))

        # Get the centroids and max distances
        bbox_dist = np.zeros((size, 3), dtype=INTERPOLATION_DTYPE)
        bbox_dist[:, 0] = self.global_bbox[:, 1] - self.global_bbox[:, 0]
        bbox_dist[:, 1] = self.global_bbox[:, 3] - self.global_bbox[:, 2]
        bbox_dist[:, 2] = self.global_bbox[:, 5] - self.global_bbox[:, 4]
        bbox_max_dist = np.max(
            np.sqrt(
                bbox_dist[:, 0] ** 2
                + bbox_dist[:, 1] ** 2
                + bbox_dist[:, 2] ** 2
            )
            / 2
        )

        bbox_centroid = np.zeros((size, 3), dtype=INTERPOLATION_DTYPE)
        bbox_centroid[:, 0] = self.global_bbox[:, 0] + bbox_dist[:, 0] / 2
        bbox_centroid[:, 1] = self.global_bbox[:, 2] + bbox_dist[:, 1] / 2
        bbox_centroid[:, 2] = self.global_bbox[:, 4] + bbox_dist[:, 2] / 2

        # Create a tree with the rank centroids
        self.log.write("info", "Creating global KD tree with rank centroids")
        self.global_tree = KDTree(bbox_centroid)
        self.search_radious = bbox_max_dist

        self.log.toc()

    def binning_hash(self, x, y, z):
        """
        """

        x_min = self.domain_min_x
        x_max = self.domain_max_x
        y_min = self.domain_min_y
        y_max = self.domain_max_y
        z_min = self.domain_min_z
        z_max = self.domain_max_z
        n_bins_1d = self.bin_size_1d
        max_bins_1d = n_bins_1d - 1

        bin_x = (np.floor((x - x_min) / ((x_max - x_min) / max_bins_1d))).astype(np.int32)
        bin_y = (np.floor((y - y_min) / ((y_max - y_min) / max_bins_1d))).astype(np.int32)
        bin_z = (np.floor((z - z_min) / ((z_max - z_min) / max_bins_1d))).astype(np.int32)

        # Clip the bins to be in the range [0, n_bins_1d - 1]
        bin_x = np.clip(bin_x, 0, max_bins_1d)
        bin_y = np.clip(bin_y, 0, max_bins_1d)
        bin_z = np.clip(bin_z, 0, max_bins_1d)

        bin = bin_x + bin_y * n_bins_1d + bin_z * n_bins_1d**2
        
        return bin

    def set_up_global_tree_domain_binning_(self, comm, global_tree_nbins=None):

        self.log.tic()

        if isinstance(global_tree_nbins, NoneType):
            global_tree_nbins = comm.Get_size()
            self.log.write(
                "info", f"nbins not provided, using {global_tree_nbins} as default"
            )

        bin_size = global_tree_nbins
        bin_size_1d = int(np.round(np.cbrt(bin_size))) 
        bin_size = bin_size_1d**3
        
        bins_per_rank = int(np.ceil(bin_size / comm.Get_size()))
            
        self.log.write(
            "info", f"Using {bin_size} as actual bin size"
        )
        self.log.write(
            "info", f"Storing {bins_per_rank} in each rank"
        )

        # Find the values that delimit a cubic boundin box
        # for the whole domain
        self.log.write("info", "Finding bounding box tha delimits the whole domain")
        rank_bbox = np.zeros((1, 6), dtype=INTERPOLATION_DTYPE)
        rank_bbox[0, 0] = np.min(self.x)
        rank_bbox[0, 1] = np.max(self.x)
        rank_bbox[0, 2] = np.min(self.y)
        rank_bbox[0, 3] = np.max(self.y)
        rank_bbox[0, 4] = np.min(self.z)
        rank_bbox[0, 5] = np.max(self.z)
        self.domain_min_x = comm.allreduce(rank_bbox[0, 0], op=MPI.MIN)
        self.domain_min_y = comm.allreduce(rank_bbox[0, 2], op=MPI.MIN)
        self.domain_min_z = comm.allreduce(rank_bbox[0, 4], op=MPI.MIN)
        self.domain_max_x = comm.allreduce(rank_bbox[0, 1], op=MPI.MAX)
        self.domain_max_y = comm.allreduce(rank_bbox[0, 3], op=MPI.MAX)
        self.domain_max_z = comm.allreduce(rank_bbox[0, 5], op=MPI.MAX)
        self.bin_size_1d = bin_size_1d
        self.bins_per_rank = bins_per_rank

        # Check in which bins the points are and check wich are the rank owners of those bins
        bins_of_points = self.binning_hash(self.x, self.y, self.z)
        bin_owner = np.floor(bins_of_points / bins_per_rank).astype(np.int32)

        # Let the rank owner of the bins know that I have data in those bins
        unique_ranks = np.unique(bin_owner)
        destinations = unique_ranks.tolist()
        bins_in_owner = [np.unique(bins_of_points[bin_owner == i]).astype(np.int32) for i in unique_ranks]
        
        sources, data_from_others = self.rt.send_recv(destination = destinations, data = bins_in_owner, dtype = np.int32, tag=0)
        
        # Create a hash table that contains the ranks that have data in the bins I own
        my_bins_range = [np.floor(self.rt.comm.Get_rank() * bins_per_rank).astype(np.int32), np.floor((self.rt.comm.Get_rank() + 1) * bins_per_rank).astype(np.int32)]
        bin_to_rank_map = {i : [] for i in range(my_bins_range[0], my_bins_range[1])}

        # Fill the has table with data
        ## Use sets to try to make it fast
        my_set = set(bin_to_rank_map.keys())
        # Iterate over the souces
        for rankk, binss in zip(sources, data_from_others):
            # Create a set with the bins in this rank
            binss_set = set(binss)
            # Check which are the bins that are in the intersection of the two sets
            intersection = my_set.intersection(binss_set)
            # If there are bins in the intersection, add the rank to the hash table
            if len(intersection) > 0:
                for i in intersection:
                    bin_to_rank_map[i].append(rankk.astype(np.int32))

        # Store the data
        self.bin_to_rank_map = bin_to_rank_map

    def scatter_probes_from_io_rank(self, io_rank, comm):
        """Scatter the probes from the rank that is used to read them - rank0 by default"""
        rank = comm.Get_rank()
        size = comm.Get_size()

        self.log.write("info", "Scattering probes")
        self.log.tic()

        # Check how many probes should be in each rank with a load balanced linear distribution
        probe_partition_sendcount = np.zeros((size), dtype=np.int64)
        if rank == io_rank:
            for i_rank in range(0, size):
                m = self.probes.shape[0]
                pe_rank = i_rank
                pe_size = comm.Get_size()
                # l = np.floor(np.double(m) / np.double(pe_size))
                # rr = np.mod(m, pe_size)
                ip = np.floor(
                    (
                        np.double(m)
                        + np.double(pe_size)
                        - np.double(pe_rank)
                        - np.double(1)
                    )
                    / np.double(pe_size)
                )
                nelv = int(ip)
                # offset_el = int(pe_rank * l + min(pe_rank, rr))
                # n = 3 * nelv
                probe_partition_sendcount[i_rank] = int(nelv)

        comm.Bcast(probe_partition_sendcount, root=io_rank)
        probe_coord_partition_sendcount = (
            probe_partition_sendcount * 3
        )  # Since each probe has 3 coordinates

        # Define some variables that need to be only in the rank that partitions
        if rank == io_rank:
            # Set the necesary arrays for identification of point
            number_of_points = self.probes.shape[0]
            self.probes_rst = np.zeros((number_of_points, 3), dtype=INTERPOLATION_DTYPE)
            self.el_owner = np.zeros((number_of_points), dtype=np.int32)
            self.glb_el_owner = np.zeros((number_of_points), dtype=np.int32)
            # self.rank_owner = np.zeros((number_of_points), dtype = np.int32)
            self.rank_owner = np.ones((number_of_points), dtype=np.int32) * -1000
            self.err_code = np.zeros(
                (number_of_points), dtype=np.int32
            )  # 0 not found, 1 is found
            self.test_pattern = np.ones(
                (number_of_points), dtype=INTERPOLATION_DTYPE
            )  # test interpolation holder
        else:
            self.probes_rst = None
            self.el_owner = None
            self.glb_el_owner = None
            self.rank_owner = None
            self.err_code = None
            self.test_pattern = None

        # Now scatter the probes and all the codes according
        # to what is needed and each rank has a partition
        self.probe_partition_sendcount = probe_partition_sendcount
        self.probe_coord_partition_sendcount = probe_coord_partition_sendcount
        ## Double
        tmp = self.rt.scatter_from_root(
            self.probes, probe_coord_partition_sendcount, io_rank, INTERPOLATION_DTYPE
        )
        self.probe_partition = tmp.reshape((int(tmp.size / 3), 3))
        tmp = self.rt.scatter_from_root(
            self.probes_rst, probe_coord_partition_sendcount, io_rank, INTERPOLATION_DTYPE
        )
        self.probe_rst_partition = tmp.reshape((int(tmp.size / 3), 3))
        ## Int
        self.el_owner_partition = self.rt.scatter_from_root(
            self.el_owner, probe_partition_sendcount, io_rank, np.int32
        )
        self.glb_el_owner_partition = self.rt.scatter_from_root(
            self.glb_el_owner, probe_partition_sendcount, io_rank, np.int32
        )
        self.rank_owner_partition = self.rt.scatter_from_root(
            self.rank_owner, probe_partition_sendcount, io_rank, np.int32
        )
        self.err_code_partition = self.rt.scatter_from_root(
            self.err_code, probe_partition_sendcount, io_rank, np.int32
        )
        ## Double
        self.test_pattern_partition = self.rt.scatter_from_root(
            self.test_pattern, probe_partition_sendcount, io_rank, INTERPOLATION_DTYPE
        )

        self.log.write("info", "done")
        self.log.toc()

        return

    def assign_local_probe_partitions(self):
        """If each rank has recieved a partition of the probes, assign them to the local variables"""

        self.log.write("info", "Assigning local probe partitions")
        self.log.tic()

        # Set the necesary arrays for identification of point
        number_of_points = self.probes.shape[0]
        self.probes_rst = np.zeros((number_of_points, 3), dtype=INTERPOLATION_DTYPE)
        self.el_owner = np.zeros((number_of_points), dtype=np.int32)
        self.glb_el_owner = np.zeros((number_of_points), dtype=np.int32)
        # self.rank_owner = np.zeros((number_of_points), dtype = np.int32)
        self.rank_owner = np.ones((number_of_points), dtype=np.int32) * -1000
        self.err_code = np.zeros(
            (number_of_points), dtype=np.int32
        )  # 0 not found, 1 is found
        self.test_pattern = np.ones(
            (number_of_points), dtype=INTERPOLATION_DTYPE
        )  # test interpolation holder

        self.probe_partition = self.probes.view()
        self.probe_rst_partition = self.probes_rst
        self.el_owner_partition = self.el_owner
        self.glb_el_owner_partition = self.glb_el_owner
        self.rank_owner_partition = self.rank_owner
        self.err_code_partition = self.err_code
        self.test_pattern_partition = self.test_pattern

        self.log.write("info", "done")
        self.log.toc()

        return

    def gather_probes_to_io_rank(self, io_rank, comm):
        """Gather the probes to the rank that is used to read them - rank0 by default"""

        self.log.write("info", "Gathering probes")
        self.log.tic()

        root = io_rank
        sendbuf = self.probe_partition.reshape((self.probe_partition.size))
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, INTERPOLATION_DTYPE)
        if not isinstance(recvbuf, NoneType):
            self.probes[:, :] = recvbuf.reshape((int(recvbuf.size / 3), 3))[:, :]
        sendbuf = self.probe_rst_partition.reshape((self.probe_rst_partition.size))
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, INTERPOLATION_DTYPE)
        if not isinstance(recvbuf, NoneType):
            self.probes_rst[:, :] = recvbuf.reshape((int(recvbuf.size / 3), 3))[:, :]
        sendbuf = self.el_owner_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.int32)
        if not isinstance(recvbuf, NoneType):
            self.el_owner[:] = recvbuf[:]
        sendbuf = self.glb_el_owner_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.int32)
        if not isinstance(recvbuf, NoneType):
            self.glb_el_owner[:] = recvbuf[:]
        sendbuf = self.rank_owner_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.int32)
        if not isinstance(recvbuf, NoneType):
            self.rank_owner[:] = recvbuf[:]
        sendbuf = self.err_code_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.int32)
        if not isinstance(recvbuf, NoneType):
            self.err_code[:] = recvbuf[:]
        sendbuf = self.test_pattern_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, INTERPOLATION_DTYPE)
        if not isinstance(recvbuf, NoneType):
            self.test_pattern[:] = recvbuf[:]

        self.log.write("info", "done")
        self.log.toc()

        return

    def find_points(
        self,
        comm,
        find_points_iterative: list =[False, 5000],
        find_points_comm_pattern="point_to_point",
        local_data_structure: str = "kdtree",
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(INTERPOLATION_DTYPE).eps * 10,
        max_iter=50,
        use_oriented_bbox = False,
    ):
        """Public method to dins points across ranks and elements"""
        self.log.write(
            "info",
            "using communication pattern: {}".format(find_points_comm_pattern),
        )
        # Check that the inputs are in the correct format 
        if not isinstance(find_points_iterative, (list, tuple)):
            find_points_iterative = fp_it_tolist(self, find_points_iterative)
        elif len(find_points_iterative) == 1:
            find_points_iterative = fp_it_tolist(self, find_points_iterative[0])
        elif len(find_points_iterative) != 2:
            raise ValueError("find_points_iterative must be a list or tuple of length 2")

        if find_points_comm_pattern == "broadcast":
            self.find_points_broadcast(
                comm,
                local_data_structure=local_data_structure,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
                tol=tol,
                max_iter=max_iter,
                use_oriented_bbox = use_oriented_bbox,
            )
        elif ((find_points_comm_pattern == "point_to_point") or (find_points_comm_pattern == "collective")) and not find_points_iterative[0]:
            self.find_points_(
                comm,
                local_data_structure=local_data_structure,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
                tol=tol,
                max_iter=max_iter,
                comm_pattern = find_points_comm_pattern,
                use_oriented_bbox = use_oriented_bbox,
            )
        elif ((find_points_comm_pattern == "point_to_point") or (find_points_comm_pattern == "collective")) and find_points_iterative[0]:
            self.find_points_iterative(
                comm,
                local_data_structure = local_data_structure,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
                tol=tol,
                batch_size=find_points_iterative[1],
                max_iter=max_iter,
                comm_pattern= find_points_comm_pattern,
                use_oriented_bbox = use_oriented_bbox,
            )
        elif find_points_comm_pattern == "rma":
            self.find_points_iterative_rma(
                comm,
                local_data_structure = local_data_structure,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
                tol=tol,
                max_iter=max_iter,
                use_oriented_bbox = use_oriented_bbox,
            )

    def find_points_broadcast(
        self,
        comm,
        local_data_structure: str = "kdtree",
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(INTERPOLATION_DTYPE).eps * 10,
        max_iter=50,
        use_oriented_bbox = False,
    ):
        """Find points using the collective implementation"""
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.rank = rank
        
        kwargs = {
            "elem_percent_expansion": elem_percent_expansion,
            "max_pts": self.max_pts,
            "use_oriented_bbox": use_oriented_bbox,
            "point_interpolator": self.ei,}
        
        if local_data_structure == "kdtree":
            self.my_tree = dstructure_kdtree(self.log, self.x, self.y, self.z, **kwargs)    
        
        elif local_data_structure == "bounding_boxes":            
            # First each rank finds their bounding box
            self.log.write("info", "Finding bounding box of sem mesh")
            self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        elif local_data_structure == "rtree":
            self.my_tree = dstructure_rtree(self.log, self.x, self.y, self.z, **kwargs)
        
        elif local_data_structure == "hashtable":
            self.my_tree = dstructure_hashtable(self.log, self.x, self.y, self.z, **kwargs)

        nelv = self.x.shape[0]
        self.ranks_ive_checked = []

        # Now recursively make more ranks communicate to search for points that have not been found
        # Start checking if I have the local ones
        denom = 1
        j = 0
        while j < size and denom <= size + 1:

            # With this logic, first every rank checks by themselves and then 2
            # check together, then 4 ... up to when all of them check
            col = int(np.floor(((rank / denom))))
            if DEBUG:
                print(
                    f"rank: {rank}, finding points. start iteration: {j}. Color: {col}"
                )
            else:
                if rank == 0:
                    print(f"Finding points. start iteration: {j}. Color: {col}")
            start_time = MPI.Wtime()
            denom = denom * 2

            search_comm = comm.Split(color=col, key=rank)
            search_rank = search_comm.Get_rank()
            search_size = search_comm.Get_size()

            search_rt = Router(search_comm)

            # Make each rank in the communicator broadcast their
            # bounding boxes to the others to search
            for broadcaster in range(0, search_size):

                # Points that I have not found to which rank or element belong to
                not_found = np.where(self.err_code_partition != 1)[0]
                n_not_found = not_found.size
                probe_not_found = self.probe_partition[not_found]
                probe_rst_not_found = self.probe_rst_partition[not_found]
                el_owner_not_found = self.el_owner_partition[not_found]
                glb_el_owner_not_found = self.glb_el_owner_partition[not_found]
                rank_owner_not_found = self.rank_owner_partition[not_found]
                err_code_not_found = self.err_code_partition[not_found]
                test_pattern_not_found = self.test_pattern_partition[not_found]

                # Tell every rank in the broadcaster of the local communicator their actual rank
                broadcaster_global_rank = np.ones((1), dtype=np.int32) * rank
                search_comm.Bcast(broadcaster_global_rank, root=broadcaster)

                # Tell every rank how much they need to allocate for the broadcaster bounding boxes
                nelv_in_broadcaster = np.ones((1), dtype=np.int32) * nelv
                search_comm.Bcast(nelv_in_broadcaster, root=broadcaster)

                # Allocate the recieve buffer for bounding boxes
                bbox_rec_buff = np.empty((nelv_in_broadcaster[0], 6), dtype=INTERPOLATION_DTYPE)

                # Only in the broadcaster copy the data
                if search_rank == broadcaster:
                    bbox_rec_buff[:, :] = np.copy(self.my_bbox[:, :])

                # Broadcast the bounding boxes so each rank can check if the points are there
                search_comm.Bcast(bbox_rec_buff, root=broadcaster)

                # Only do the search, if my rank has not already searched the broadcaster
                if broadcaster_global_rank not in self.ranks_ive_checked:

                    if local_data_structure == "bounding_boxes":
                        # Find a candidate rank to check
                        i = 0
                        if self.progress_bar:
                            pbar = tqdm(total=n_not_found)
                        for pt in probe_not_found:
                            found_candidate = False
                            for e in range(0, bbox_rec_buff.shape[0]):
                                if pt_in_bbox(
                                    pt, bbox_rec_buff[e], rel_tol=elem_percent_expansion
                                ):
                                    found_candidate = True
                                    # el_owner_not_found[i] = e
                                    err_code_not_found[i] = 1
                                    # rank_owner_not_found[i] = broadcaster_global_rank

                            if found_candidate:
                                i = i + 1
                                if self.progress_bar:
                                    pbar.update(1)
                                continue
                            i = i + 1
                            if self.progress_bar:
                                pbar.update(1)
                        if self.progress_bar:
                            pbar.close()
                    elif local_data_structure == "kdtree":

                        # Get bbox centroids and max radius from center to
                        #  corner for the broadcaster
                        bbox_centroids, bbox_maxdist = get_bbox_centroids_and_max_dist(
                            bbox_rec_buff
                        )

                        # Create the KDtree
                        broadcaster_tree = KDTree(bbox_centroids)
                        # Query the tree with the probes to reduce the bbox search
                        candidate_elements = broadcaster_tree.query_ball_point(
                            x=probe_not_found,
                            r=bbox_maxdist * (1 + 1e-6),
                            p=2.0,
                            eps=elem_percent_expansion,
                            workers=1,
                            return_sorted=False,
                            return_length=False,
                        )

                        # Do a bbox search over the candidate elements, just as it used to be done
                        # (The KD tree allows to avoid searching ALL elements)
                        i = 0
                        if self.progress_bar:
                            pbar = tqdm(total=n_not_found)
                        for pt in probe_not_found:
                            found_candidate = False
                            for e in candidate_elements[i]:
                                if pt_in_bbox(
                                    pt, bbox_rec_buff[e], rel_tol=elem_percent_expansion
                                ):
                                    found_candidate = True
                                    # el_owner_not_found[i] = e
                                    err_code_not_found[i] = 1
                                    # rank_owner_not_found[i] = broadcaster_global_rank

                            if found_candidate:
                                i = i + 1
                                if self.progress_bar:
                                    pbar.update(1)
                                continue
                            i = i + 1
                            if self.progress_bar:
                                pbar.update(1)
                        if self.progress_bar:
                            pbar.close()

                    # Now let the brodcaster gather the points that the other ranks think it has
                    # broadcaster_is_candidate =
                    # np.where(rank_owner_not_found == broadcaster_global_rank)[0]
                    broadcaster_is_candidate = np.where(err_code_not_found == 1)[0]

                    self.ranks_ive_checked.append(broadcaster_global_rank[0])
                else:
                    # If this rank has already checked the broadcaster, just produce an empty list
                    # broadcaster_is_candidate = np.where(rank_owner_not_found == -10000)[0]
                    broadcaster_is_candidate = np.where(err_code_not_found == 10000)[0]

                probe_broadcaster_is_candidate = probe_not_found[
                    broadcaster_is_candidate
                ]
                probe_rst_broadcaster_is_candidate = probe_rst_not_found[
                    broadcaster_is_candidate
                ]
                el_owner_broadcaster_is_candidate = el_owner_not_found[
                    broadcaster_is_candidate
                ]
                glb_el_owner_broadcaster_is_candidate = glb_el_owner_not_found[
                    broadcaster_is_candidate
                ]
                rank_owner_broadcaster_is_candidate = rank_owner_not_found[
                    broadcaster_is_candidate
                ]
                err_code_broadcaster_is_candidate = err_code_not_found[
                    broadcaster_is_candidate
                ]
                test_pattern_broadcaster_is_candidate = test_pattern_not_found[
                    broadcaster_is_candidate
                ]

                root = broadcaster
                tmp, probe_sendcount_broadcaster_is_candidate = (
                    search_rt.gather_in_root(
                        probe_broadcaster_is_candidate.reshape(
                            (probe_broadcaster_is_candidate.size)
                        ),
                        root,
                        INTERPOLATION_DTYPE,
                    )
                )
                if not isinstance(tmp, NoneType):
                    probe_broadcaster_has = tmp.reshape((int(tmp.size / 3), 3))

                tmp, _ = search_rt.gather_in_root(
                    probe_rst_broadcaster_is_candidate.reshape(
                        (probe_rst_broadcaster_is_candidate.size)
                    ),
                    root,
                    INTERPOLATION_DTYPE,
                )
                if not isinstance(tmp, NoneType):
                    probe_rst_broadcaster_has = tmp.reshape((int(tmp.size / 3), 3))
                (
                    el_owner_broadcaster_has,
                    el_owner_sendcount_broadcaster_is_candidate,
                ) = search_rt.gather_in_root(
                    el_owner_broadcaster_is_candidate, root, np.int32
                )
                glb_el_owner_broadcaster_has, _ = search_rt.gather_in_root(
                    glb_el_owner_broadcaster_is_candidate, root, np.int32
                )
                rank_owner_broadcaster_has, _ = search_rt.gather_in_root(
                    rank_owner_broadcaster_is_candidate, root, np.int32
                )
                err_code_broadcaster_has, _ = search_rt.gather_in_root(
                    err_code_broadcaster_is_candidate, root, np.int32
                )
                test_pattern_broadcaster_has, _ = search_rt.gather_in_root(
                    test_pattern_broadcaster_is_candidate, root, INTERPOLATION_DTYPE
                )

                # Now let the broadcaster check if it really had the point.
                # It will go through all the elements again and check rst coordinates
                if search_rank == broadcaster:

                    probes_info = {}
                    probes_info["probes"] = probe_broadcaster_has
                    probes_info["probes_rst"] = probe_rst_broadcaster_has
                    probes_info["el_owner"] = el_owner_broadcaster_has
                    probes_info["glb_el_owner"] = glb_el_owner_broadcaster_has
                    probes_info["rank_owner"] = rank_owner_broadcaster_has
                    probes_info["err_code"] = err_code_broadcaster_has
                    probes_info["test_pattern"] = test_pattern_broadcaster_has
                    probes_info["rank"] = broadcaster_global_rank
                    probes_info["offset_el"] = self.offset_el

                    mesh_info = {}
                    mesh_info["x"] = self.x
                    mesh_info["y"] = self.y
                    mesh_info["z"] = self.z
                    mesh_info["bbox"] = self.my_bbox
                    if hasattr(self, "my_tree"):
                        mesh_info["kd_tree"] = self.my_tree
                    if hasattr(self, "my_bbox_centroids"):
                        mesh_info["bbox_max_dist"] = self.my_bbox_maxdist
                    if hasattr(self, "local_data_structure"):
                        mesh_info["local_data_structure"] = self.local_data_structure

                    settings = {}
                    settings["not_found_code"] = -10
                    settings["use_test_pattern"] = True
                    settings["elem_percent_expansion"] = elem_percent_expansion
                    settings["progress_bar"] = self.progress_bar
                    settings["find_pts_tol"] = tol
                    settings["find_pts_max_iterations"] = max_iter

                    buffers = {}
                    buffers["r"] = self.r
                    buffers["s"] = self.s
                    buffers["t"] = self.t
                    buffers["test_interp"] = self.test_interp

                    [
                        probe_broadcaster_has,
                        probe_rst_broadcaster_has,
                        el_owner_broadcaster_has,
                        glb_el_owner_broadcaster_has,
                        rank_owner_broadcaster_has,
                        err_code_broadcaster_has,
                        test_pattern_broadcaster_has,
                    ] = self.ei.find_rst(
                        probes_info, mesh_info, settings, buffers=buffers
                    )

                # Now scatter the results back to all the other ranks
                root = broadcaster
                if search_rank == root:
                    sendbuf = probe_broadcaster_has.reshape(probe_broadcaster_has.size)
                else:
                    sendbuf = None
                recvbuf = search_rt.scatter_from_root(
                    sendbuf,
                    probe_sendcount_broadcaster_is_candidate,
                    root,
                    INTERPOLATION_DTYPE,
                )
                probe_broadcaster_is_candidate[:, :] = recvbuf.reshape(
                    (int(recvbuf.size / 3), 3)
                )[:, :]
                if search_rank == root:
                    sendbuf = probe_rst_broadcaster_has.reshape(
                        probe_rst_broadcaster_has.size
                    )
                else:
                    sendbuf = None
                recvbuf = search_rt.scatter_from_root(
                    sendbuf,
                    probe_sendcount_broadcaster_is_candidate,
                    root,
                    INTERPOLATION_DTYPE,
                )
                probe_rst_broadcaster_is_candidate[:, :] = recvbuf.reshape(
                    (int(recvbuf.size / 3), 3)
                )[:, :]
                if search_rank == root:
                    sendbuf = el_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = search_rt.scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.int32,
                )
                el_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    # sendbuf = el_owner_broadcaster_has+self.offset_el
                    sendbuf = glb_el_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = search_rt.scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.int32,
                )
                glb_el_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = rank_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = search_rt.scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.int32,
                )
                rank_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = err_code_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = search_rt.scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.int32,
                )
                err_code_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = test_pattern_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = search_rt.scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    INTERPOLATION_DTYPE,
                )
                test_pattern_broadcaster_is_candidate[:] = recvbuf[:]

                # Now that the data is back at the original ranks,
                # put it in the place of the not found list that it should be
                probe_not_found[broadcaster_is_candidate, :] = (
                    probe_broadcaster_is_candidate[:, :]
                )
                probe_rst_not_found[broadcaster_is_candidate, :] = (
                    probe_rst_broadcaster_is_candidate[:, :]
                )
                el_owner_not_found[broadcaster_is_candidate] = (
                    el_owner_broadcaster_is_candidate[:]
                )
                glb_el_owner_not_found[broadcaster_is_candidate] = (
                    glb_el_owner_broadcaster_is_candidate[:]
                )
                rank_owner_not_found[broadcaster_is_candidate] = (
                    rank_owner_broadcaster_is_candidate[:]
                )
                err_code_not_found[broadcaster_is_candidate] = (
                    err_code_broadcaster_is_candidate[:]
                )
                test_pattern_not_found[broadcaster_is_candidate] = (
                    test_pattern_broadcaster_is_candidate[:]
                )

                # at the end of the broadcaster run, update the information
                #  from the previously not found data
                self.probe_partition[not_found, :] = probe_not_found[:, :]
                self.probe_rst_partition[not_found] = probe_rst_not_found[:, :]
                self.el_owner_partition[not_found] = el_owner_not_found[:]
                self.glb_el_owner_partition[not_found] = glb_el_owner_not_found[:]
                self.rank_owner_partition[not_found] = rank_owner_not_found[:]
                self.err_code_partition[not_found] = err_code_not_found[:]
                self.test_pattern_partition[not_found] = test_pattern_not_found[:]

            if DEBUG:
                print(
                    f"rank: {rank}, finding points. finished iteration: {j}. time(s): {MPI.Wtime() - start_time}"
                )
            else:
                if rank == 0:
                    print(
                        f"rank: {rank}, finding points. finished iteration: {j}. time(s): {MPI.Wtime() - start_time}"
                    )

            j = j + 1

            del search_rt
            search_comm.Free()

        # Final check
        for j in range(0, len(self.test_pattern_partition)):
            # After all iteration are done, check if some points
            # were not found. Use the error code and the test pattern
            if (
                self.err_code_partition[j] != 1
                and self.test_pattern_partition[j] > test_tol
            ):
                self.err_code_partition[j] = 0

            # Check also if the rst are too big, then it needs to be outside
            # if ( abs(self.probe_rst_partition[j, 0]) +  abs(self.probe_rst_partition[j, 1]) +
            # abs(self.probe_rst_partition[j, 2]) ) > 3.5:
            #    self.err_code_partition[j] = 0

        return

    def find_points_(
        self,
        comm,
        local_data_structure: str = "kdtree",
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(INTERPOLATION_DTYPE).eps * 10,
        max_iter=50,
        comm_pattern = "point_to_point",
        use_oriented_bbox = False,
    ):
        """Find points using the point to point implementation"""
        rank = comm.Get_rank()
        self.rank = rank

        self.log.write("info", "Finding points - start")
        self.log.tic()
        start_time = MPI.Wtime()
        
        kwargs = {
            "elem_percent_expansion": elem_percent_expansion,
            "max_pts": self.max_pts,
            "use_oriented_bbox": use_oriented_bbox,
            "point_interpolator": self.ei,}
        
        if local_data_structure == "kdtree":
            self.my_tree = dstructure_kdtree(self.log, self.x, self.y, self.z, **kwargs)    
        
        elif local_data_structure == "bounding_boxes":            
            # First each rank finds their bounding box
            self.log.write("info", "Finding bounding box of sem mesh")
            self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        elif local_data_structure == "rtree":
            self.my_tree = dstructure_rtree(self.log, self.x, self.y, self.z, **kwargs)
        
        elif local_data_structure == "hashtable":
            self.my_tree = dstructure_hashtable(self.log, self.x, self.y, self.z, **kwargs)

        # nelv = self.x.shape[0]
        self.ranks_ive_checked = []

        # Get candidate ranks from a global kd tree
        # These are the destination ranks
        self.log.write("info", "Obtaining candidate ranks and sources")
        my_dest, _ = get_candidate_ranks(self, comm)

        # Create temporary arrays that store the points that have not been found
        not_found = np.where(self.err_code_partition != 1)[0]
        n_not_found = not_found.size
        probe_not_found = self.probe_partition[not_found]
        probe_rst_not_found = self.probe_rst_partition[not_found]
        el_owner_not_found = self.el_owner_partition[not_found]
        glb_el_owner_not_found = self.glb_el_owner_partition[not_found]
        rank_owner_not_found = self.rank_owner_partition[not_found]
        err_code_not_found = self.err_code_partition[not_found]
        test_pattern_not_found = self.test_pattern_partition[not_found]

        # Send data to my candidates and recieve from ranks where I am candidate
        self.log.write("info", "Send data to candidates and recieve from sources")

        # Pack and send/recv the probe data
        p_probes = pack_data(array_list=[probe_not_found, probe_rst_not_found])
        p_info = pack_data(array_list=[el_owner_not_found, glb_el_owner_not_found, rank_owner_not_found, err_code_not_found])
        
        ## Send and receive
        my_source, buff_p_probes = self.rt.transfer_data( comm_pattern,
            destination=my_dest, data=p_probes, dtype=INTERPOLATION_DTYPE, tag=1
        )
        _, buff_p_info = self.rt.transfer_data(comm_pattern,
            destination=my_dest, data=p_info, dtype=np.int32, tag=2
        )
        _, buff_test_pattern = self.rt.transfer_data(comm_pattern,
            destination=my_dest, data=test_pattern_not_found, dtype=INTERPOLATION_DTYPE, tag=3
        )

        # Unpack the data 
        buff_probes, buff_probes_rst = unpack_source_data(packed_source_data=buff_p_probes, number_of_arrays=2, equal_length=True, final_shape=(-1, 3))
        buff_el_owner, buff_glb_el_owner, buff_rank_owner, buff_err_code = unpack_source_data(packed_source_data=buff_p_info, number_of_arrays=4, equal_length=True)
        
        # Set the information for the coordinate search in this rank
        self.log.write("info", "Find rst coordinates for the points")
        mesh_info = {}
        mesh_info["x"] = self.x
        mesh_info["y"] = self.y
        mesh_info["z"] = self.z
        mesh_info["bbox"] = self.my_bbox
        if hasattr(self, "my_tree"):
            mesh_info["kd_tree"] = self.my_tree
        if hasattr(self, "my_bbox_maxdist"):
            mesh_info["bbox_max_dist"] = self.my_bbox_maxdist
        if hasattr(self, "local_data_structure"):
            mesh_info["local_data_structure"] = self.local_data_structure

        settings = {}
        settings["not_found_code"] = -10
        settings["use_test_pattern"] = True
        settings["elem_percent_expansion"] = elem_percent_expansion
        settings["progress_bar"] = self.progress_bar
        settings["find_pts_tol"] = tol
        settings["find_pts_max_iterations"] = max_iter

        buffers = {}
        buffers["r"] = self.r
        buffers["s"] = self.s
        buffers["t"] = self.t
        buffers["test_interp"] = self.test_interp

        # Now find the rst coordinates for the points stored in each of the buffers
        for source_index in range(0, len(my_source)):

            self.log.write(
                "debug", f"Processing batch: {source_index} out of {len(my_source)}"
            )

            probes_info = {}
            probes_info["probes"] = buff_probes[source_index]
            probes_info["probes_rst"] = buff_probes_rst[source_index]
            probes_info["el_owner"] = buff_el_owner[source_index]
            probes_info["glb_el_owner"] = buff_glb_el_owner[source_index]
            probes_info["rank_owner"] = buff_rank_owner[source_index]
            probes_info["err_code"] = buff_err_code[source_index]
            probes_info["test_pattern"] = buff_test_pattern[source_index]
            probes_info["rank"] = rank
            probes_info["offset_el"] = self.offset_el

            [
                buff_probes[source_index],
                buff_probes_rst[source_index],
                buff_el_owner[source_index],
                buff_glb_el_owner[source_index],
                buff_rank_owner[source_index],
                buff_err_code[source_index],
                buff_test_pattern[source_index],
            ] = self.ei.find_rst(probes_info, mesh_info, settings, buffers=buffers)

        # Set the request to Recieve back the data that I have sent to my candidates
        self.log.write("info", "Send data to sources and recieve from candidates")

        # Pack and send/recv the probe data
        p_buff_probes = pack_destination_data(destination_data=[buff_probes, buff_probes_rst])
        p_buff_info = pack_destination_data(destination_data=[buff_el_owner, buff_glb_el_owner, buff_rank_owner, buff_err_code])

        # Send and recieve
        _, obuff_p_probes = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=p_buff_probes, dtype=INTERPOLATION_DTYPE, tag=11
        )
        _, obuff_p_info = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=p_buff_info, dtype=np.int32, tag=12
        )
        _, obuff_test_pattern = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=buff_test_pattern, dtype=INTERPOLATION_DTYPE, tag=13
        )
 
        # Unpack the data
        obuff_probes, obuff_probes_rst = unpack_source_data(packed_source_data=obuff_p_probes, number_of_arrays=2, equal_length=True, final_shape=(-1, 3))
        obuff_el_owner, obuff_glb_el_owner, obuff_rank_owner, obuff_err_code = unpack_source_data(packed_source_data=obuff_p_info, number_of_arrays=4, equal_length=True)

        # Free resources from previous buffers if possible
        del (
            buff_probes,
            buff_probes_rst,
            buff_el_owner,
            buff_glb_el_owner,
            buff_rank_owner,
            buff_err_code,
            buff_test_pattern,
        )

        # Now loop through all the points in the buffers that
        # have been sent back and determine which point was found
        self.log.write(
            "info", "Determine which points were found and find best candidate"
        )
        for point in range(0, n_not_found):

            # These are the error code and test patterns for
            # this point from all the ranks that sent back
            all_err_codes = [arr[point] for arr in obuff_err_code]
            all_test_patterns = [arr[point] for arr in obuff_test_pattern]

            # Check if any rank had certainty that it had found the point
            found_err_code = np.where(np.array(all_err_codes) == 1)[0]

            # If the point was found in any rank, just choose the first
            # one in the list (in case there was more than one founder):
            if found_err_code.size > 0:
                index = found_err_code[0]
                self.probe_partition[point, :] = obuff_probes[index][point, :]
                self.probe_rst_partition[point, :] = obuff_probes_rst[index][point, :]
                self.el_owner_partition[point] = obuff_el_owner[index][point]
                self.glb_el_owner_partition[point] = obuff_glb_el_owner[index][point]
                self.rank_owner_partition[point] = obuff_rank_owner[index][point]
                self.err_code_partition[point] = obuff_err_code[index][point]
                self.test_pattern_partition[point] = obuff_test_pattern[index][point]

                # skip the rest of the loop
                continue

            # If the point was not found with certainty, then choose as
            # owner the the one that produced the smaller error in the test pattern
            try:
                min_test_pattern = np.where(
                    np.array(all_test_patterns) == np.array(all_test_patterns).min()
                )[0]
            except ValueError:
                min_test_pattern = np.array([])
            if min_test_pattern.size > 0:
                index = min_test_pattern[0]
                self.probe_partition[point, :] = obuff_probes[index][point, :]
                self.probe_rst_partition[point, :] = obuff_probes_rst[index][point, :]
                self.el_owner_partition[point] = obuff_el_owner[index][point]
                self.glb_el_owner_partition[point] = obuff_glb_el_owner[index][point]
                self.rank_owner_partition[point] = obuff_rank_owner[index][point]
                self.err_code_partition[point] = obuff_err_code[index][point]
                self.test_pattern_partition[point] = obuff_test_pattern[index][point]

        # Go through the points again, if the test pattern was
        #  too large, mark that point as not found
        for j in range(0, len(self.test_pattern_partition)):
            # After all iteration are done, check if some points
            #  were not found. Use the error code and the test pattern
            if (
                self.err_code_partition[j] != 1
                and self.test_pattern_partition[j] > test_tol
            ):
                self.err_code_partition[j] = 0

        comm.Barrier()
        self.log.write("info", "Finding points - finished")
        self.log.toc()

        return

    def find_points_iterative(
        self,
        comm,
        local_data_structure: str = "kdtree",
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(INTERPOLATION_DTYPE).eps * 10,
        max_iter=50,
        batch_size=5000,
        comm_pattern = "point_to_point",
        use_oriented_bbox = False,
    ):
        """Find points using the point to point implementation"""
        rank = comm.Get_rank()
        self.rank = rank

        self.log.write("info", "Finding points - start")
        self.log.tic()
        start_time = MPI.Wtime()

        kwargs = {
            "elem_percent_expansion": elem_percent_expansion,
            "max_pts": self.max_pts,
            "use_oriented_bbox": use_oriented_bbox,
            "point_interpolator": self.ei,}

        if local_data_structure == "kdtree":
            self.my_tree = dstructure_kdtree(self.log, self.x, self.y, self.z, **kwargs)    
        
        elif local_data_structure == "bounding_boxes":            
            # First each rank finds their bounding box
            self.log.write("info", "Finding bounding box of sem mesh")
            self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        elif local_data_structure == "rtree":
            self.my_tree = dstructure_rtree(self.log, self.x, self.y, self.z, **kwargs)
        
        elif local_data_structure == "hashtable":
            self.my_tree = dstructure_hashtable(self.log, self.x, self.y, self.z, **kwargs)

        # nelv = self.x.shape[0]
        self.ranks_ive_checked = []

        # Get candidate ranks from a global kd tree
        # These are the destination ranks
        self.log.write("info", "Obtaining candidate ranks and sources")
        my_dest, candidate_ranks_list = get_candidate_ranks(self, comm)
        
        self.log.write("info", "Determining maximun number of candidates")
        max_candidates = np.ones((1), dtype=np.int32) * len(my_dest)
        max_candidates = comm.allreduce(max_candidates, op=MPI.MAX)
        if batch_size > max_candidates[0]: batch_size = max_candidates[0]
         
        if batch_size == 1:
            # Obtain the number of columns corresponding to the maximum number of candidates among all points
            # Then create a numpy array padding for points that have less candidates
            num_rows = len(candidate_ranks_list)
            max_col = max(len(row) for row in candidate_ranks_list)
            candidates_per_point = np.full((num_rows, max_col), -1, dtype=np.int32)
            for i, row in enumerate(candidate_ranks_list):
                candidates_per_point[i, :len(row)] = row

        number_of_batches = int(np.ceil(max_candidates[0] / batch_size))
        self.log.write("info", f"Perfoming {number_of_batches} search iterations processing max {batch_size} candidates per iteration")

        for search_iteration in range(0, number_of_batches):

            self.log.write("info", f"Search iteration: {search_iteration+1} out of {number_of_batches}")

            start = int(search_iteration * batch_size)
            end = int((search_iteration + 1) * batch_size)
            if end > len(my_dest):
                end = len(my_dest)
            try:
                my_it_dest = my_dest[start:end]
            except IndexError:
                my_it_dest = []

            # This should never happen but if it does, make sure it is a list
            if not isinstance(my_it_dest, list):
                self.log.write("warning", "my_it_dest is not a list, making it one")
                my_it_dest = [my_it_dest]

            # If batch size is 1, only send the actual points that said are in candidate.
            # This should always be done, currently it is not to simplify keeping track of things
            if batch_size == 1:
                if self.global_tree_type == "domain_binning":
                    self.log.write("warning", "With batch size = 1, we have noticed that domain binning might fail due to non overlapping of bins. If it does fail, select batch size = 2 or higher")
                mask = (self.err_code_partition != 1)
                if len(my_it_dest) > 0:
                    candidate_mask = np.any(candidates_per_point == my_it_dest[0], axis=1)
                    combined_mask = mask & candidate_mask
                else:
                    combined_mask = mask
                not_found = np.flatnonzero(combined_mask)
            else:
                not_found = np.flatnonzero(self.err_code_partition != 1)

            n_not_found = not_found.size
            probe_not_found = self.probe_partition[not_found]
            probe_rst_not_found = self.probe_rst_partition[not_found]
            el_owner_not_found = self.el_owner_partition[not_found]
            glb_el_owner_not_found = self.glb_el_owner_partition[not_found]
            rank_owner_not_found = self.rank_owner_partition[not_found]
            err_code_not_found = self.err_code_partition[not_found]
            test_pattern_not_found = self.test_pattern_partition[not_found]

            # Send data to my candidates and recieve from ranks where I am candidate
            self.log.write("info", "Send data to candidates and recieve from sources")
            
            # Pack and send/recv the probe data
            p_probes = pack_data(array_list=[probe_not_found, probe_rst_not_found])
            p_info = pack_data(array_list=[el_owner_not_found, glb_el_owner_not_found, rank_owner_not_found, err_code_not_found])
            
            ## Send and receive
            my_source, buff_p_probes = self.rt.transfer_data( comm_pattern,
                destination=my_it_dest, data=p_probes, dtype=INTERPOLATION_DTYPE, tag=1
            )
            _, buff_p_info = self.rt.transfer_data(comm_pattern,
                destination=my_it_dest, data=p_info, dtype=np.int32, tag=2
            )
            _, buff_test_pattern = self.rt.transfer_data(comm_pattern,
                destination=my_it_dest, data=test_pattern_not_found, dtype=INTERPOLATION_DTYPE, tag=3
            )

            # Unpack the data 
            buff_probes, buff_probes_rst = unpack_source_data(packed_source_data=buff_p_probes, number_of_arrays=2, equal_length=True, final_shape=(-1, 3))
            buff_el_owner, buff_glb_el_owner, buff_rank_owner, buff_err_code = unpack_source_data(packed_source_data=buff_p_info, number_of_arrays=4, equal_length=True)
            
            # Set the information for the coordinate search in this rank
            self.log.write("info", "Find rst coordinates for the points")
            mesh_info = {}
            mesh_info["x"] = self.x
            mesh_info["y"] = self.y
            mesh_info["z"] = self.z
            mesh_info["bbox"] = self.my_bbox
            if hasattr(self, "my_tree"):
                mesh_info["kd_tree"] = self.my_tree
            if hasattr(self, "my_bbox_maxdist"):
                mesh_info["bbox_max_dist"] = self.my_bbox_maxdist
            if hasattr(self, "local_data_structure"):
                mesh_info["local_data_structure"] = self.local_data_structure

            settings = {}
            settings["not_found_code"] = -10
            settings["use_test_pattern"] = True
            settings["elem_percent_expansion"] = elem_percent_expansion
            settings["progress_bar"] = self.progress_bar
            settings["find_pts_tol"] = tol
            settings["find_pts_max_iterations"] = max_iter

            buffers = {}
            buffers["r"] = self.r
            buffers["s"] = self.s
            buffers["t"] = self.t
            buffers["test_interp"] = self.test_interp

            # Now find the rst coordinates for the points stored in each of the buffers
            for source_index in range(0, len(my_source)):

                self.log.write(
                    "debug", f"Processing batch: {source_index} out of {len(my_source)}"
                )

                probes_info = {}
                probes_info["probes"] = buff_probes[source_index]
                probes_info["probes_rst"] = buff_probes_rst[source_index]
                probes_info["el_owner"] = buff_el_owner[source_index]
                probes_info["glb_el_owner"] = buff_glb_el_owner[source_index]
                probes_info["rank_owner"] = buff_rank_owner[source_index]
                probes_info["err_code"] = buff_err_code[source_index]
                probes_info["test_pattern"] = buff_test_pattern[source_index]
                probes_info["rank"] = rank
                probes_info["offset_el"] = self.offset_el

                [
                    buff_probes[source_index],
                    buff_probes_rst[source_index],
                    buff_el_owner[source_index],
                    buff_glb_el_owner[source_index],
                    buff_rank_owner[source_index],
                    buff_err_code[source_index],
                    buff_test_pattern[source_index],
                ] = self.ei.find_rst(probes_info, mesh_info, settings, buffers=buffers)

            # Set the request to Recieve back the data that I have sent to my candidates
            self.log.write("info", "Send data to sources and recieve from candidates")

            # Pack and send/recv the probe data
            p_buff_probes = pack_destination_data(destination_data=[buff_probes, buff_probes_rst])
            p_buff_info = pack_destination_data(destination_data=[buff_el_owner, buff_glb_el_owner, buff_rank_owner, buff_err_code])

            # Send and recieve
            _, obuff_p_probes = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=p_buff_probes, dtype=INTERPOLATION_DTYPE, tag=11
            )
            _, obuff_p_info = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=p_buff_info, dtype=np.int32, tag=12
            )
            _, obuff_test_pattern = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=buff_test_pattern, dtype=INTERPOLATION_DTYPE, tag=13
            )

            # If no point was sent from this rank, then all buffers will be empty
            # so skip the rest of the loop
            if n_not_found < 1:
                continue
            
            # Unpack the data
            obuff_probes, obuff_probes_rst = unpack_source_data(packed_source_data=obuff_p_probes, number_of_arrays=2, equal_length=True, final_shape=(-1, 3))
            obuff_el_owner, obuff_glb_el_owner, obuff_rank_owner, obuff_err_code = unpack_source_data(packed_source_data=obuff_p_info, number_of_arrays=4, equal_length=True)

            # Free resources from previous buffers if possible
            del (
                buff_probes,
                buff_probes_rst,
                buff_el_owner,
                buff_glb_el_owner,
                buff_rank_owner,
                buff_err_code,
                buff_test_pattern,
            )

            # Skip the rest of the if this rank did not have candidates to send to
            if len(my_it_dest) == 0:
                continue

            # Now loop through all the points in the buffers that
            # have been sent back and determine which point was found
            self.log.write(
                "info", "Determine which points were found and find best candidate"
            )
            for relative_point, absolute_point  in enumerate(not_found):

                # These are the error code and test patterns for
                # this point from all the ranks that sent back
                all_err_codes = [arr[relative_point] for arr in obuff_err_code]
                all_test_patterns = [arr[relative_point] for arr in obuff_test_pattern]

                # Check if any rank had certainty that it had found the point
                found_err_code = np.where(np.array(all_err_codes) == 1)[0]

                # If the point was found in any rank, just choose the first
                # one in the list (in case there was more than one founder):
                if found_err_code.size > 0:
                    index = found_err_code[0]
                    self.probe_partition[absolute_point, :] = obuff_probes[index][relative_point, :]
                    self.probe_rst_partition[absolute_point, :] = obuff_probes_rst[index][relative_point, :]
                    self.el_owner_partition[absolute_point] = obuff_el_owner[index][relative_point]
                    self.glb_el_owner_partition[absolute_point] = obuff_glb_el_owner[index][relative_point]
                    self.rank_owner_partition[absolute_point] = obuff_rank_owner[index][relative_point]
                    self.err_code_partition[absolute_point] = obuff_err_code[index][relative_point]
                    self.test_pattern_partition[absolute_point] = obuff_test_pattern[index][relative_point]

                    # skip the rest of the loop
                    continue

                # If the point was not found with certainty, then choose as
                # owner the the one that produced the smaller error in the test pattern
                try:
                    min_test_pattern = np.where(
                        np.array(all_test_patterns) == np.array(all_test_patterns).min()
                    )[0]
                except ValueError:
                    min_test_pattern = np.array([])
                if min_test_pattern.size > 0:
                    index = min_test_pattern[0]
                    if obuff_test_pattern[index][relative_point] < self.test_pattern_partition[absolute_point]:
                        self.probe_partition[absolute_point, :] = obuff_probes[index][relative_point, :]
                        self.probe_rst_partition[absolute_point, :] = obuff_probes_rst[index][relative_point, :]
                        self.el_owner_partition[absolute_point] = obuff_el_owner[index][relative_point]
                        self.glb_el_owner_partition[absolute_point] = obuff_glb_el_owner[index][relative_point]
                        self.rank_owner_partition[absolute_point] = obuff_rank_owner[index][relative_point]
                        self.err_code_partition[absolute_point] = obuff_err_code[index][relative_point]
                        self.test_pattern_partition[absolute_point] = obuff_test_pattern[index][relative_point]

        # Go through the points again, if the test pattern was
        #  too large, mark that point as not found
        for j in range(0, len(self.test_pattern_partition)):
            # After all iteration are done, check if some points
            #  were not found. Use the error code and the test pattern
            if (
                self.err_code_partition[j] != 1
                and self.test_pattern_partition[j] > test_tol
            ):
                self.err_code_partition[j] = 0

        comm.Barrier()
        self.log.write("info", "Finding points - finished")
        self.log.toc()

        return
    
    def find_points_iterative_rma(
        self,
        comm,
        local_data_structure: str = "kdtree",
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(INTERPOLATION_DTYPE).eps * 10,
        max_iter=50,
        use_oriented_bbox = False,
    ):
        """Find points using the point to point implementation"""
        rank = comm.Get_rank()
        self.rank = rank

        self.log.write("info", "Finding points - start")
        self.log.tic()
        start_time = MPI.Wtime()

        self.log.write("warning", "RMA mode is known to miss some points that can be found otherwise. If you encounter some not found points, consider changing communication pattern")

        batch_size = 1
        if batch_size != 1:
            raise ValueError(
                "batch_size != 1 is not supported in the RMA mode"
            )

        kwargs = {
            "elem_percent_expansion": elem_percent_expansion,
            "max_pts": self.max_pts,
            "use_oriented_bbox": use_oriented_bbox,
            "point_interpolator": self.ei,}

        if local_data_structure == "kdtree":
            self.my_tree = dstructure_kdtree(self.log, self.x, self.y, self.z, **kwargs)    
        
        elif local_data_structure == "bounding_boxes":            
            # First each rank finds their bounding box
            self.log.write("info", "Finding bounding box of sem mesh")
            self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        elif local_data_structure == "rtree":
            self.my_tree = dstructure_rtree(self.log, self.x, self.y, self.z, **kwargs)
        
        elif local_data_structure == "hashtable":
            self.my_tree = dstructure_hashtable(self.log, self.x, self.y, self.z, **kwargs)

        # nelv = self.x.shape[0]
        self.ranks_ive_sent_to = []
        self.ranks_ive_checked = []

        # Get candidate ranks from a global data structure
        # These are the destination ranks
        self.log.write("info", "Obtaining candidate ranks and sources")
        my_dest, candidate_ranks_list = get_candidate_ranks(self, comm)
        # Put my own rank first if it is in the list
        if comm.Get_rank() in my_dest:
            my_dest_ = [comm.Get_rank()] + my_dest
            my_dest = []
            for _, d in enumerate(my_dest_):
                if d not in my_dest:
                    my_dest.append(d)

        max_candidates = np.ones((1), dtype=np.int32) * len(my_dest)
        max_candidates = comm.allreduce(max_candidates, op=MPI.MAX)
        if batch_size > max_candidates[0]: batch_size = max_candidates[0]
        self.log.write("info", f"Max candidates in a rank was: {max_candidates}")
 
        # Obtain the number of columns corresponding to the maximum number of candidates among all points
        # Then create a numpy array padding for points that have less candidates
        num_rows = len(candidate_ranks_list)
        max_col = max(len(row) for row in candidate_ranks_list)
        candidates_per_point = np.full((num_rows, max_col), -1, dtype=np.int32)
        for i, row in enumerate(candidate_ranks_list):
            candidates_per_point[i, :len(row)] = row

        del candidate_ranks_list
        gc.collect()
 
        # Initialize windows
        ## Find sizes    
        mask = (self.err_code_partition != 1)
        total_not_found = np.flatnonzero(mask)     
        total_n_not_found = total_not_found.size
        ## Check how many points need to be send to the most popular rank
        max_points_to_send = 0
        for dest in my_dest:    
            candidate_mask = np.any(candidates_per_point == dest, axis=1)
            combined_mask = mask & candidate_mask
            not_found_at_this_candidate = np.flatnonzero(combined_mask)    
            n_not_found_at_this_candidate = not_found_at_this_candidate.size
            if max_points_to_send < n_not_found_at_this_candidate:
                max_points_to_send = n_not_found_at_this_candidate
        ## see how much memory should be allocated per window (in number of items of a data type to be defined later)
        max_points_to_allocate = self.rt.comm.allreduce(max_points_to_send, op=MPI.MAX)
        max_probe_pack = max_points_to_allocate*2*3
        max_info_pack = max_points_to_allocate*4 
        max_test_pattern_pack = max_points_to_allocate
        ## Create windows    
        rma_inputs = { "search_done": {"window_size": comm.Get_size(), "dtype": np.int32, "fill_value": 0},
                        "find_busy": {"window_size": 1, "dtype": np.int32, "fill_value": -1},
                        "find_done": {"window_size": 1, "dtype": np.int32, "fill_value": -1},
                        "find_n_not_found": {"window_size": 1, "dtype": np.int64, "fill_value": 0},
                        "find_p_probes": {"window_size": max_probe_pack, "dtype": INTERPOLATION_DTYPE, "fill_value": None},
                        "find_p_info": {"window_size": max_info_pack, "dtype": np.int32, "fill_value": None},
                        "find_test_pattern": {"window_size": max_test_pattern_pack, "dtype": INTERPOLATION_DTYPE, "fill_value": None},
                        "verify_busy": {"window_size": 1, "dtype": np.int32, "fill_value": -1},
                        "verify_done": {"window_size": 1, "dtype": np.int32, "fill_value": -1},
                        "verify_n_not_found": {"window_size": 1, "dtype": np.int64, "fill_value": 0},
                        "verify_p_probes": {"window_size": max_probe_pack, "dtype": INTERPOLATION_DTYPE, "fill_value": None},
                        "verify_p_info": {"window_size": max_info_pack, "dtype": np.int32, "fill_value": None},
                        "verify_test_pattern": {"window_size": max_test_pattern_pack, "dtype": INTERPOLATION_DTYPE, "fill_value": None}
                      }
        rma = RMAWindow(self.rt.comm, rma_inputs)

        # Start the search
        search_iteration = -1
        search_flag = True  
        # Check if all ranks are done
        keep_searching = np.zeros((1), dtype=np.int32)
        am_i_done = False
        i_sent_data = False
        last_log = 0
        log_entry = 1
        self.data_in_transit = {"dummy": False}
        self.points_sent = {}
        checked_data = False
        returned_data = False
        comm.Barrier()
        while search_flag:
            search_iteration += 1

            # Sincrhonize once before proceeding
            # This is currently needed to avoid races.
            # The flags that are used to indicate if data is availbale seems to not be sufficient
            # For some reason, the atomic operations are not working as I want. Maybe because of also checking my own rank?
            # Having a workaround should make everything faster 
            #comm.Barrier()

            #keep_searching_ = rma.search_done.get(source = 0, displacement=0)
            keep_searching = np.sum(rma.search_done.get(source = 0, displacement=0))
            log_time = int(np.floor(MPI.Wtime() - start_time))
            if (np.mod(log_time, INTERPOLATION_LOG_TIME) == 0 and log_time != last_log) or search_iteration == 0:
                if search_iteration == 0: self.log.write("info", f"Starting search iterations. Rank 0 will attempt to log every {INTERPOLATION_LOG_TIME} seconds unless it is busy processing data")
                self.log.write("info", f'Log entry: {log_entry}, search iteration {search_iteration+1} in progress. We keep searching on : {comm.Get_size() - keep_searching} ranks')            
                last_log = log_time
                log_entry += 1

            mask = (self.err_code_partition != 1)
            combined_mask = mask
            total_not_found = np.flatnonzero(combined_mask)        
            total_n_not_found = total_not_found.size
            
            if total_n_not_found <= 0 and not am_i_done:
                # Say that this rank is done
                rma.search_done.put(dest = 0, data = 1, dtype=np.int32, displacement=comm.Get_rank()) 
                am_i_done = True
 
            # Send points if you need to
            if total_n_not_found > 0 and (len(my_dest) != len(self.ranks_ive_checked)):
                for dest in my_dest:
                    if dest not in self.ranks_ive_checked and dest not in self.ranks_ive_sent_to:
                        # Check if the destination rank is busy
                        busy_buff = rma.find_busy.compare_and_swap(value_to_check=-1, value_to_put=self.rt.comm.Get_rank(), dest=dest)

                        if busy_buff != -1:
                            continue  # The rank is busy, and my rank lost the race, skip it for now
                            
                        # Obtain relevant ranks for this destination:
                        mask = (self.err_code_partition != 1)
                        candidate_mask = np.any(candidates_per_point == dest, axis=1)
                        combined_mask = mask & candidate_mask
                        not_found_at_this_candidate = np.flatnonzero(combined_mask)    
                        n_not_found_at_this_candidate = not_found_at_this_candidate.size
                        #if n_not_found_at_this_candidate < 1:
                        #    # Reset the busy flag
                        #    rma.find_busy.put(dest=dest, data=-1, dtype=np.int32)
                        #    self.ranks_ive_checked.append(dest)
                        #    continue
 
                        # Send the data - Lock communication in first instance and flush/unlock in the last one
                        rma.find_n_not_found.put(dest=dest, data = n_not_found_at_this_candidate, dtype=np.int64, lock=True, flush=False, unlock=False)
                        rma.find_p_probes.put_sequence(dest, data = [self.probe_partition, self.probe_rst_partition], lock=False, flush=False, unlock=False, mask=combined_mask)
                        rma.find_p_info.put_sequence(dest, data = [self.el_owner_partition, self.glb_el_owner_partition, self.rank_owner_partition, self.err_code_partition], lock=False, flush=False, unlock=False, mask=combined_mask)
                        rma.find_test_pattern.put(dest=dest, data = self.test_pattern_partition, lock=False, flush=False, unlock=False, mask = combined_mask)
                        rma.find_done.put(dest=dest, data = 1, dtype=np.int32, lock=False, flush=True, unlock=True)

                        # Store variables for later
                        self.ranks_ive_sent_to.append(dest) 
                        self.data_in_transit[int(dest)] = True
                        self.points_sent[int(dest)] = not_found_at_this_candidate

                        # If it is the first iteration, and I have to check in my rank, then do not send to anyone else yet.
                        # This is to avoid sending excesive data to others in mesh to mesh interpolation
                        if search_iteration == 0 and dest == self.rt.comm.Get_rank():
                            break # Get out of the send loop

            # Now find points from other ranks if anyone has sent me data
            if rma.find_busy.buff[0] != -1 and rma.find_done.buff[0] == 1:
                if not checked_data:
                    my_source = [rma.find_busy.buff[0].copy()]
                    # Give it the correct format.
                    buff_probes, buff_probes_rst = unpack_source_data(packed_source_data=[rma.find_p_probes.buff[:rma.find_n_not_found.buff[0]*2*3].view()], number_of_arrays=2, equal_length=True, final_shape=(-1, 3))
                    buff_el_owner, buff_glb_el_owner, buff_rank_owner, buff_err_code = unpack_source_data(packed_source_data=[rma.find_p_info.buff[:rma.find_n_not_found.buff[0]*4].view()], number_of_arrays=4, equal_length=True)
                    buff_test_pattern = [rma.find_test_pattern.buff[:rma.find_n_not_found.buff[0]].view()]

                    # Set the information for the coordinate search in this rank
                    mesh_info = {}
                    mesh_info["x"] = self.x
                    mesh_info["y"] = self.y
                    mesh_info["z"] = self.z
                    mesh_info["bbox"] = self.my_bbox
                    if hasattr(self, "my_tree"):
                        mesh_info["kd_tree"] = self.my_tree
                    if hasattr(self, "my_bbox_maxdist"):
                        mesh_info["bbox_max_dist"] = self.my_bbox_maxdist
                    if hasattr(self, "local_data_structure"):
                        mesh_info["local_data_structure"] = self.local_data_structure

                    settings = {}
                    settings["not_found_code"] = -10
                    settings["use_test_pattern"] = True
                    settings["elem_percent_expansion"] = elem_percent_expansion
                    settings["progress_bar"] = self.progress_bar
                    settings["find_pts_tol"] = tol
                    settings["find_pts_max_iterations"] = max_iter

                    buffers = {}
                    buffers["r"] = self.r
                    buffers["s"] = self.s
                    buffers["t"] = self.t
                    buffers["test_interp"] = self.test_interp

                    # Now find the rst coordinates for the points stored in each of the buffers
                    for source_index in range(0, len(my_source)):

                        self.log.write(
                            "debug", f"Processing batch: {source_index} out of {len(my_source)}"
                        )

                        if rma.find_n_not_found.buff[0] != 0:

                            probes_info = {}
                            probes_info["probes"] = buff_probes[source_index]
                            probes_info["probes_rst"] = buff_probes_rst[source_index]
                            probes_info["el_owner"] = buff_el_owner[source_index]
                            probes_info["glb_el_owner"] = buff_glb_el_owner[source_index]
                            probes_info["rank_owner"] = buff_rank_owner[source_index]
                            probes_info["err_code"] = buff_err_code[source_index]
                            probes_info["test_pattern"] = buff_test_pattern[source_index]
                            probes_info["rank"] = rank
                            probes_info["offset_el"] = self.offset_el
                            [
                                buff_probes[source_index],
                                buff_probes_rst[source_index],
                                buff_el_owner[source_index],
                                buff_glb_el_owner[source_index],
                                buff_rank_owner[source_index],
                                buff_err_code[source_index],
                                buff_test_pattern[source_index],
                            ] = self.ei.find_rst(probes_info, mesh_info, settings, buffers=buffers)
                    
                    # Reset the flag
                    checked_data = True
                    returned_data = False

                # Send the data back to the source rank - Follow the same handshake as before, but this time, do not try to get a new rank if this one is busy. Simply wait until it can recieve data
                if not returned_data:

                    # Check if the destination rank is busy
                    busy_buff = rma.verify_busy.compare_and_swap(value_to_check=-1, value_to_put=self.rt.comm.Get_rank(), dest=my_source[0])

                    if busy_buff != -1:
                        pass  # The rank is busy, and my rank lost the race, skip it for now
                    else:

                        # Put the data back - Lock communication in first instance and flush/unlock in the last one
                        n_checked = rma.find_n_not_found.buff[0]
                        # Mask to send all the data
                        mask = np.ones((n_checked,), dtype=bool)

                        rma.verify_n_not_found.put(dest=my_source[0], data = n_checked, dtype=np.int64, lock=True, flush=False, unlock=False)
                        rma.verify_p_probes.put_sequence(dest=my_source[0], data = [buff_probes[0], buff_probes_rst[0]], lock=False, flush=False, unlock=False, mask=mask)
                        rma.verify_p_info.put_sequence(dest=my_source[0], data = [buff_el_owner[0], buff_glb_el_owner[0], buff_rank_owner[0], buff_err_code[0]], lock=False, flush=False, unlock=False, mask=mask)
                        rma.verify_test_pattern.put(dest=my_source[0], data = buff_test_pattern[0], lock=False, flush=False, unlock=False) 
                        rma.verify_done.put(dest=my_source[0], data = 1, dtype=np.int32, lock=False, flush=True, unlock=True)

                        # Signal that my buffer is now ready to be used to find points
                        rma.find_busy.put(dest=self.rt.comm.Get_rank(), data=-1, dtype=np.int32, lock=True, flush=False, unlock=False)
                        rma.find_done.put(dest=self.rt.comm.Get_rank(), data=-1, dtype=np.int32, lock=False, flush=False, unlock=False)
                        rma.find_n_not_found.put(dest=self.rt.comm.Get_rank(), data= 0, dtype=np.int64, lock=False, flush=True, unlock=True)
                        
                        # Reset flags
                        checked_data = False
                        returned_data = True

            i_sent_data = np.any([self.data_in_transit[dest_] for dest_ in self.ranks_ive_sent_to])
            if i_sent_data and rma.verify_busy.buff[0] != -1 and rma.verify_done.buff[0] == 1:
                # The previous loop will wait until there is data here. So we can continue
                # Give it the correct format.
                obuff_probes, obuff_probes_rst = unpack_source_data(packed_source_data=[rma.verify_p_probes.buff[:rma.verify_n_not_found.buff[0]*2*3].view()], number_of_arrays=2, equal_length=True, final_shape=(-1, 3))
                obuff_el_owner, obuff_glb_el_owner, obuff_rank_owner, obuff_err_code = unpack_source_data(packed_source_data=[rma.verify_p_info.buff[:rma.verify_n_not_found.buff[0]*4].view()], number_of_arrays=4, equal_length=True)
                obuff_test_pattern = [rma.verify_test_pattern.buff[:rma.verify_n_not_found.buff[0]].view()]

                # Signal that my buffer is now ready to be used to find points
                self.ranks_ive_checked.append(int(rma.verify_busy.buff[0]))

                # Now loop through all the points in the buffers that
                # have been sent back and determine which point was found
                for relative_point, absolute_point  in enumerate(self.points_sent[int(rma.verify_busy.buff[0])]):

                    # These are the error code and test patterns for
                    # this point from all the ranks that sent back
                    all_err_codes = [arr[relative_point] for arr in obuff_err_code]
                    all_test_patterns = [arr[relative_point] for arr in obuff_test_pattern]

                    # Check if any rank had certainty that it had found the point
                    found_err_code = np.where(np.array(all_err_codes) == 1)[0]

                    # If the point was found in any rank, just choose the first
                    # one in the list (in case there was more than one founder):
                    if found_err_code.size > 0:
                        index = found_err_code[0]
                        self.probe_partition[absolute_point, :] = obuff_probes[index][relative_point, :]
                        self.probe_rst_partition[absolute_point, :] = obuff_probes_rst[index][relative_point, :]
                        self.el_owner_partition[absolute_point] = obuff_el_owner[index][relative_point]
                        self.glb_el_owner_partition[absolute_point] = obuff_glb_el_owner[index][relative_point]
                        self.rank_owner_partition[absolute_point] = obuff_rank_owner[index][relative_point]
                        self.err_code_partition[absolute_point] = obuff_err_code[index][relative_point]
                        self.test_pattern_partition[absolute_point] = obuff_test_pattern[index][relative_point]

                        # skip the rest of the loop
                        continue

                    # If the point was not found with certainty, then choose as
                    # owner the the one that produced the smaller error in the test pattern
                    try:
                        min_test_pattern = np.where(
                            np.array(all_test_patterns) == np.array(all_test_patterns).min()
                        )[0]
                    except ValueError:
                        min_test_pattern = np.array([])
                    if min_test_pattern.size > 0:
                        index = min_test_pattern[0]
                        if (obuff_test_pattern[index][relative_point] < self.test_pattern_partition[absolute_point]) and (self.err_code_partition[absolute_point] != 1):
                            self.probe_partition[absolute_point, :] = obuff_probes[index][relative_point, :]
                            self.probe_rst_partition[absolute_point, :] = obuff_probes_rst[index][relative_point, :]
                            self.el_owner_partition[absolute_point] = obuff_el_owner[index][relative_point]
                            self.glb_el_owner_partition[absolute_point] = obuff_glb_el_owner[index][relative_point]
                            self.rank_owner_partition[absolute_point] = obuff_rank_owner[index][relative_point]
                            self.err_code_partition[absolute_point] = obuff_err_code[index][relative_point]
                            self.test_pattern_partition[absolute_point] = obuff_test_pattern[index][relative_point]
                
                # Signal I am ready for more data
                rma.verify_busy.put(dest = self.rt.comm.Get_rank(), data = -1, dtype=np.int32, lock=True, flush=False, unlock=False)
                rma.verify_done.put(dest = self.rt.comm.Get_rank(), data = -1, dtype=np.int32, lock=False, flush=False, unlock=False)
                rma.verify_n_not_found.put(dest = self.rt.comm.Get_rank(), data = 0, dtype=np.int64, lock=False, flush=True, unlock=True)

                # Reset some of the flags
                self.data_in_transit[int(rma.verify_busy.buff[0])] = False

                if len(self.ranks_ive_checked) == len(my_dest) and not am_i_done:
                    # Say that this rank is done
                    rma.search_done.put(dest=0, data=1, dtype=np.int32, displacement=comm.Get_rank())
                    am_i_done = True
             
            
            if int(keep_searching) == int(self.rt.comm.Get_size()):
                self.log.write("info", "All ranks are done searching, exiting")
                search_flag = False
                continue

        # Go through the points again, if the test pattern was
        #  too large, mark that point as not found
        for j in range(0, len(self.test_pattern_partition)):
            # After all iteration are done, check if some points
            #  were not found. Use the error code and the test pattern
            if (
                self.err_code_partition[j] != 1
                and self.test_pattern_partition[j] > test_tol
            ):
                self.err_code_partition[j] = 0

        comm.Barrier()
        self.log.write("info", "Finding points - finished")
        self.log.toc()

        return


    def redistribute_probes_to_owners_from_io_rank(self, io_rank, comm):
        """Redistribute the probes to the ranks that
        have been determined in the search"""

        self.log.write("info", "Scattering probes")
        self.log.tic()

        rank = comm.Get_rank()
        size = comm.Get_size()

        probes = self.probes
        probes_rst = self.probes_rst
        el_owner = self.el_owner
        rank_owner = self.rank_owner
        err_code = self.err_code

        # Sort the points by rank to scatter them easily
        self.log.write("debug", "Scattering probes - sorting")
        if rank == io_rank:

            # Before the sorting, assign all the not found probes to rank zero.
            # The points with error code 0 will be ignored in the interpolation routine
            # Doing this will avoid an error when after interpolating we try to gather the points
            self.rank_owner[np.where(err_code == 0)] = 0
            rank_owner[np.where(err_code == 0)] = 0

            sort_by_rank = np.argsort(rank_owner)

            sorted_probes = probes[sort_by_rank]
            sorted_probes_rst = probes_rst[sort_by_rank]
            sorted_el_owner = el_owner[sort_by_rank]
            sorted_rank_owner = rank_owner[sort_by_rank]
            sorted_err_code = err_code[sort_by_rank]
        else:
            sort_by_rank = None

        # Check the sendcounts in number of probes
        self.log.write("debug", "Scattering probes - Defining sendcounts")
        sendcounts = np.zeros((size), dtype=np.int64)
        if rank == io_rank:
            sendcounts[:] = np.bincount(rank_owner, minlength=size)

        self.log.write("debug", "Scattering probes - Broadcasting")
        comm.Bcast(sendcounts, root=0)

        root = io_rank
        # Redistribute probes
        self.log.write("debug", "Scattering probes - probes")
        if rank == root:
            sendbuf = sorted_probes.reshape((sorted_probes.size))
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts * 3, root, INTERPOLATION_DTYPE)
        my_probes = recvbuf.reshape((int(recvbuf.size / 3), 3))

        # Redistribute probes rst
        self.log.write("debug", "Scattering probes - probes rst")
        if rank == root:
            sendbuf = sorted_probes_rst.reshape((sorted_probes_rst.size))
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts * 3, root, INTERPOLATION_DTYPE)
        my_probes_rst = recvbuf.reshape((int(recvbuf.size / 3), 3))

        # Redistribute err_code
        self.log.write("debug", "Scattering probes - err code")
        if rank == root:
            sendbuf = sorted_err_code.reshape((sorted_err_code.size))
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts, root, np.int32)
        my_err_code = recvbuf

        # Redistribute el_owner
        self.log.write("debug", "Scattering probes - el owner")
        if rank == root:
            sendbuf = sorted_el_owner.reshape((sorted_el_owner.size))
            # print(sendbuf)
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts, root, np.int32)
        # print(recvbuf)
        my_el_owner = recvbuf

        # Redistribute el_owner
        self.log.write("debug", "Scattering probes - rank owner")
        if rank == root:
            sendbuf = sorted_rank_owner.reshape((sorted_rank_owner.size))
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts, root, np.int32)
        my_rank_owner = recvbuf

        self.log.write("info", "Assigning my data")
        self.my_probes = my_probes
        self.my_probes_rst = my_probes_rst
        self.my_err_code = my_err_code
        self.my_el_owner = my_el_owner
        self.my_rank_owner = my_rank_owner
        self.sendcounts = sendcounts
        self.sort_by_rank = sort_by_rank

        self.log.write("info", "done")
        self.log.toc()

        return

    def redistribute_probes_to_owners(self):
        """Redistribute the probes to the ranks that
        have been determined in the search"""

        self.log.write("info", "Redistributing probes to owners")
        self.log.tic()

        # Assing the partitions
        self.probes[:, :] = self.probe_partition[:, :]
        self.probes_rst[:, :] = self.probe_rst_partition[:, :]
        self.el_owner[:] = self.el_owner_partition[:]
        self.glb_el_owner[:] = self.glb_el_owner_partition[:]
        self.rank_owner[:] = self.rank_owner_partition[:]
        self.err_code[:] = self.err_code_partition[:]
        self.test_pattern[:] = self.test_pattern_partition[:]

        # Rename some of the variables
        probes = self.probe_partition
        probes_rst = self.probe_rst_partition
        el_owner = self.el_owner_partition
        rank_owner = self.rank_owner_partition
        err_code = self.err_code_partition
        local_probe_index = np.arange(0, probes.shape[0], dtype=np.int64)

        # Prepare the send buffers
        destinations = []
        local_probe_index_sent_to_destination = []
        probe_data = []
        probe_rst_data = []
        el_owner_data = []
        rank_owner_data = []
        err_code_data = []
        for rank in range(0, self.rt.comm.Get_size()):
            probes_to_send_to_this_rank = np.where(rank_owner == rank)[0]
            if probes_to_send_to_this_rank.size > 0:
                destinations.append(rank)
                local_probe_index_sent_to_destination.append(
                    local_probe_index[probes_to_send_to_this_rank]
                )
                probe_data.append(probes[probes_to_send_to_this_rank])
                probe_rst_data.append(probes_rst[probes_to_send_to_this_rank])
                el_owner_data.append(el_owner[probes_to_send_to_this_rank])
                rank_owner_data.append(rank_owner[probes_to_send_to_this_rank])
                err_code_data.append(err_code[probes_to_send_to_this_rank])

        # Send the data to the destinations
        sources, source_probes = self.rt.send_recv(
            destination=destinations, data=probe_data, dtype=probe_data[0].dtype, tag=1
        )
        _, source_probes_rst = self.rt.send_recv(
            destination=destinations, data=probe_rst_data, dtype=probe_rst_data[0].dtype, tag=2
        )
        _, source_el_owner = self.rt.send_recv(
            destination=destinations, data=el_owner_data, dtype=el_owner_data[0].dtype, tag=3
        )
        _, source_rank_owner = self.rt.send_recv(
            destination=destinations,
            data=rank_owner_data,
            dtype=rank_owner_data[0].dtype,
            tag=4,
        )
        _, source_err_code = self.rt.send_recv(
            destination=destinations, data=err_code_data, dtype=err_code_data[0].dtype, tag=5
        )

        # Then reshape the data form the probes
        for source_index in range(0, len(sources)):
            source_probes[source_index] = source_probes[source_index].reshape(-1, 3)
            source_probes_rst[source_index] = source_probes_rst[source_index].reshape(
                -1, 3
            )

        # Now simply assign the data.
        # These are the probes tha I own from each of those sources
        self.my_sources = sources
        self.my_probes = source_probes
        self.my_probes_rst = source_probes_rst
        self.my_el_owner = source_el_owner
        self.my_rank_owner = source_rank_owner
        self.my_err_code = source_err_code

        # Keep track of which there the probes that I sent
        self.destinations = destinations
        self.local_probe_index_sent_to_destination = (
            local_probe_index_sent_to_destination
        )

        self.log.write("info", "done")
        self.log.toc()

        return

    def interpolate_field_from_rst(self, sampled_field):
        """Interpolate the field from the rst coordinates found"""

        self.log.write("info", "Interpolating field from rst coordinates")
        self.log.tic()

        if isinstance(self.my_probes_rst, list):
            # The inputs were distributed
            # So we return a list with the sample fields for the points of each rank that sent data to this one

            sampled_field_at_probe = []

            for i in range(0, len(self.my_probes_rst)):
                probes_info = {}
                if hasattr(self, "my_probes"):
                    probes_info["probes"] = self.my_probes[i]
                probes_info["probes_rst"] = self.my_probes_rst[i]
                probes_info["el_owner"] = self.my_el_owner[i]
                probes_info["err_code"] = self.my_err_code[i]

                settings = {}
                settings["progress_bar"] = self.progress_bar

                sampled_field_at_probe.append(
                    self.ei.interpolate_field_from_rst(
                        probes_info,
                        interpolation_buffer=self.test_interp,
                        sampled_field=sampled_field,
                        settings=settings,
                    )
                )

        else:
            # The inputs were in rank 0

            # Probes info
            probes_info = {}
            if hasattr(self, "my_probes"):
                probes_info["probes"] = self.my_probes
            probes_info["probes_rst"] = self.my_probes_rst
            probes_info["el_owner"] = self.my_el_owner
            probes_info["err_code"] = self.my_err_code

            # Settings
            settings = {}
            settings["progress_bar"] = self.progress_bar

            sampled_field_at_probe = self.ei.interpolate_field_from_rst(
                probes_info,
                interpolation_buffer=self.test_interp,
                sampled_field=sampled_field,
                settings=settings,
            )

        self.log.toc()

        return sampled_field_at_probe


def pt_in_bbox(pt, bbox, rel_tol=0.01):
    """Determine if the point is in the bounding box"""
    # rel_tol=1% enlargement of the bounding box by default

    state = False
    found_x = False
    found_y = False
    found_z = False

    d = bbox[1] - bbox[0]
    tol = d * rel_tol / 2
    if pt[0] >= bbox[0] - tol and pt[0] <= bbox[1] + tol:
        found_x = True

    d = bbox[3] - bbox[2]
    tol = d * rel_tol / 2
    if pt[1] >= bbox[2] - tol and pt[1] <= bbox[3] + tol:
        found_y = True

    d = bbox[5] - bbox[4]
    tol = d * rel_tol / 2
    if pt[2] >= bbox[4] - tol and pt[2] <= bbox[5] + tol:
        found_z = True

    if found_x is True and found_y is True and found_z is True:
        state = True
    else:
        state = False

    return state

def pt_in_bbox_vectorized(pt, bboxes, rel_tol=0.01):
    """
    Check if a point is inside multiple bounding boxes.
    
    Parameters:
        pt : array-like of shape (3,)
            The (x, y, z) coordinates of the point.
        bboxes : np.ndarray of shape (N, 6)
            Each row is [xmin, xmax, ymin, ymax, zmin, zmax].
        rel_tol : float
            Relative tolerance to expand each bounding box.
    
    Returns:
        valid_mask : np.ndarray of shape (N,)
            Boolean array indicating whether the point is inside each bounding box.
    """
    # For x dimension:
    dx = bboxes[:, 1] - bboxes[:, 0]
    tol_x = dx * rel_tol / 2.0
    lower_x = bboxes[:, 0] - tol_x
    upper_x = bboxes[:, 1] + tol_x

    # For y dimension:
    dy = bboxes[:, 3] - bboxes[:, 2]
    tol_y = dy * rel_tol / 2.0
    lower_y = bboxes[:, 2] - tol_y
    upper_y = bboxes[:, 3] + tol_y

    # For z dimension:
    dz = bboxes[:, 5] - bboxes[:, 4]
    tol_z = dz * rel_tol / 2.0
    lower_z = bboxes[:, 4] - tol_z
    upper_z = bboxes[:, 5] + tol_z

    return ((pt[0] >= lower_x) & (pt[0] <= upper_x) &
            (pt[1] >= lower_y) & (pt[1] <= upper_y) &
            (pt[2] >= lower_z) & (pt[2] <= upper_z))

def get_bbox_from_coordinates(x, y, z):
    """Determine if point is inside bounding box"""

    nelv = x.shape[0]
    # lx = x.shape[3]  # This is not a mistake. This is how the data is read
    # ly = x.shape[2]
    # lz = x.shape[1]

    bbox = np.zeros((nelv, 6), dtype=INTERPOLATION_DTYPE)

    for e in range(0, nelv):
        bbox[e, 0] = np.min(x[e, :, :, :])
        bbox[e, 1] = np.max(x[e, :, :, :])
        bbox[e, 2] = np.min(y[e, :, :, :])
        bbox[e, 3] = np.max(y[e, :, :, :])
        bbox[e, 4] = np.min(z[e, :, :, :])
        bbox[e, 5] = np.max(z[e, :, :, :])

    return bbox

def get_bbox_from_coordinates_rtree(x, y, z, rel_tol=0.01):

    nelv = x.shape[0]
    # lx = x.shape[3]  # This is not a mistake. This is how the data is read
    # ly = x.shape[2]
    # lz = x.shape[1]

    bbox = np.zeros((nelv, 6), dtype=INTERPOLATION_DTYPE)

    bbox[:, 0] = np.min(x, axis=(1, 2, 3))
    bbox[:, 1] = np.min(y, axis=(1, 2, 3))
    bbox[:, 2] = np.min(z, axis=(1, 2, 3))
    bbox[:, 3] = np.max(x, axis=(1, 2, 3))
    bbox[:, 4] = np.max(y, axis=(1, 2, 3))
    bbox[:, 5] = np.max(z, axis=(1, 2, 3))

    dx = bbox[:, 3] - bbox[:, 0]
    tol_x = dx * rel_tol / 2.0

    dy = bbox[:, 4] - bbox[:, 1]
    tol_y = dy * rel_tol / 2.0

    dz = bbox[:, 5] - bbox[:, 2]
    tol_z = dz * rel_tol / 2.0

    bbox[:, 0] -= tol_x
    bbox[:, 1] -= tol_y
    bbox[:, 2] -= tol_z
    bbox[:, 3] += tol_x
    bbox[:, 4] += tol_y
    bbox[:, 5] += tol_z

    return bbox
    
def create_rtee(bbox):
    """
    """
    # Create 3D rtree index
    prop = rtree_index.Property()
    prop.dimension = 3
    idx = rtree_index.Index(properties=prop)

    for e in range(bbox.shape[0]):
        bbox_e = bbox[e, :]
        idx.insert(e, (bbox_e[0], bbox_e[1], bbox_e[2], bbox_e[3], bbox_e[4], bbox_e[5]))

    return idx

def get_bbox_centroids_and_max_dist(bbox):
    """
    Get centroids from the bounding boxes.
    """

    # Then find the centroids of each bbox and the maximun bbox radious from centroid to corner
    bbox_dist = np.zeros((bbox.shape[0], 3), dtype=INTERPOLATION_DTYPE)
    bbox_dist[:, 0] = bbox[:, 1] - bbox[:, 0]
    bbox_dist[:, 1] = bbox[:, 3] - bbox[:, 2]
    bbox_dist[:, 2] = bbox[:, 5] - bbox[:, 4]

    bbox_max_dist = np.max(
        np.sqrt(bbox_dist[:, 0] ** 2 + bbox_dist[:, 1] ** 2 + bbox_dist[:, 2] ** 2) / 2
    )

    bbox_centroid = np.zeros((bbox.shape[0], 3), dtype=INTERPOLATION_DTYPE)
    bbox_centroid[:, 0] = bbox[:, 0] + bbox_dist[:, 0] / 2
    bbox_centroid[:, 1] = bbox[:, 2] + bbox_dist[:, 1] / 2
    bbox_centroid[:, 2] = bbox[:, 4] + bbox_dist[:, 2] / 2

    return bbox_centroid, bbox_max_dist * (1 + 1e-2)


def get_communication_pairs(self, global_rank_candidate_dict, comm):
    """Get the pairs of ranks that will communicate"""
    size = comm.Get_size()

    # Create a list with all the ranks
    ranks = [ii for ii in range(0, size)]

    # Get all unique pairs and colours
    if isinstance(global_rank_candidate_dict, NoneType):
        pairs = list(combinations(ranks, 2))
    else:
        pairs_temp = [
            (i, j) for i in range(0, size) for j in global_rank_candidate_dict[i]
        ]
        pairs = []
        for tt in range(0, len(pairs_temp)):
            i = pairs_temp[tt][0]
            j = pairs_temp[tt][1]
            if (i, j) not in pairs and (j, i) not in pairs and i != j:
                pairs.append((i, j))

    # Find the pairs in which my rank is part of
    rank = comm.Get_rank()
    my_pairs = [(i, j) for i, j in pairs if i == rank or j == rank]
    my_source_dest = [i if i != rank else j for (i, j) in my_pairs]
    my_source_dest.append(rank)
    my_source_dest.sort()

    return my_pairs, my_source_dest


def domain_binning_map_bin_to_rank(mesh_to_bin, nx, ny, nz, comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a bin to mesh map by checking if any point
    # in the SEM mesh resides in bin mesh cell
    mesh_to_bin = [item for sublist in mesh_to_bin for item in sublist]
    mesh_to_bin = np.unique(mesh_to_bin)
    mesh_to_bin_map = np.zeros((1, nx * ny * nz), dtype=np.int64)
    mesh_to_bin_map[0, mesh_to_bin] = np.int64(1)
    # mesh_to_bin_map indicates that this rank has points in the cells of
    # the bin mesh that have been marked with a 1.
    # Now gather the mesh to bin map of all ranks
    global_mesh_to_bin_map = np.zeros((size * nx * ny * nz), dtype=np.int64)
    comm.Allgather(
        [mesh_to_bin_map.flatten(), MPI.INT], [global_mesh_to_bin_map, MPI.INT]
    )
    global_mesh_to_bin_map = global_mesh_to_bin_map.reshape((size, nx * ny * nz))

    # Create a dictionary that has the bin to the associated rank list
    bin_to_rank_map = {"bin_subdivision": "associated rank list"}
    for i in range(nx * ny * nz):
        incidences = global_mesh_to_bin_map[:, i]
        bin_to_rank_map[i] = np.where(incidences == 1)[0]

    return bin_to_rank_map


def get_candidate_ranks(self, comm):
    """
    Get the candidate ranks for each rank.
    """

    if self.global_tree_type == "rank_bbox":
        # Obtain the candidates of each point
        ## Do it in a batched way to avoid memory issues

        chunk_size = self.max_pts
        n_chunks = int(np.ceil(self.probe_partition.shape[0] / chunk_size))

        # Attempting to optimize. Currently using the old version (the one in else.)
        # Consider activating the one on top by setting 0==0. Maybe it is better for more ranks
        candidate_ranks_per_point = []

        for chunk_id in range(0, n_chunks):

            start = chunk_id * chunk_size
            end = (chunk_id + 1) * chunk_size
            if end > self.probe_partition.shape[0]:
                end = self.probe_partition.shape[0]

            candidate_ranks_per_point_ = self.global_tree.query_ball_point(
                x=self.probe_partition[start:end],
                r=self.search_radious * (1 + 1e-6),
                p=2.0,
                eps=1e-8,
                workers=1,
                return_sorted=False,
                return_length=False,
            )

            refined_candidates = refine_candidates(probes=self.probe_partition[start:end], candidate_elements=candidate_ranks_per_point_, bboxes=self.global_bbox, rel_tol=0.01)

            candidate_ranks_per_point.extend(refined_candidates)

        # Give it the same format that the kdtree search gives   
        candidate_ranks_per_point = np.array(candidate_ranks_per_point, dtype=object)
         
        # Obtain the unique candidates of this rank
        ## 1. flatten the list of lists
        flattened_list = [
            item for sublist in candidate_ranks_per_point for item in sublist
        ]
        ## 2. count the number of times each rank appears
        counts = collections_counter(flattened_list)
        ## 3. filter the ranks that appear more than once
        candidate_ranks = list(set(flattened_list))
        ## 4. sort the ranks by the number of points that has it as candidate
        candidate_ranks.sort(key=lambda x: counts[x], reverse=True)

    elif self.global_tree_type == "domain_binning":

        # Calculate the number of chunks once
        chunk_size = self.max_pts
        num_points = self.probe_partition.shape[0]
        n_chunks = int(np.ceil(num_points / chunk_size))

        # Check for the maximun number of chunks across ranks
        n_chunks = comm.allreduce(n_chunks, op=MPI.MAX)

        candidate_ranks_per_point = []
        iferror = False

        for chunk_id in range(n_chunks):
            # Determine start and end indices for this chunk
            start = chunk_id * chunk_size
            end = min((chunk_id + 1) * chunk_size, num_points)
            
            # Extract coordinates for the current chunk in one go
            chunk_pts = self.probe_partition[start:end]
            xx, yy, zz = chunk_pts[:, 0], chunk_pts[:, 1], chunk_pts[:, 2]
            
            # Compute the bin for each probe point in this chunk
            probe_to_bin = self.binning_hash(xx, yy, zz)
            
            # Compute unique bins from the probe points and then find their owners
            unique_probe_to_bin = np.unique(probe_to_bin)
            # Compute owner per unique bin (vectorized floor division)
            owners = np.floor_divide(unique_probe_to_bin, self.bins_per_rank).astype(np.int32)
            # Get unique owner values and preserve their order
            unique_owners, _ = np.unique(owners, return_index=True)
            unique_probe_to_bin_owner = unique_owners.tolist()  # Now a list of owner IDs
            
            # For each owner, group the bins that belong to it
            bin_data = [
                unique_probe_to_bin[owners == owner].astype(np.int32)
                for owner in unique_probe_to_bin_owner
            ]
            
            # Communicate: send unique bin information to remote processes
            sources, data_from_others = self.rt.send_recv(
                destination=unique_probe_to_bin_owner, data=bin_data, dtype=np.int32, tag=0
            )
            
            # For each result from the remote call, build the candidate arrays:
            return_data = []
            for bins in data_from_others:
                # Build rank arrays for each bin using the mapping, then create a companion bin array
                rank_arrays = [np.array(self.bin_to_rank_map[b], dtype=np.int32) for b in bins]
                bin_arrays = [
                    np.full_like(rank, fill_value=b, dtype=np.int32)
                    for b, rank in zip(bins, rank_arrays)
                ]
                # Concatenate across bins and stack into a 2D array: first column for bin, second for rank candidate
                concatenated_ranks = np.concatenate(rank_arrays)
                concatenated_bins = np.concatenate(bin_arrays)
                return_data.append(np.stack((concatenated_bins, concatenated_ranks), axis=1))
            
            # Communicate: send back the return_data and receive the final candidate data from remote nodes
            _, returned_data = self.rt.send_recv(
                destination=sources, data=return_data, dtype=np.int32, tag=1
            )
            
            # For efficiency, reshape each array only once and build a mapping from owner to candidate array
            owner_to_data = {
                owner: ret.reshape(-1, 2)
                for owner, ret in zip(unique_probe_to_bin_owner, returned_data)
            }
            
            try:
                candidate_ranks_per_point_ = [
                    ret_data[ret_data[:, 0] == b, 1].tolist()
                    for b, ret_data in ((b, owner_to_data[int(b // self.bins_per_rank)]) for b in probe_to_bin)
                ]
            except KeyError:
                iferror = True
                self.log.write("error", f"Something is failing with the domain partitioning to interpolate. You have {self.bins_per_rank} bins per rank. Consider decreasing it first and then increasing it. If it does not work, change the global_tree_type to rank_bbox when initializing probes.")
            
            self.rt.comm.Barrier()
            stopping = self.rt.comm.allreduce(iferror, op=MPI.SUM)
            if stopping > 0:
                self.log.write("error", "Error in domain partitioning. Exiting.")
                sys.exit(1)

            candidate_ranks_per_point.extend(candidate_ranks_per_point_)

        # Format the candidate ranks to match the desired object array output
        candidate_ranks_per_point = np.array(candidate_ranks_per_point, dtype=object)

        # Now, from the candidates per point, get the candidates
        # that this rank has.
        # 1. Flatten the list of lists
        flattened_list = [
            item for sublist in candidate_ranks_per_point for item in sublist
        ]
        # 2. Count the number of times each rank appears
        counts = collections_counter(flattened_list)
        # 3. Filter the ranks that appear more than once
        candidate_ranks = list(set(flattened_list))
        # 4. Sort the ranks by the number of points that has it as candidate
        candidate_ranks.sort(key=lambda x: counts[x], reverse=True)
    else:
        raise ValueError("Global tree has not been set up")

    return candidate_ranks, candidate_ranks_per_point


def domain_binning_map_probe_to_rank(self, probe_to_bin_map):

    # Now for each probe use the bin to rank map
    # to find the candidate ranks for each probe
    probe_to_rank = [
        [self.bin_to_rank_map[bin] for bin in probe_to_bin_map[i]]
        for i in range(len(probe_to_bin_map))
    ]
    # In the previous map every point gets a set of lists, now make it
    # just one list with the unique ranks.
    probe_to_rank_map = np.array([
        np.unique([item for sublist in probe_to_rank[i] for item in sublist]).tolist()
        for i in range(len(probe_to_rank))
    ], dtype = object)

    return probe_to_rank_map


def get_global_candidate_ranks(comm, candidate_ranks):
    """Get an array with the candidate ranks of all ranks"""

    size = comm.Get_size()

    # Get arrays with number of candidates per rank
    ## Tell how many there are per rank
    n_candidate_ranks_per_rank = np.zeros((size), dtype=np.int64)
    sendbuf = np.ones((1), dtype=np.int64) * len(candidate_ranks)
    comm.Allgather([sendbuf, MPI.INT], [n_candidate_ranks_per_rank, MPI.INT])

    ## Allocate an array in all ranks that tells which are the rank candidates in all other ranks
    nc = np.max(n_candidate_ranks_per_rank)
    rank_candidates_in_all_ranks = np.zeros((size, nc), dtype=np.int64)
    rank_candidates_in_this_rank = (
        np.ones((1, nc), dtype=np.int64) * -1
    )  # Set default to -1 to filter out easily later
    for i in range(0, len(candidate_ranks)):
        rank_candidates_in_this_rank[0, i] = candidate_ranks[i]
    comm.Allgather(
        [rank_candidates_in_this_rank, MPI.INT], [rank_candidates_in_all_ranks, MPI.INT]
    )  # This all gather can be changed for gather and broadcast

    return rank_candidates_in_all_ranks

def fp_it_tolist(self, value):
    if type(value) is int:
        self.log.write("warning", "find_points_iterative must be a list or tuple, received int. Converting to list")
        return [True, value]
    elif type(value) is bool:
        self.log.write("warning", "find_points_iterative must be a list or tuple, received bool. Converting to list")
        self.log.write("warning", "Setting comm batch size to 5000. Only used if find_points_iterative is True")
        return [value, 5000]
    return value

def pack_data(array_list: list = None):

    """
    Pack data into a single array for sending over MPI.
    
    Parameters:
        array_list : list of np.ndarray
            List of arrays to be packed.
    
    Returns:
        packed_data : np.ndarray
            Single packed array.
    """
    if array_list is None:
        return None

    # Concatenate all arrays into a single array
    packed_data = np.concatenate([arr.flatten() for arr in array_list])
    return packed_data

def unpack_data(packed_data: np.ndarray = None, number_of_arrays: int = None, equal_length: bool = True, final_shape: tuple = None):
    """
    """
    if packed_data is None:
        return []

    unpacked_data = []

    if equal_length:
        array_size = packed_data.size // number_of_arrays
        for i in range(number_of_arrays):
            start_index = i * array_size
            end_index = (i + 1) * array_size
            upacked = packed_data[start_index:end_index].view()
            if final_shape is not None:
                upacked.shape = final_shape
                unpacked_data.append(upacked)
            else:   
                unpacked_data.append(upacked)
    else:
        raise NotImplementedError("Unpacking with different sizes is not implemented yet")
        # Calculate the size of each array
    return unpacked_data

def pack_destination_data(destination_data: list =  None):
    """
    """

    number_of_arrays =  len(destination_data)
    number_of_destinations = len(destination_data[0])
    output_buffers = []
    for destination_index in range(0, number_of_destinations):
        arrays_to_pack = [destination_data[i][destination_index] for i in range(0, number_of_arrays)]
        packed_destination_data = pack_data(arrays_to_pack)
        output_buffers.append(packed_destination_data)
    return output_buffers

def unpack_source_data(packed_source_data: list = None, number_of_arrays: int = 1, equal_length: bool = True, final_shape: tuple = None):
    """
    """

    # Allocate
    output_buffers = [[] for _ in range(number_of_arrays)]
    number_of_sources = len(packed_source_data)
    
    for source_index in range(0, number_of_sources):
        unpacked_source_data = unpack_data(packed_source_data[source_index], number_of_arrays=number_of_arrays, equal_length=equal_length, final_shape=final_shape)
        for arrays_index in range(0, number_of_arrays):
            output_buffers[arrays_index].append(unpacked_source_data[arrays_index])

    return output_buffers


from abc import ABC, abstractmethod


class dstructure(ABC):
    """Interface for multiple point interpolators"""

    def __init__(self):
        """Initialize"""

    @abstractmethod
    def search(self):
        """search the points"""

class dstructure_kdtree(dstructure):
    """
    """

    def __init__(self, logger, x: np.ndarray, y: np.ndarray, z: np.ndarray, **kwargs):
        """Initialize"""
        super().__init__()

        self.log = logger
        self.elem_percent_expansion = kwargs.get("elem_percent_expansion", 0.01)
        self.max_pts = kwargs.get("max_pts", 128)
        self.use_obb = kwargs.get("use_oriented_bbox", False)
        ei = kwargs.get("point_interpolator", None)

        # First each rank finds their bounding box
        self.log.write("info", "Finding bounding box of sem mesh")
        self.my_bbox = get_bbox_from_coordinates(x, y, z)
        
        # Get the oriented bbox data
        if self.use_obb:
            if hasattr(ei, "get_obb"): 
                self.log.write("info", "Finding oriented bounding box of sem mesh")
                self.obb_c, self.obb_jinv = ei.get_obb(x, y, z, max_pts=self.max_pts)
            else:
                self.log.write("error", "You are trying to use the OBB feature, but the ei object does not have the get_obb method. Please check your code.")
                raise ValueError("The ei object does not have the get_obb method. Please check your code.")

        # Expand the bounding boxes just once to make it faster later
        self.expanded_bbox = np.empty_like(self.my_bbox) 
        rel_tol = self.elem_percent_expansion
        # For x
        dx = self.my_bbox[:, 1] - self.my_bbox[:, 0]
        tol_x = dx * rel_tol / 2.0
        self.expanded_bbox[:, 0] = self.my_bbox[:, 0] - tol_x
        self.expanded_bbox[:, 1] = self.my_bbox[:, 1] + tol_x
        # For y
        dy = self.my_bbox[:, 3] - self.my_bbox[:, 2]
        tol_y = dy * rel_tol / 2.0
        self.expanded_bbox[:, 2] = self.my_bbox[:, 2] - tol_y
        self.expanded_bbox[:, 3] = self.my_bbox[:, 3] + tol_y
        # For z
        dz = self.my_bbox[:, 5] - self.my_bbox[:, 4]
        tol_z = dz * rel_tol / 2.0
        self.expanded_bbox[:, 4] = self.my_bbox[:, 4] - tol_z
        self.expanded_bbox[:, 5] = self.my_bbox[:, 5] + tol_z
    
        # Get bbox centroids and max radius from center to corner
        self.my_bbox_centroids, self.my_bbox_maxdist = (
            get_bbox_centroids_and_max_dist(self.my_bbox)
        )

        # Build a KDtree with my information
        self.log.write("info", "Creating KD tree with local bbox centroids") 
        self.my_tree = KDTree(self.my_bbox_centroids)

    def search(self, probes: np.ndarray, progress_bar = False, **kwargs):

        chunk_size = self.max_pts
        n_chunks = int(np.ceil(probes.shape[0] / chunk_size))
        element_candidates = []

        for chunk_id in range(n_chunks):
            start = chunk_id * chunk_size
            end = (chunk_id + 1) * chunk_size
            if end > probes.shape[0]:
                end = probes.shape[0]

            candidate_elements = self.my_tree.query_ball_point(
                x=probes[start:end],
                r=self.my_bbox_maxdist * (1 + 1e-6),
                p=2.0,
                eps=self.elem_percent_expansion,
                workers=1,
                return_sorted=False,
                return_length=False,
            )

            # New way of checking as of april 4 2025
            # I am already passing the expanded bounding box, so I take relative tolerance = 0
            element_candidates_ = refine_candidates(probes[start:end], candidate_elements, self.expanded_bbox, rel_tol=0)

            # Add a new refinement with the obb as well
            if self.use_obb: 
                element_candidates__ = refine_candidates_obb(probes[start:end], element_candidates_, self.obb_c, self.obb_jinv)
                element_candidates_ = element_candidates__

            # Extend with the chunked data
            element_candidates.extend(element_candidates_)
            
        if self.use_obb:
            self.log.write("info", "obb was used to refine search")

        return element_candidates

class dstructure_rtree(dstructure):
    """
    """

    def __init__(self, logger, x: np.ndarray, y: np.ndarray, z: np.ndarray, **kwargs):
        """Initialize"""
        super().__init__()

        self.log = logger
        self.elem_percent_expansion = kwargs.get("elem_percent_expansion", 0.01)
        self.max_pts = kwargs.get("max_pts", 128)
        self.use_obb = kwargs.get("use_oriented_bbox", False)
        ei = kwargs.get("point_interpolator", None)

        if rtree_index is None:
            raise ImportError(
                "Rtree is not installed, please install it with pip install rtree to use this feature"
            )
            
        self.log.write("info", "Finding bounding box of sem mesh")
        self.my_bbox = get_bbox_from_coordinates_rtree(x, y, z, rel_tol=self.elem_percent_expansion)
        
        # Get the oriented bbox data
        if self.use_obb:
            if hasattr(ei, "get_obb"): 
                self.log.write("info", "Finding oriented bounding box of sem mesh")
                self.obb_c, self.obb_jinv = ei.get_obb(x, y, z, max_pts=self.max_pts)
            else:
                self.log.write("error", "You are trying to use the OBB feature, but the ei object does not have the get_obb method. Please check your code.")
                raise ValueError("The ei object does not have the get_obb method. Please check your code.")

        self.log.write("info", "Creating Rtree with local bbox centroids")
        self.my_tree = create_rtee(self.my_bbox) 
            
    def search(self, probes: np.ndarray, **kwargs):
        
        element_candidates = []
        for pt in range(probes.shape[0]):
            query_point_ = (probes[pt, 0], probes[pt, 1], probes[pt, 2])
            query_point = query_point_ + query_point_
            element_candidates.append(list(self.my_tree.intersection(query_point)))
            
        # Add a new refinement with the obb as well  
        if self.use_obb:
                
            element_candidates_ = refine_candidates_obb(probes, element_candidates, self.obb_c, self.obb_jinv)
            element_candidates = element_candidates_
            
            self.log.write("info", "obb was used to refine search")

        return element_candidates

class dstructure_hashtable(dstructure):
    """
    """

    def __init__(self, logger, x: np.ndarray, y: np.ndarray, z: np.ndarray, **kwargs):
        """Initialize"""
        super().__init__()

        self.log = logger
        self.elem_percent_expansion = kwargs.get("elem_percent_expansion", 0.01)
        self.max_pts = kwargs.get("max_pts", 128)
        self.use_obb = kwargs.get("use_oriented_bbox", False)
        ei = kwargs.get("point_interpolator", None)
        
        # First each rank finds their bounding box
        self.log.write("info", "Finding bounding box of sem mesh")
        self.my_bbox = get_bbox_from_coordinates(x, y, z)
        
        # Get the oriented bbox data
        if self.use_obb:
            if hasattr(ei, "get_obb"): 
                self.log.write("info", "Finding oriented bounding box of sem mesh")
                self.obb_c, self.obb_jinv = ei.get_obb(x, y, z, max_pts=self.max_pts)
            else:
                self.log.write("error", "You are trying to use the OBB feature, but the ei object does not have the get_obb method. Please check your code.")
                raise ValueError("The ei object does not have the get_obb method. Please check your code.")

        # Make the mesh fill the space of its bounding box
        self.log.write("info", "Filling bbox space for correct hashtable finding")
        x_r, y_r, z_r, _ = linearize_elements(x, y, z, factor=2, rel_tol=self.elem_percent_expansion)

        bin_size = x.shape[0]
        bin_size_1d = int(np.round(np.cbrt(bin_size))) 
        bin_size = bin_size_1d**3

        # Find the values that delimit a cubic boundin box
        # for the whole domain
        self.log.write("info", "Finding bounding box tha delimits the ranks")
        
        self.domain_min_x = np.min(x)
        self.domain_min_y = np.min(y)
        self.domain_min_z = np.min(z)
        self.domain_max_x = np.max(x)
        self.domain_max_y = np.max(y)
        self.domain_max_z = np.max(z) 
        self.bin_size_1d = bin_size_1d

        # See wich element has points in which bin
        self.log.write("info", "Creating bin mesh for the rank")
        bins_of_points = self.binning_hash(x_r, y_r, z_r)
        
        # Create the empty bin to rank map
        approach = 1
        if approach == 0:
        
            self.bin_to_elem_map = {i : (np.unique(np.where(bins_of_points == i)[0]).astype(np.int32)).tolist() for i in range(0, bin_size)}

        # Optimized 
        elif approach == 1:

            nelv = bins_of_points.shape[0]
            bin_size = bin_size_1d ** 3

            # Initialize an empty dictionary
            bin_to_elem_map = {}

            # Fill the bins
            for el in range(nelv):
                
                unique_bins = np.unique(bins_of_points[el])
                for b in unique_bins:
                    if b not in bin_to_elem_map:
                        bin_to_elem_map[b] = []
                    
                    if el not in bin_to_elem_map[b]:
                        bin_to_elem_map[b].append(el)

            # Fill the empty bins for completion
            for i in range(bin_size):
                if i not in bin_to_elem_map:
                    bin_to_elem_map[i] = []

            self.bin_to_elem_map = bin_to_elem_map


    def binning_hash(self, x, y, z):
        """
        """

        x_min = self.domain_min_x
        x_max = self.domain_max_x
        y_min = self.domain_min_y
        y_max = self.domain_max_y
        z_min = self.domain_min_z
        z_max = self.domain_max_z
        n_bins_1d = self.bin_size_1d
        max_bins_1d = n_bins_1d - 1

        bin_x = (np.floor((x - x_min) / ((x_max - x_min) / max_bins_1d))).astype(np.int32)
        bin_y = (np.floor((y - y_min) / ((y_max - y_min) / max_bins_1d))).astype(np.int32)
        bin_z = (np.floor((z - z_min) / ((z_max - z_min) / max_bins_1d))).astype(np.int32)

        # Clip the bins to be in the range [0, n_bins_1d - 1]
        bin_x = np.clip(bin_x, 0, max_bins_1d)
        bin_y = np.clip(bin_y, 0, max_bins_1d)
        bin_z = np.clip(bin_z, 0, max_bins_1d)

        bin = bin_x + bin_y * n_bins_1d + bin_z * n_bins_1d**2
        
        return bin

            
    def search(self, probes: np.ndarray, **kwargs):

        probe_to_bin = self.binning_hash(probes[:, 0], probes[:, 1], probes[:, 2])

        element_candidates = [self.bin_to_elem_map[probe_to_bin[i]] for i in range(0, probes.shape[0])]

        element_candidates = refine_candidates(probes, element_candidates, self.my_bbox, rel_tol=self.elem_percent_expansion)
        
        if self.use_obb:
                
            element_candidates_ = refine_candidates_obb(probes, element_candidates, self.obb_c, self.obb_jinv)
            element_candidates = element_candidates_
            
            self.log.write("info", "obb was used to refine search")

        return element_candidates

def refine_candidates(probes, candidate_elements, bboxes, rel_tol=0.01):
    """
    Refine candidate elements for each probe by keeping only those where the probe 
    lies within the corresponding expanded bounding box.

    Note: mutates `candidate_elements` by trimming consumed candidates so that
    a single probe with many candidates can be processed across multiple batches.
    """

    max_batch_size = INTERPOLATION_MAX_CANDIDATE_IN_ITERATION
    refined_candidates = [[] for _ in range(probes.shape[0])]
    start = 0
    end = probes.shape[0]

    while start < end:
        probe_indices = []
        candidate_indices = []
        it_entries = 0
        i_partial = False  # did we only partially consume probe i?

        for i in range(start, end):
            cands = candidate_elements[i]
            if cands:
                remaining = max_batch_size - it_entries
                if remaining <= 0:
                    break

                if len(cands) > remaining:
                    # take only what fits, leave the rest for the next loop
                    probe_indices.extend([i] * remaining)
                    candidate_indices.extend(cands[:remaining])
                    candidate_elements[i] = cands[remaining:]  # mutate: keep leftovers
                    it_entries += remaining
                    i_partial = True
                    break
                else:
                    # consume all of this probe's candidates
                    probe_indices.extend([i] * len(cands))
                    candidate_indices.extend(cands)
                    it_entries += len(cands)
                    candidate_elements[i] = []  # mark consumed

                if it_entries >= max_batch_size:
                    break

        # If we partially consumed probe i, revisit it next; otherwise move past i
        if i_partial:
            start = i  # same probe still has leftovers
        else:
            # If the loop ran at least once, `i` is defined; otherwise start==end and we won't get here
            start = i + 1 if 'i' in locals() else end

        # Nothing gathered this round; continue to next window (e.g., all empties)
        if not probe_indices:
            continue

        probe_indices = np.asarray(probe_indices, dtype=np.intp)
        candidate_indices = np.asarray(candidate_indices, dtype=np.intp)

        # Points and candidate bboxes for the gathered pairs
        pts = probes[probe_indices]
        candidate_bboxes = bboxes[candidate_indices]

        # Compute expanded bounds
        if rel_tol > 1e-6:
            dx = candidate_bboxes[:, 1] - candidate_bboxes[:, 0]
            dy = candidate_bboxes[:, 3] - candidate_bboxes[:, 2]
            dz = candidate_bboxes[:, 5] - candidate_bboxes[:, 4]
            tol_x = dx * rel_tol / 2.0
            tol_y = dy * rel_tol / 2.0
            tol_z = dz * rel_tol / 2.0
            lower_x = candidate_bboxes[:, 0] - tol_x
            upper_x = candidate_bboxes[:, 1] + tol_x
            lower_y = candidate_bboxes[:, 2] - tol_y
            upper_y = candidate_bboxes[:, 3] + tol_y
            lower_z = candidate_bboxes[:, 4] - tol_z
            upper_z = candidate_bboxes[:, 5] + tol_z
        else:
            lower_x = candidate_bboxes[:, 0]; upper_x = candidate_bboxes[:, 1]
            lower_y = candidate_bboxes[:, 2]; upper_y = candidate_bboxes[:, 3]
            lower_z = candidate_bboxes[:, 4]; upper_z = candidate_bboxes[:, 5]

        # Inside-AABB test
        valid_mask = (
            (pts[:, 0] >= lower_x) & (pts[:, 0] <= upper_x) &
            (pts[:, 1] >= lower_y) & (pts[:, 1] <= upper_y) &
            (pts[:, 2] >= lower_z) & (pts[:, 2] <= upper_z)
        )

        if not np.any(valid_mask):
            continue

        valid_probe_indices = probe_indices[valid_mask]
        valid_candidate_indices = candidate_indices[valid_mask]

        order = np.argsort(valid_probe_indices, kind='stable')
        valid_probe_sorted = valid_probe_indices[order]
        valid_candidate_sorted = valid_candidate_indices[order]

        unique_probes, start_idx, counts = np.unique(
            valid_probe_sorted, return_index=True, return_counts=True
        )

        for probe, idx0, cnt in zip(unique_probes, start_idx, counts):
            refined_candidates[probe].extend(
                valid_candidate_sorted[idx0:idx0 + cnt].tolist()
            )

    return refined_candidates

def refine_candidates_obb(probes, candidate_elements, obb_c, obb_jinv):
    """
    Refine candidate elements for each probe by keeping only those where the probe 
    lies within the corresponding oriented bounding box (OBB).

    Notes:
      - Mutates `candidate_elements[i]` by trimming consumed indices.
      - Batches by total flattened candidate pairs, not number of probes.
    """
    max_batch_size = INTERPOLATION_MAX_CANDIDATE_IN_ITERATION
    refined_candidates = [[] for _ in range(probes.shape[0])]
    start = 0
    end = probes.shape[0]

    while start < end:
        probe_indices = []
        candidate_indices = []
        it_entries = 0
        i_partial = False  # did we only partially consume probe i?

        for i in range(start, end):
            cands = candidate_elements[i]
            if cands:
                remaining = max_batch_size - it_entries
                if remaining <= 0:
                    break

                if len(cands) > remaining:
                    # take only what fits, leave the rest
                    probe_indices.extend([i] * remaining)
                    candidate_indices.extend(cands[:remaining])
                    candidate_elements[i] = cands[remaining:]   # mutate: keep leftovers
                    it_entries += remaining
                    i_partial = True
                    break
                else:
                    # consume all candidates for this probe
                    probe_indices.extend([i] * len(cands))
                    candidate_indices.extend(cands)
                    it_entries += len(cands)
                    candidate_elements[i] = []  # consumed

                if it_entries >= max_batch_size:
                    break

        # If partially consumed, revisit same probe next; otherwise move past it
        if i_partial:
            start = i
        else:
            start = i + 1 if 'i' in locals() else end

        # Nothing gathered this round (e.g., all empties) -> continue
        if not probe_indices:
            continue

        probe_indices = np.asarray(probe_indices, dtype=np.intp)
        candidate_indices = np.asarray(candidate_indices, dtype=np.intp)

        # Gather points and OBB params for these pairs
        pts = probes[probe_indices]                    # (N, 3)
        candidate_bboxes_c = obb_c[candidate_indices]  # (N, 3)
        candidate_bboxes_jinv = obb_jinv[candidate_indices]  # (N, 3, 3)

        # Batched transform into each candidate's OBB local coordinates
        diff = (pts - candidate_bboxes_c).reshape(-1, 3, 1)          # (N, 3, 1)
        check = np.matmul(candidate_bboxes_jinv, diff).reshape(-1, 3) # (N, 3)
        check = np.abs(check)

        # Inside test with small tolerance (no extra alloc)
        valid_mask = np.all(check <= (1.0 + 1e-6), axis=1)

        if not np.any(valid_mask):
            continue

        valid_probe_indices = probe_indices[valid_mask]
        valid_candidate_indices = candidate_indices[valid_mask]

        # Group by probe and append
        order = np.argsort(valid_probe_indices, kind='stable')
        valid_probe_sorted = valid_probe_indices[order]
        valid_candidate_sorted = valid_candidate_indices[order]

        unique_probes, start_idx, counts = np.unique(
            valid_probe_sorted, return_index=True, return_counts=True
        )

        for probe, idx0, cnt in zip(unique_probes, start_idx, counts):
            refined_candidates[probe].extend(
                valid_candidate_sorted[idx0: idx0 + cnt].tolist()
            )

    return refined_candidates

def expand_elements(arrays: list = None, rel_tol = 0.01):
    '''Scale the elements to an expanded range'''
    
    rescaled_arrays = []
    for x in arrays:
        # Calculate scaling parameters
        element_min = np.min(x, axis=(1, 2, 3), keepdims=True)
        element_max = np.max(x, axis=(1, 2, 3), keepdims=True)
        delta = element_max - element_min
        new_min = element_min - delta * rel_tol / 2
        new_max = element_max + delta * rel_tol / 2

        # Rescale
        rescaled_arrays.append(new_min + (x - element_min) * (new_max - new_min) / delta)

    return rescaled_arrays

def linearize_elements(x, y, z, factor: int = 2, rel_tol = 0.01):
    '''Scale the elements to an expanded range'''
    
    # Calculate scaling parameters
    min_x = np.min(x, axis=(1, 2, 3))
    max_x = np.max(x, axis=(1, 2, 3))
    dx = max_x - min_x

    min_y = np.min(y, axis=(1, 2, 3))
    max_y = np.max(y, axis=(1, 2, 3))
    dy = max_y - min_y

    min_z = np.min(z, axis=(1, 2, 3))
    max_z = np.max(z, axis=(1, 2, 3))
    dz = max_z - min_z

    # Calculate the new min and max values
    new_min_x = min_x - dx * rel_tol / 2
    new_max_x = max_x + dx * rel_tol / 2
    new_min_y = min_y - dy * rel_tol / 2
    new_max_y = max_y + dy * rel_tol / 2
    new_min_z = min_z - dz * rel_tol / 2
    new_max_z = max_z + dz * rel_tol / 2

    # Create new linear grid over the elements
    nelv = x.shape[0]
    lz = x.shape[1]
    ly = x.shape[2]
    lx = x.shape[3]
    x_r = np.empty((nelv, factor*lz, factor*ly, factor*lx), dtype=x.dtype)
    y_r = np.empty((nelv, factor*lz, factor*ly, factor*lx), dtype=y.dtype)
    z_r = np.empty((nelv, factor*lz, factor*ly, factor*lx), dtype=z.dtype)
    for e in range(nelv):
        x_1d = np.linspace(new_min_x[e], new_max_x[e], lx*factor)
        y_1d = np.linspace(new_min_y[e], new_max_y[e], ly*factor)
        z_1d = np.linspace(new_min_z[e], new_max_z[e], lz*factor)

        zz, yy, xx = np.meshgrid(z_1d, y_1d, x_1d, indexing='ij')

        x_r[e] = xx
        y_r[e] = yy
        z_r[e] = zz

    bbox = np.zeros((nelv, 6), dtype=INTERPOLATION_DTYPE)
    bbox[:,0] = min_x
    bbox[:,1] = max_x
    bbox[:,2] = min_y
    bbox[:,3] = max_y
    bbox[:,4] = min_z
    bbox[:,5] = max_z
    

    return x_r, y_r, z_r, bbox
