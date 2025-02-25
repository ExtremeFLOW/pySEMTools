""" Contains the interpolator class"""

import os
from itertools import combinations
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from mpi4py import MPI  # for the timer
from .point_interpolator.point_interpolator_factory import get_point_interpolator
from ..monitoring.logger import Logger
from ..comm.router import Router
from collections import Counter as collections_counter

NoneType = type(None)

DEBUG = os.getenv("PYSEMTOOLS_DEBUG", "False").lower() in ("true", "1", "t")


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
    ):
        self.log = Logger(comm=comm, module_name="Interpolator")
        self.log.write("info", "Initializing Interpolator object")
        self.log.tic()

        # Instance communication object
        self.rt = Router(comm)

        self.x = x
        self.y = y
        self.z = z
        self.probes = probes

        self.point_interpolator_type = point_interpolator_type
        self.max_pts = max_pts
        self.max_elems = max_elems

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

        elif (find_points_comm_pattern == "point_to_point") or (find_points_comm_pattern == "collective"):
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
        rank_bbox = np.zeros((1, 6), dtype=np.double)
        rank_bbox[0, 0] = np.min(self.x)
        rank_bbox[0, 1] = np.max(self.x)
        rank_bbox[0, 2] = np.min(self.y)
        rank_bbox[0, 3] = np.max(self.y)
        rank_bbox[0, 4] = np.min(self.z)
        rank_bbox[0, 5] = np.max(self.z)

        # Gather the bounding boxes in all ranks
        self.global_bbox = np.zeros((size * 6), dtype=np.double)
        comm.Allgather(
            [rank_bbox.flatten(), MPI.DOUBLE], [self.global_bbox, MPI.DOUBLE]
        )
        self.global_bbox = self.global_bbox.reshape((size, 6))

        # Get the centroids and max distances
        bbox_dist = np.zeros((size, 3), dtype=np.double)
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

        bbox_centroid = np.zeros((size, 3))
        bbox_centroid[:, 0] = self.global_bbox[:, 0] + bbox_dist[:, 0] / 2
        bbox_centroid[:, 1] = self.global_bbox[:, 2] + bbox_dist[:, 1] / 2
        bbox_centroid[:, 2] = self.global_bbox[:, 4] + bbox_dist[:, 2] / 2

        # Create a tree with the rank centroids
        self.log.write("info", "Creating global KD tree with rank centroids")
        self.global_tree = KDTree(bbox_centroid)
        self.search_radious = bbox_max_dist

        self.log.toc()

    def set_up_global_tree_domain_binning_(self, comm, global_tree_nbins=None):

        self.log.tic()

        if isinstance(global_tree_nbins, NoneType):
            global_tree_nbins = comm.Get_size()
            self.log.write(
                "info", f"nbins not provided, using {global_tree_nbins} as default"
            )

        bin_size = global_tree_nbins

        self.log.write("info", f"Using global bin mesh of size {bin_size}")

        # Find the values that delimit a cubic boundin box
        # for the whole domain
        self.log.write("info", "Finding bounding box tha delimits the whole domain")
        rank_bbox = np.zeros((1, 6), dtype=np.double)
        rank_bbox[0, 0] = np.min(self.x)
        rank_bbox[0, 1] = np.max(self.x)
        rank_bbox[0, 2] = np.min(self.y)
        rank_bbox[0, 3] = np.max(self.y)
        rank_bbox[0, 4] = np.min(self.z)
        rank_bbox[0, 5] = np.max(self.z)
        domain_min_x = comm.allreduce(rank_bbox[0, 0], op=MPI.MIN)
        domain_min_y = comm.allreduce(rank_bbox[0, 2], op=MPI.MIN)
        domain_min_z = comm.allreduce(rank_bbox[0, 4], op=MPI.MIN)
        domain_max_x = comm.allreduce(rank_bbox[0, 1], op=MPI.MAX)
        domain_max_y = comm.allreduce(rank_bbox[0, 3], op=MPI.MAX)
        domain_max_z = comm.allreduce(rank_bbox[0, 5], op=MPI.MAX)

        ## Find the ratio between ditances for domains that are
        ## not well distributed
        per_ = (
            domain_max_x
            - domain_min_x
            + domain_max_y
            - domain_min_y
            + domain_max_z
            - domain_min_z
        )
        ratiox = (domain_max_x - domain_min_x) / per_
        ratioy = (domain_max_y - domain_min_y) / per_
        ratioz = (domain_max_z - domain_min_z) / per_

        ## Select the number of points trying to respect the
        ## ratio constraints
        nz = int(np.round(np.cbrt(bin_size * ratioz)))
        remaining = bin_size / nz  # This is the product of nx and ny we aim for
        sum_ratios = ratiox + ratioy
        nx = int(np.round(np.sqrt((ratiox / sum_ratios) * remaining)))
        ny = int(np.round(np.sqrt((ratioy / sum_ratios) * remaining)))
        nz = int(bin_size / (nx * ny))

        self.log.write("info", "Creating global bin mesh")
        # create bin linear mesh
        bin_x = np.linspace(domain_min_x, domain_max_x, nx + 1)
        bin_y = np.linspace(domain_min_y, domain_max_y, ny + 1)
        bin_z = np.linspace(domain_min_z, domain_max_z, nz + 1)
        # bin mesh spacing
        dx = (domain_max_x - domain_min_x) / nx
        dy = (domain_max_y - domain_min_y) / ny
        dz = (domain_max_z - domain_min_z) / nz
        search_radious = np.sqrt(dx**2 + dy**2 + dz**2) / 2
        # Replace the bin mesh vertices with the centroids
        bin_x = bin_x[:-1] + dx / 2
        bin_y = bin_y[:-1] + dy / 2
        bin_z = bin_z[:-1] + dz / 2
        # create 3d bin 'mesh'. Using the centroids
        xx, yy, zz = np.meshgrid(bin_x, bin_y, bin_z, indexing="ij")
        bin_mesh_centroids = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

        # Create the global tree to make searches in the bin mesh
        self.log.write("info", "Creating global KD tree with bin mesh centroids")
        self.global_tree = KDTree(bin_mesh_centroids)

        # For this rank, determine which points are in which
        # bin mesh cell
        self.log.write("info", "Create map from sem mesh to bin mesh")
        mesh_to_bin = self.global_tree.query_ball_point(
            x=np.array([self.x.flatten(), self.y.flatten(), self.z.flatten()]).T,
            r=(search_radious) * (1 + 1e-6),
            p=2.0,
            eps=1e-8,
            workers=1,
            return_sorted=False,
            return_length=False,
        )

        self.log.write("info", "Create map from bin mesh to rank")
        self.bin_to_rank_map = domain_binning_map_bin_to_rank(
            mesh_to_bin, nx, ny, nz, comm
        )
        self.search_radious = search_radious

        self.log.toc()

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
            self.probes_rst = np.zeros((number_of_points, 3), dtype=np.double)
            self.el_owner = np.zeros((number_of_points), dtype=np.int64)
            self.glb_el_owner = np.zeros((number_of_points), dtype=np.int64)
            # self.rank_owner = np.zeros((number_of_points), dtype = np.int64)
            self.rank_owner = np.ones((number_of_points), dtype=np.int64) * -1000
            self.err_code = np.zeros(
                (number_of_points), dtype=np.int64
            )  # 0 not found, 1 is found
            self.test_pattern = np.ones(
                (number_of_points), dtype=np.double
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
            self.probes, probe_coord_partition_sendcount, io_rank, np.double
        )
        self.probe_partition = tmp.reshape((int(tmp.size / 3), 3))
        tmp = self.rt.scatter_from_root(
            self.probes_rst, probe_coord_partition_sendcount, io_rank, np.double
        )
        self.probe_rst_partition = tmp.reshape((int(tmp.size / 3), 3))
        ## Int
        self.el_owner_partition = self.rt.scatter_from_root(
            self.el_owner, probe_partition_sendcount, io_rank, np.int64
        )
        self.glb_el_owner_partition = self.rt.scatter_from_root(
            self.glb_el_owner, probe_partition_sendcount, io_rank, np.int64
        )
        self.rank_owner_partition = self.rt.scatter_from_root(
            self.rank_owner, probe_partition_sendcount, io_rank, np.int64
        )
        self.err_code_partition = self.rt.scatter_from_root(
            self.err_code, probe_partition_sendcount, io_rank, np.int64
        )
        ## Double
        self.test_pattern_partition = self.rt.scatter_from_root(
            self.test_pattern, probe_partition_sendcount, io_rank, np.double
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
        self.probes_rst = np.zeros((number_of_points, 3), dtype=np.double)
        self.el_owner = np.zeros((number_of_points), dtype=np.int64)
        self.glb_el_owner = np.zeros((number_of_points), dtype=np.int64)
        # self.rank_owner = np.zeros((number_of_points), dtype = np.int64)
        self.rank_owner = np.ones((number_of_points), dtype=np.int64) * -1000
        self.err_code = np.zeros(
            (number_of_points), dtype=np.int64
        )  # 0 not found, 1 is found
        self.test_pattern = np.ones(
            (number_of_points), dtype=np.double
        )  # test interpolation holder

        self.probe_partition = self.probes
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
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.double)
        if not isinstance(recvbuf, NoneType):
            self.probes[:, :] = recvbuf.reshape((int(recvbuf.size / 3), 3))[:, :]
        sendbuf = self.probe_rst_partition.reshape((self.probe_rst_partition.size))
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.double)
        if not isinstance(recvbuf, NoneType):
            self.probes_rst[:, :] = recvbuf.reshape((int(recvbuf.size / 3), 3))[:, :]
        sendbuf = self.el_owner_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.int64)
        if not isinstance(recvbuf, NoneType):
            self.el_owner[:] = recvbuf[:]
        sendbuf = self.glb_el_owner_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.int64)
        if not isinstance(recvbuf, NoneType):
            self.glb_el_owner[:] = recvbuf[:]
        sendbuf = self.rank_owner_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.int64)
        if not isinstance(recvbuf, NoneType):
            self.rank_owner[:] = recvbuf[:]
        sendbuf = self.err_code_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.int64)
        if not isinstance(recvbuf, NoneType):
            self.err_code[:] = recvbuf[:]
        sendbuf = self.test_pattern_partition
        recvbuf, _ = self.rt.gather_in_root(sendbuf, root, np.double)
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
        use_kdtree=True,
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(np.double).eps * 10,
        max_iter=50,
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
                use_kdtree=use_kdtree,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
                tol=tol,
                max_iter=max_iter,
            )
        elif ((find_points_comm_pattern == "point_to_point") or (find_points_comm_pattern == "collective")) and not find_points_iterative[0]:
            self.find_points_(
                comm,
                use_kdtree=use_kdtree,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
                tol=tol,
                max_iter=max_iter,
                comm_pattern = find_points_comm_pattern
            )
        elif ((find_points_comm_pattern == "point_to_point") or (find_points_comm_pattern == "collective")) and find_points_iterative[0]:
            self.find_points_iterative(
                comm,
                use_kdtree=use_kdtree,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
                tol=tol,
                batch_size=find_points_iterative[1],
                max_iter=max_iter,
            )

    def find_points_broadcast(
        self,
        comm,
        use_kdtree=True,
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(np.double).eps * 10,
        max_iter=50,
    ):
        """Find points using the collective implementation"""
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.rank = rank

        # First each rank finds their bounding box
        self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        if use_kdtree:
            # Get bbox centroids and max radius from center to corner
            self.my_bbox_centroids, self.my_bbox_maxdist = (
                get_bbox_centroids_and_max_dist(self.my_bbox)
            )

            # Build a KDtree with my information
            self.my_tree = KDTree(self.my_bbox_centroids)

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
                broadcaster_global_rank = np.ones((1), dtype=np.int64) * rank
                search_comm.Bcast(broadcaster_global_rank, root=broadcaster)

                # Tell every rank how much they need to allocate for the broadcaster bounding boxes
                nelv_in_broadcaster = np.ones((1), dtype=np.int64) * nelv
                search_comm.Bcast(nelv_in_broadcaster, root=broadcaster)

                # Allocate the recieve buffer for bounding boxes
                bbox_rec_buff = np.empty((nelv_in_broadcaster[0], 6), dtype=np.double)

                # Only in the broadcaster copy the data
                if search_rank == broadcaster:
                    bbox_rec_buff[:, :] = np.copy(self.my_bbox[:, :])

                # Broadcast the bounding boxes so each rank can check if the points are there
                search_comm.Bcast(bbox_rec_buff, root=broadcaster)

                # Only do the search, if my rank has not already searched the broadcaster
                if broadcaster_global_rank not in self.ranks_ive_checked:

                    if not use_kdtree:
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
                    elif use_kdtree:

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
                        np.double,
                    )
                )
                if not isinstance(tmp, NoneType):
                    probe_broadcaster_has = tmp.reshape((int(tmp.size / 3), 3))

                tmp, _ = search_rt.gather_in_root(
                    probe_rst_broadcaster_is_candidate.reshape(
                        (probe_rst_broadcaster_is_candidate.size)
                    ),
                    root,
                    np.double,
                )
                if not isinstance(tmp, NoneType):
                    probe_rst_broadcaster_has = tmp.reshape((int(tmp.size / 3), 3))
                (
                    el_owner_broadcaster_has,
                    el_owner_sendcount_broadcaster_is_candidate,
                ) = search_rt.gather_in_root(
                    el_owner_broadcaster_is_candidate, root, np.int64
                )
                glb_el_owner_broadcaster_has, _ = search_rt.gather_in_root(
                    glb_el_owner_broadcaster_is_candidate, root, np.int64
                )
                rank_owner_broadcaster_has, _ = search_rt.gather_in_root(
                    rank_owner_broadcaster_is_candidate, root, np.int64
                )
                err_code_broadcaster_has, _ = search_rt.gather_in_root(
                    err_code_broadcaster_is_candidate, root, np.int64
                )
                test_pattern_broadcaster_has, _ = search_rt.gather_in_root(
                    test_pattern_broadcaster_is_candidate, root, np.double
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
                        mesh_info["bbox_max_dist"] = self.my_bbox_maxdist

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
                    np.double,
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
                    np.double,
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
                    np.int64,
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
                    np.int64,
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
                    np.int64,
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
                    np.int64,
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
                    np.double,
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
        use_kdtree=True,
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(np.double).eps * 10,
        max_iter=50,
        comm_pattern = "point_to_point"
    ):
        """Find points using the point to point implementation"""
        rank = comm.Get_rank()
        self.rank = rank

        self.log.write("info", "Finding points - start")
        self.log.tic()
        start_time = MPI.Wtime()

        # First each rank finds their bounding box
        self.log.write("info", "Finding bounding box of sem mesh")
        self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        if use_kdtree:
            # Get bbox centroids and max radius from center to corner
            self.my_bbox_centroids, self.my_bbox_maxdist = (
                get_bbox_centroids_and_max_dist(self.my_bbox)
            )

            # Build a KDtree with my information
            self.log.write("info", "Creating KD tree with local bbox centroids")
            self.my_tree = KDTree(self.my_bbox_centroids)

        # nelv = self.x.shape[0]
        self.ranks_ive_checked = []

        # Get candidate ranks from a global kd tree
        # These are the destination ranks
        self.log.write("info", "Obtaining candidate ranks and sources")
        my_dest = get_candidate_ranks(self, comm)

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

        my_source, buff_probes = self.rt.transfer_data( comm_pattern,
            destination=my_dest, data=probe_not_found, dtype=np.double, tag=1
        )
        _, buff_probes_rst = self.rt.transfer_data( comm_pattern,
            destination=my_dest, data=probe_rst_not_found, dtype=np.double, tag=2
        )
        _, buff_el_owner = self.rt.transfer_data( comm_pattern,
            destination=my_dest, data=el_owner_not_found, dtype=np.int64, tag=3
        )
        _, buff_glb_el_owner = self.rt.transfer_data( comm_pattern,
            destination=my_dest, data=glb_el_owner_not_found, dtype=np.int64, tag=4
        )
        _, buff_rank_owner = self.rt.transfer_data( comm_pattern,
            destination=my_dest, data=rank_owner_not_found, dtype=np.int64, tag=5
        )
        _, buff_err_code = self.rt.transfer_data( comm_pattern,
            destination=my_dest, data=err_code_not_found, dtype=np.int64, tag=6
        )
        _, buff_test_pattern = self.rt.transfer_data( comm_pattern,
            destination=my_dest, data=test_pattern_not_found, dtype=np.double, tag=7
        )

        # Reshape the data from the probes
        for source_index in range(0, len(my_source)):
            buff_probes[source_index] = buff_probes[source_index].reshape(-1, 3)
            buff_probes_rst[source_index] = buff_probes_rst[source_index].reshape(-1, 3)

        # Set the information for the coordinate search in this rank
        self.log.write("info", "Find rst coordinates for the points")
        mesh_info = {}
        mesh_info["x"] = self.x
        mesh_info["y"] = self.y
        mesh_info["z"] = self.z
        mesh_info["bbox"] = self.my_bbox
        if hasattr(self, "my_tree"):
            mesh_info["kd_tree"] = self.my_tree
            mesh_info["bbox_max_dist"] = self.my_bbox_maxdist

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

        _, obuff_probes = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=buff_probes, dtype=np.double, tag=11
        )
        _, obuff_probes_rst = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=buff_probes_rst, dtype=np.double, tag=12
        )
        _, obuff_el_owner = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=buff_el_owner, dtype=np.int64, tag=13
        )
        _, obuff_glb_el_owner = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=buff_glb_el_owner, dtype=np.int64, tag=14
        )
        _, obuff_rank_owner = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=buff_rank_owner, dtype=np.int64, tag=15
        )
        _, obuff_err_code = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=buff_err_code, dtype=np.int64, tag=16
        )
        _, obuff_test_pattern = self.rt.transfer_data( comm_pattern,
            destination=my_source, data=buff_test_pattern, dtype=np.double, tag=17
        )

        # Reshape the data from the probes
        for dest_index in range(0, len(my_dest)):
            obuff_probes[dest_index] = obuff_probes[dest_index].reshape(-1, 3)
            obuff_probes_rst[dest_index] = obuff_probes_rst[dest_index].reshape(-1, 3)

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
            min_test_pattern = np.where(
                np.array(all_test_patterns) == np.array(all_test_patterns).min()
            )[0]
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
        use_kdtree=True,
        test_tol=1e-4,
        elem_percent_expansion=0.01,
        tol=np.finfo(np.double).eps * 10,
        max_iter=50,
        batch_size=5000,
        comm_pattern = "point_to_point"
    ):
        """Find points using the point to point implementation"""
        rank = comm.Get_rank()
        self.rank = rank

        self.log.write("info", "Finding points - start")
        self.log.tic()
        start_time = MPI.Wtime()

        # First each rank finds their bounding box
        self.log.write("info", "Finding bounding box of sem mesh")
        self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        if use_kdtree:
            # Get bbox centroids and max radius from center to corner
            self.my_bbox_centroids, self.my_bbox_maxdist = (
                get_bbox_centroids_and_max_dist(self.my_bbox)
            )

            # Build a KDtree with my information
            self.log.write("info", "Creating KD tree with local bbox centroids")
            self.my_tree = KDTree(self.my_bbox_centroids)

        # nelv = self.x.shape[0]
        self.ranks_ive_checked = []

        # Get candidate ranks from a global kd tree
        # These are the destination ranks
        self.log.write("info", "Obtaining candidate ranks and sources")
        my_dest = get_candidate_ranks(self, comm)

        self.log.write("info", "Determining maximun number of candidates")
        max_candidates = np.ones((1), dtype=np.int64) * len(my_dest)
        max_candidates = comm.allreduce(max_candidates, op=MPI.MAX)
        if batch_size > max_candidates[0]: batch_size = max_candidates[0]
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

            my_source, buff_probes = self.rt.transfer_data( comm_pattern,
                destination=my_it_dest, data=probe_not_found, dtype=np.double, tag=1
            )
            _, buff_probes_rst = self.rt.transfer_data( comm_pattern,
                destination=my_it_dest, data=probe_rst_not_found, dtype=np.double, tag=2
            )
            _, buff_el_owner = self.rt.transfer_data( comm_pattern,
                destination=my_it_dest, data=el_owner_not_found, dtype=np.int64, tag=3
            )
            _, buff_glb_el_owner = self.rt.transfer_data( comm_pattern,
                destination=my_it_dest, data=glb_el_owner_not_found, dtype=np.int64, tag=4
            )
            _, buff_rank_owner = self.rt.transfer_data( comm_pattern,
                destination=my_it_dest, data=rank_owner_not_found, dtype=np.int64, tag=5
            )
            _, buff_err_code = self.rt.transfer_data( comm_pattern,
                destination=my_it_dest, data=err_code_not_found, dtype=np.int64, tag=6
            )
            _, buff_test_pattern = self.rt.transfer_data( comm_pattern,
                destination=my_it_dest, data=test_pattern_not_found, dtype=np.double, tag=7
            )

            # Reshape the data from the probes
            for source_index in range(0, len(my_source)):
                buff_probes[source_index] = buff_probes[source_index].reshape(-1, 3)
                buff_probes_rst[source_index] = buff_probes_rst[source_index].reshape(-1, 3)

            # Set the information for the coordinate search in this rank
            self.log.write("info", "Find rst coordinates for the points")
            mesh_info = {}
            mesh_info["x"] = self.x
            mesh_info["y"] = self.y
            mesh_info["z"] = self.z
            mesh_info["bbox"] = self.my_bbox
            if hasattr(self, "my_tree"):
                mesh_info["kd_tree"] = self.my_tree
                mesh_info["bbox_max_dist"] = self.my_bbox_maxdist

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

            _, obuff_probes = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=buff_probes, dtype=np.double, tag=11
            )
            _, obuff_probes_rst = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=buff_probes_rst, dtype=np.double, tag=12
            )
            _, obuff_el_owner = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=buff_el_owner, dtype=np.int64, tag=13
            )
            _, obuff_glb_el_owner = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=buff_glb_el_owner, dtype=np.int64, tag=14
            )
            _, obuff_rank_owner = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=buff_rank_owner, dtype=np.int64, tag=15
            )
            _, obuff_err_code = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=buff_err_code, dtype=np.int64, tag=16
            )
            _, obuff_test_pattern = self.rt.transfer_data( comm_pattern,
                destination=my_source, data=buff_test_pattern, dtype=np.double, tag=17
            )

            # If no point was sent from this rank, then all buffers will be empty
            # so skip the rest of the loop
            if n_not_found < 1:
                continue

            # Reshape the data from the probes
            for dest_index in range(0, len(my_it_dest)):
                obuff_probes[dest_index] = obuff_probes[dest_index].reshape(-1, 3)
                obuff_probes_rst[dest_index] = obuff_probes_rst[dest_index].reshape(-1, 3)

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
                min_test_pattern = np.where(
                    np.array(all_test_patterns) == np.array(all_test_patterns).min()
                )[0]
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
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts * 3, root, np.double)
        my_probes = recvbuf.reshape((int(recvbuf.size / 3), 3))

        # Redistribute probes rst
        self.log.write("debug", "Scattering probes - probes rst")
        if rank == root:
            sendbuf = sorted_probes_rst.reshape((sorted_probes_rst.size))
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts * 3, root, np.double)
        my_probes_rst = recvbuf.reshape((int(recvbuf.size / 3), 3))

        # Redistribute err_code
        self.log.write("debug", "Scattering probes - err code")
        if rank == root:
            sendbuf = sorted_err_code.reshape((sorted_err_code.size))
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts, root, np.int64)
        my_err_code = recvbuf

        # Redistribute el_owner
        self.log.write("debug", "Scattering probes - el owner")
        if rank == root:
            sendbuf = sorted_el_owner.reshape((sorted_el_owner.size))
            # print(sendbuf)
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts, root, np.int64)
        # print(recvbuf)
        my_el_owner = recvbuf

        # Redistribute el_owner
        self.log.write("debug", "Scattering probes - rank owner")
        if rank == root:
            sendbuf = sorted_rank_owner.reshape((sorted_rank_owner.size))
        else:
            sendbuf = None
        recvbuf = self.rt.scatter_from_root(sendbuf, sendcounts, root, np.int64)
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
        sources, source_probes = self.rt.all_to_all(
            destination=destinations, data=probe_data, dtype=probe_data[0].dtype
        )
        _, source_probes_rst = self.rt.all_to_all(
            destination=destinations, data=probe_rst_data, dtype=probe_rst_data[0].dtype
        )
        _, source_el_owner = self.rt.all_to_all(
            destination=destinations, data=el_owner_data, dtype=el_owner_data[0].dtype
        )
        _, source_rank_owner = self.rt.all_to_all(
            destination=destinations,
            data=rank_owner_data,
            dtype=rank_owner_data[0].dtype,
        )
        _, source_err_code = self.rt.all_to_all(
            destination=destinations, data=err_code_data, dtype=err_code_data[0].dtype
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

        if isinstance(self.my_probes, list):
            # The inputs were distributed
            # So we return a list with the sample fields for the points of each rank that sent data to this one

            sampled_field_at_probe = []

            for i in range(0, len(self.my_probes)):
                probes_info = {}
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


def get_bbox_from_coordinates(x, y, z):
    """Determine if point is inside bounding box"""

    nelv = x.shape[0]
    # lx = x.shape[3]  # This is not a mistake. This is how the data is read
    # ly = x.shape[2]
    # lz = x.shape[1]

    bbox = np.zeros((nelv, 6), dtype=np.double)

    for e in range(0, nelv):
        bbox[e, 0] = np.min(x[e, :, :, :])
        bbox[e, 1] = np.max(x[e, :, :, :])
        bbox[e, 2] = np.min(y[e, :, :, :])
        bbox[e, 3] = np.max(y[e, :, :, :])
        bbox[e, 4] = np.min(z[e, :, :, :])
        bbox[e, 5] = np.max(z[e, :, :, :])

    return bbox


def get_bbox_centroids_and_max_dist(bbox):
    """
    Get centroids from the bounding boxes.
    """

    # Then find the centroids of each bbox and the maximun bbox radious from centroid to corner
    bbox_dist = np.zeros((bbox.shape[0], 3))
    bbox_dist[:, 0] = bbox[:, 1] - bbox[:, 0]
    bbox_dist[:, 1] = bbox[:, 3] - bbox[:, 2]
    bbox_dist[:, 2] = bbox[:, 5] - bbox[:, 4]

    bbox_max_dist = np.max(
        np.sqrt(bbox_dist[:, 0] ** 2 + bbox_dist[:, 1] ** 2 + bbox_dist[:, 2] ** 2) / 2
    )

    bbox_centroid = np.zeros((bbox.shape[0], 3))
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

            # Check if the points are really in the bounding box
            candidate_ranks_per_point_bool_ = np.array(
                [[pt_in_bbox(point, self.global_bbox[candidate], rel_tol = 0.01) for candidate in candidate_ranks_per_point_[i]]
                for i, point in enumerate(self.probe_partition[start:end])], dtype=object)
            
            # True candidates
            ## Dont make it an array (Keep it as a list), to do it later for candidate ranks per point
            true_candidate_ranks = [[candidate for candidate, bool_candidate in zip(candidate_ranks_per_point_[i], candidate_ranks_per_point_bool_[i]) if bool_candidate]
                for i, point in enumerate(self.probe_partition[start:end])]
             
            candidate_ranks_per_point.extend(true_candidate_ranks)

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

        # Search in which global coarse mesh cell each probe in
        # this rank resides
        probe_to_bin_map = self.global_tree.query_ball_point(
            x=self.probe_partition,
            r=self.search_radious * (1 + 1e-6),
            p=2.0,
            eps=1e-8,
            workers=1,
            return_sorted=False,
            return_length=False,
        )

        # Now map from bins to ranks
        candidate_ranks_per_point = domain_binning_map_probe_to_rank(
            self, probe_to_bin_map
        )

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

    return candidate_ranks


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