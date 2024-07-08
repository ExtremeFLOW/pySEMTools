import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from tqdm import tqdm
from .mpi_ops import *
from .sem import element_interpolator_c
from mpi4py import MPI  # for the timer
from itertools import combinations

NoneType = type(None)

# Variables for debugging
debug = False
tr = 7


class interpolator_c:
    def __init__(
        self,
        x,
        y,
        z,
        probes,
        comm,
        progress_bar=False,
        modal_search=True,
        use_tensor=False,
        use_torch=False,
        max_pts=128,
        max_elems=1,
    ):

        self.x = x
        self.y = y
        self.z = z
        self.probes = probes

        self.use_tensor = use_tensor
        self.use_torch = use_torch
        self.max_pts = max_pts
        self.max_elems = max_elems

        if use_torch == True:
            import torch

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if use_tensor == True or use_torch == True:
            from .tensor_sem import element_interpolator_c as ei_c

            self.ei = ei_c(
                x.shape[1],
                modal_search=modal_search,
                use_torch=use_torch,
                max_pts=max_pts,
                max_elems=max_elems,
            )

            if not use_torch:
                self.r = np.zeros((max_pts, max_elems, 1, 1), dtype=np.double)
                self.s = np.zeros((max_pts, max_elems, 1, 1), dtype=np.double)
                self.t = np.zeros((max_pts, max_elems, 1, 1), dtype=np.double)
                self.test_interp = np.zeros((max_pts, max_elems, 1, 1), dtype=np.double)
            else:
                self.r = torch.zeros(
                    (max_pts, max_elems, 1, 1), dtype=torch.float64, device=device
                )
                self.s = torch.zeros(
                    (max_pts, max_elems, 1, 1), dtype=torch.float64, device=device
                )
                self.t = torch.zeros(
                    (max_pts, max_elems, 1, 1), dtype=torch.float64, device=device
                )
                self.test_interp = torch.zeros(
                    (max_pts, max_elems, 1, 1), dtype=torch.float64, device=device
                )

            if comm.Get_rank() == 0:
                print("Using device: {}".format(self.r.device))

        else:
            self.ei = element_interpolator_c(x.shape[1], modal_search=modal_search)

        self.progress_bar = progress_bar

        # Find the element offset of each rank so you can store the global element number
        nelv = self.x.shape[0]
        sendbuf = np.ones((1), np.intc) * nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - nelv

    def scatter_probes_from_io_rank(self, io_rank, comm):

        rank = comm.Get_rank()
        size = comm.Get_size()

        # Check how many probes should be in each rank with a load balanced linear distribution
        probe_partition_sendcount = np.zeros((size), dtype=np.intc)
        if rank == io_rank:
            for i_rank in range(0, size):
                M = self.probes.shape[0]
                pe_rank = i_rank
                pe_size = comm.Get_size()
                L = np.floor(np.double(M) / np.double(pe_size))
                R = np.mod(M, pe_size)
                Ip = np.floor(
                    (
                        np.double(M)
                        + np.double(pe_size)
                        - np.double(pe_rank)
                        - np.double(1)
                    )
                    / np.double(pe_size)
                )
                nelv = int(Ip)
                offset_el = int(pe_rank * L + min(pe_rank, R))
                n = 3 * nelv
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
            self.el_owner = np.zeros((number_of_points), dtype=np.intc)
            self.glb_el_owner = np.zeros((number_of_points), dtype=np.intc)
            # self.rank_owner = np.zeros((number_of_points), dtype = np.intc)
            self.rank_owner = np.ones((number_of_points), dtype=np.intc) * -1000
            self.err_code = np.zeros(
                (number_of_points), dtype=np.intc
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

        # Now scatter the probes and all the codes according to what is needed and each rank has a partition
        self.probe_partition_sendcount = probe_partition_sendcount
        self.probe_coord_partition_sendcount = probe_coord_partition_sendcount
        ## Double
        tmp = scatter_from_root(
            self.probes, probe_coord_partition_sendcount, io_rank, np.double, comm
        )
        self.probe_partition = tmp.reshape((int(tmp.size / 3), 3))
        tmp = scatter_from_root(
            self.probes_rst, probe_coord_partition_sendcount, io_rank, np.double, comm
        )
        self.probe_rst_partition = tmp.reshape((int(tmp.size / 3), 3))
        ## Int
        self.el_owner_partition = scatter_from_root(
            self.el_owner, probe_partition_sendcount, io_rank, np.intc, comm
        )
        self.glb_el_owner_partition = scatter_from_root(
            self.glb_el_owner, probe_partition_sendcount, io_rank, np.intc, comm
        )
        self.rank_owner_partition = scatter_from_root(
            self.rank_owner, probe_partition_sendcount, io_rank, np.intc, comm
        )
        self.err_code_partition = scatter_from_root(
            self.err_code, probe_partition_sendcount, io_rank, np.intc, comm
        )
        ## Double
        self.test_pattern_partition = scatter_from_root(
            self.test_pattern, probe_partition_sendcount, io_rank, np.double, comm
        )

        return

    def gather_probes_to_io_rank(self, io_rank, comm):

        rank = comm.Get_rank()
        size = comm.Get_size()

        root = io_rank
        sendbuf = self.probe_partition.reshape((self.probe_partition.size))
        recvbuf, _ = gather_in_root(sendbuf, root, np.double, comm)
        if type(recvbuf) != NoneType:
            self.probes[:, :] = recvbuf.reshape((int(recvbuf.size / 3), 3))[:, :]
        sendbuf = self.probe_rst_partition.reshape((self.probe_rst_partition.size))
        recvbuf, _ = gather_in_root(sendbuf, root, np.double, comm)
        if type(recvbuf) != NoneType:
            self.probes_rst[:, :] = recvbuf.reshape((int(recvbuf.size / 3), 3))[:, :]
        sendbuf = self.el_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc, comm)
        if type(recvbuf) != NoneType:
            self.el_owner[:] = recvbuf[:]
        sendbuf = self.glb_el_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc, comm)
        if type(recvbuf) != NoneType:
            self.glb_el_owner[:] = recvbuf[:]
        sendbuf = self.rank_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc, comm)
        if type(recvbuf) != NoneType:
            self.rank_owner[:] = recvbuf[:]
        sendbuf = self.err_code_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc, comm)
        if type(recvbuf) != NoneType:
            self.err_code[:] = recvbuf[:]
        sendbuf = self.test_pattern_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.double, comm)
        if type(recvbuf) != NoneType:
            self.test_pattern[:] = recvbuf[:]

        return

    def find_points(
        self, comm, use_kdtree=True, test_tol=1e-4, elem_percent_expansion=0.01
    ):

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

            # With this logic, first every rank checks by themselves and then 2 check together, then 4 ... up to when all of them check
            col = int(np.floor(((rank / denom))))
            if debug:
                print(
                    "rank: {}, finding points. start iteration: {}. Color: {}".format(
                        rank, j, col
                    )
                )
            else:
                if rank == 0:
                    print(
                        "rank: {}, finding points. start iteration: {}. Color: {}".format(
                            rank, j, col
                        )
                    )
            start_time = MPI.Wtime()
            denom = denom * 2

            search_comm = comm.Split(color=col, key=rank)
            search_rank = search_comm.Get_rank()
            search_size = search_comm.Get_size()

            # Make each rank in the communicator broadcast their bounding boxes to the others to search
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
                broadcaster_global_rank = np.ones((1), dtype=np.intc) * rank
                search_comm.Bcast(broadcaster_global_rank, root=broadcaster)

                # Tell every rank how much they need to allocate for the broadcaster bounding boxes
                nelv_in_broadcaster = np.ones((1), dtype=np.intc) * nelv
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

                    # ================================================================================
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

                            if found_candidate == True:
                                i = i + 1
                                if self.progress_bar:
                                    pbar.update(1)
                                continue
                            i = i + 1
                            if self.progress_bar:
                                pbar.update(1)
                        if self.progress_bar:
                            pbar.close()
                    # ================================================================================
                    elif use_kdtree:

                        # Get bbox centroids and max radius from center to corner for the broadcaster
                        bbox_centroids, bbox_maxdist = get_bbox_centroids_and_max_dist(
                            bbox_rec_buff
                        )

                        # Create the KDtree
                        broadcaster_tree = KDTree(bbox_centroids)
                        # Query the tree with the probes to reduce the bbox search
                        candidate_elements = broadcaster_tree.query_ball_point(
                            x=probe_not_found,
                            r=bbox_maxdist,
                            p=2.0,
                            eps=elem_percent_expansion,
                            workers=1,
                            return_sorted=False,
                            return_length=False,
                        )

                        # Do a bbox search over the candidate elements, just as it used to be done (The KD tree allows to avoid searching ALL elements)
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

                            if found_candidate == True:
                                i = i + 1
                                if self.progress_bar:
                                    pbar.update(1)
                                continue
                            i = i + 1
                            if self.progress_bar:
                                pbar.update(1)
                        if self.progress_bar:
                            pbar.close()
                    # ================================================================================

                    # Now let the brodcaster gather the points that the other ranks think it has
                    # broadcaster_is_candidate = np.where(rank_owner_not_found == broadcaster_global_rank)[0]
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
                tmp, probe_sendcount_broadcaster_is_candidate = gather_in_root(
                    probe_broadcaster_is_candidate.reshape(
                        (probe_broadcaster_is_candidate.size)
                    ),
                    root,
                    np.double,
                    search_comm,
                )
                if type(tmp) != NoneType:
                    probe_broadcaster_has = tmp.reshape((int(tmp.size / 3), 3))

                # For debugging
                if broadcaster_global_rank == tr and rank == tr and debug == True:
                    print(probe_sendcount_broadcaster_is_candidate / 3)
                    # print(probe_broadcaster_has)

                tmp, _ = gather_in_root(
                    probe_rst_broadcaster_is_candidate.reshape(
                        (probe_rst_broadcaster_is_candidate.size)
                    ),
                    root,
                    np.double,
                    search_comm,
                )
                if type(tmp) != NoneType:
                    probe_rst_broadcaster_has = tmp.reshape((int(tmp.size / 3), 3))
                (
                    el_owner_broadcaster_has,
                    el_owner_sendcount_broadcaster_is_candidate,
                ) = gather_in_root(
                    el_owner_broadcaster_is_candidate, root, np.intc, search_comm
                )
                glb_el_owner_broadcaster_has, _ = gather_in_root(
                    glb_el_owner_broadcaster_is_candidate, root, np.intc, search_comm
                )
                rank_owner_broadcaster_has, _ = gather_in_root(
                    rank_owner_broadcaster_is_candidate, root, np.intc, search_comm
                )
                err_code_broadcaster_has, _ = gather_in_root(
                    err_code_broadcaster_is_candidate, root, np.intc, search_comm
                )
                test_pattern_broadcaster_has, _ = gather_in_root(
                    test_pattern_broadcaster_is_candidate, root, np.double, search_comm
                )

                # Now let the broadcaster check if it really had the point. It will go through all the elements again and check rst coordinates
                if search_rank == broadcaster:
                    (
                        probe_broadcaster_has,
                        probe_rst_broadcaster_has,
                        el_owner_broadcaster_has,
                        glb_el_owner_broadcaster_has,
                        rank_owner_broadcaster_has,
                        err_code_broadcaster_has,
                        test_pattern_broadcaster_has,
                    ) = self.find_rst(
                        probe_broadcaster_has,
                        probe_rst_broadcaster_has,
                        el_owner_broadcaster_has,
                        glb_el_owner_broadcaster_has,
                        rank_owner_broadcaster_has,
                        err_code_broadcaster_has,
                        test_pattern_broadcaster_has,
                        broadcaster_global_rank,
                        self.offset_el,
                        not_found_code=-10,
                        use_kdtree=use_kdtree,
                        elem_percent_expansion=elem_percent_expansion,
                    )

                # Now scatter the results back to all the other ranks
                root = broadcaster
                if search_rank == root:
                    sendbuf = probe_broadcaster_has.reshape(probe_broadcaster_has.size)
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    probe_sendcount_broadcaster_is_candidate,
                    root,
                    np.double,
                    search_comm,
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
                recvbuf = scatter_from_root(
                    sendbuf,
                    probe_sendcount_broadcaster_is_candidate,
                    root,
                    np.double,
                    search_comm,
                )
                probe_rst_broadcaster_is_candidate[:, :] = recvbuf.reshape(
                    (int(recvbuf.size / 3), 3)
                )[:, :]
                if search_rank == root:
                    sendbuf = el_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.intc,
                    search_comm,
                )
                el_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    # sendbuf = el_owner_broadcaster_has+self.offset_el
                    sendbuf = glb_el_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.intc,
                    search_comm,
                )
                glb_el_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = rank_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.intc,
                    search_comm,
                )
                rank_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = err_code_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.intc,
                    search_comm,
                )
                err_code_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = test_pattern_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.double,
                    search_comm,
                )
                test_pattern_broadcaster_is_candidate[:] = recvbuf[:]

                # Now that the data is back at the original ranks, put it in the place of the not found list that it should be
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

                # at the end of the broadcaster run, update the information from the previously not found data
                self.probe_partition[not_found, :] = probe_not_found[:, :]
                self.probe_rst_partition[not_found] = probe_rst_not_found[:, :]
                self.el_owner_partition[not_found] = el_owner_not_found[:]
                self.glb_el_owner_partition[not_found] = glb_el_owner_not_found[:]
                self.rank_owner_partition[not_found] = rank_owner_not_found[:]
                self.err_code_partition[not_found] = err_code_not_found[:]
                self.test_pattern_partition[not_found] = test_pattern_not_found[:]

            if debug:
                print(
                    "rank: {}, finding points. finished iteration: {}. time(s): {}".format(
                        rank, j, MPI.Wtime() - start_time
                    )
                )
            else:
                if rank == 0:
                    print(
                        "rank: {}, finding points. finished iteration: {}. time(s): {}".format(
                            rank, j, MPI.Wtime() - start_time
                        )
                    )

            j = j + 1
            search_comm.Free()

        # Final check
        for j in range(0, len(self.test_pattern_partition)):
            # After all iteration are done, check if some points were not found. Use the error code and the test pattern
            if (
                self.err_code_partition[j] != 1
                and self.test_pattern_partition[j] > test_tol
            ):
                self.err_code_partition[j] = 0

            # Check also if the rst are too big, then it needs to be outside
            # if ( abs(self.probe_rst_partition[j, 0]) +  abs(self.probe_rst_partition[j, 1]) +  abs(self.probe_rst_partition[j, 2]) ) > 3.5:
            #    self.err_code_partition[j] = 0

        return

    def find_points_comm_pairs(
        self,
        comm,
        use_kdtree=True,
        test_tol=1e-4,
        communicate_candidate_pairs=True,
        elem_percent_expansion=0.01,
    ):

        rank = comm.Get_rank()
        size = comm.Get_size()
        self.rank = rank

        if debug:
            print("rank: {}, finding points. start".format(rank))
        else:
            if rank == 0:
                print("rank: {}, finding points. start".format(rank))
        start_time = MPI.Wtime()

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

        # Get candidate ranks from a global kd tree
        # These are the destination ranks
        my_dest = get_candidate_ranks(self, comm)

        # Get a global array with the candidates in all other ranks to determine the best way to communicate
        global_rank_candidate = get_global_candidate_ranks(self, comm, my_dest)

        # Get the ranks that have me in their dest, so they become my sources (This tank will check their data)
        my_source = np.where(np.any(global_rank_candidate == rank, axis=1))[0]

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

        # Tell every rank how many points not found each other rank has
        not_found_in_this_rank = np.ones((1), dtype=np.intc) * n_not_found
        not_found_in_all_ranks = np.zeros((comm.Get_size()), dtype=np.intc)
        comm.Allgather(
            [not_found_in_this_rank, MPI.INT], [not_found_in_all_ranks, MPI.INT]
        )  # This allgather can be changed with point2point

        # Check how many buffers to create to recieve points from other ranks that think this rank is a candidate
        n_buff = len(my_source)
        # Create buffer for data from other ranks
        buff_probes = []
        buff_probes_rst = []
        buff_el_owner = []
        buff_glb_el_owner = []
        buff_rank_owner = []
        buff_err_code = []
        buff_test_pattern = []
        for ni in range(n_buff):
            # The points in each buffer depend on the points on the sending rank
            npt = not_found_in_all_ranks[my_source[ni]]
            buff_probes.append(np.zeros((npt, 3), dtype=np.double))
            buff_probes_rst.append(np.zeros((npt, 3), dtype=np.double))
            buff_el_owner.append(np.zeros((npt), dtype=np.intc))
            buff_glb_el_owner.append(np.zeros((npt), dtype=np.intc))
            buff_rank_owner.append(np.ones((npt), dtype=np.intc) * -1000)
            buff_err_code.append(
                np.zeros((npt), dtype=np.intc)
            )  # 0 not found, 1 is found
            buff_test_pattern.append(
                np.ones((npt), dtype=np.double)
            )  # test interpolation holder

        # This rank will send its data to the other ranks in the dest list.
        # These buffers will be used to store the data when the points are sent back
        on_buff = len(my_dest)
        obuff_probes = []
        obuff_probes_rst = []
        obuff_el_owner = []
        obuff_glb_el_owner = []
        obuff_rank_owner = []
        obuff_err_code = []
        obuff_test_pattern = []
        for ni in range(on_buff):
            # The points in each buffer are the same points in this rank
            npt = n_not_found
            obuff_probes.append(np.zeros((npt, 3), dtype=np.double))
            obuff_probes_rst.append(np.zeros((npt, 3), dtype=np.double))
            obuff_el_owner.append(np.zeros((npt), dtype=np.intc))
            obuff_glb_el_owner.append(np.zeros((npt), dtype=np.intc))
            obuff_rank_owner.append(np.ones((npt), dtype=np.intc) * -1000)
            obuff_err_code.append(
                np.zeros((npt), dtype=np.intc)
            )  # 0 not found, 1 is found
            obuff_test_pattern.append(
                np.ones((npt), dtype=np.double)
            )  # test interpolation holder

        # Set the request to Recieve the data from the other ranks that have me as a candidate
        req_list_r = []
        for source_index in range(0, len(my_source)):
            source = my_source[source_index]
            req_list_r.append([])
            req_list_r[source_index].append(
                comm.Irecv(buff_probes[source_index], source=source, tag=1)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_probes_rst[source_index], source=source, tag=2)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_el_owner[source_index], source=source, tag=3)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_glb_el_owner[source_index], source=source, tag=4)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_rank_owner[source_index], source=source, tag=5)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_err_code[source_index], source=source, tag=6)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_test_pattern[source_index], source=source, tag=7)
            )

        # Set and complete the request to send my points to my candidates
        req_list_s = []
        dest_index = -1
        for dest in my_dest:
            dest_index = dest_index + 1
            req_list_s.append([])
            req_list_s[dest_index].append(comm.Isend(probe_not_found, dest=dest, tag=1))
            req_list_s[dest_index].append(
                comm.Isend(probe_rst_not_found, dest=dest, tag=2)
            )
            req_list_s[dest_index].append(
                comm.Isend(el_owner_not_found, dest=dest, tag=3)
            )
            req_list_s[dest_index].append(
                comm.Isend(glb_el_owner_not_found, dest=dest, tag=4)
            )
            req_list_s[dest_index].append(
                comm.Isend(rank_owner_not_found, dest=dest, tag=5)
            )
            req_list_s[dest_index].append(
                comm.Isend(err_code_not_found, dest=dest, tag=6)
            )
            req_list_s[dest_index].append(
                comm.Isend(test_pattern_not_found, dest=dest, tag=7)
            )

            # complete the send request
            for req in req_list_s[dest_index]:
                req.wait()

        # Complete the request to recieve the data in the buffers
        for source_index in range(0, len(my_source)):
            for req in req_list_r[source_index]:
                req.wait()

        # Now find the rst coordinates for the points stored in each of the buffers
        for source_index in range(0, len(my_source)):
            [
                buff_probes[source_index],
                buff_probes_rst[source_index],
                buff_el_owner[source_index],
                buff_glb_el_owner[source_index],
                buff_rank_owner[source_index],
                buff_err_code[source_index],
                buff_test_pattern[source_index],
            ] = self.find_rst(
                buff_probes[source_index],
                buff_probes_rst[source_index],
                buff_el_owner[source_index],
                buff_glb_el_owner[source_index],
                buff_rank_owner[source_index],
                buff_err_code[source_index],
                buff_test_pattern[source_index],
                rank,
                self.offset_el,
                not_found_code=-10,
                use_kdtree=use_kdtree,
                elem_percent_expansion=elem_percent_expansion,
            )

        # Set the request to Recieve back the data that I have sent to my candidates
        oreq_list_r = []
        for source_index in range(0, len(my_dest)):
            source = my_dest[source_index]
            oreq_list_r.append([])
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_probes[source_index], source=source, tag=11)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_probes_rst[source_index], source=source, tag=12)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_el_owner[source_index], source=source, tag=13)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_glb_el_owner[source_index], source=source, tag=14)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_rank_owner[source_index], source=source, tag=15)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_err_code[source_index], source=source, tag=16)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_test_pattern[source_index], source=source, tag=17)
            )

        # Set the request to send this data back to the rank that sent it to me
        oreq_list_s = []
        for dest_index in range(0, len(my_source)):
            dest = my_source[dest_index]
            oreq_list_s.append([])
            oreq_list_s[dest_index].append(
                comm.Isend(buff_probes[dest_index], dest=dest, tag=11)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_probes_rst[dest_index], dest=dest, tag=12)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_el_owner[dest_index], dest=dest, tag=13)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_glb_el_owner[dest_index], dest=dest, tag=14)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_rank_owner[dest_index], dest=dest, tag=15)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_err_code[dest_index], dest=dest, tag=16)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_test_pattern[dest_index], dest=dest, tag=17)
            )

            # Complete the send request
            for req in oreq_list_s[dest_index]:
                req.wait()

        # Complete the request to recieve the data in the buffers
        for index in range(0, len(my_dest)):
            for req in oreq_list_r[index]:
                req.wait()

        # Now loop trhough all the points in the buffers that have been sent back and determine which point was found
        for point in range(0, n_not_found):

            # These are the error code and test patterns for this point from all the ranks that sent back
            all_err_codes = [arr[point] for arr in obuff_err_code]
            all_test_patterns = [arr[point] for arr in obuff_test_pattern]

            # Check if any rank had certainty that it had found the point
            found_err_code = np.where(np.array(all_err_codes) == 1)[0]

            # If the point was found in any rank, just choose the first one in the list (in case there was more than one founder):
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

            # If the point was not found with certainty, then choose as owner the the one that produced the smaller error in the test pattern
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

        # Go through the points again, if the test pattern was too large, mark that point as not found
        for j in range(0, len(self.test_pattern_partition)):
            # After all iteration are done, check if some points were not found. Use the error code and the test pattern
            if (
                self.err_code_partition[j] != 1
                and self.test_pattern_partition[j] > test_tol
            ):
                self.err_code_partition[j] = 0

        if debug:
            print(
                "rank: {}, finding points. finished. time(s): {}".format(
                    rank, MPI.Wtime() - start_time
                )
            )
        else:
            if rank == 0:
                print(
                    "rank: {}, finding points. finished. time(s): {}".format(
                        rank, MPI.Wtime() - start_time
                    )
                )

        return

    def find_rst(
        self,
        probes,
        probes_rst,
        el_owner,
        glb_el_owner,
        rank_owner,
        err_code,
        test_pattern,
        rank,
        offset_el,
        not_found_code=-10,
        use_kdtree=True,
        use_test_pattern=True,
        elem_percent_expansion=0.01,
    ):

        # Reset the element owner and the error code so this rank checks again
        # el_owner[:] = 0
        err_code[:] = not_found_code

        # Find the elements in this rank that could hold the point. Check all elements even if you had a guide as an input
        # ================================================================================
        if not use_kdtree:
            element_candidates = []
            i = 0
            if self.progress_bar:
                pbar = tqdm(total=probes.shape[0])
            for pt in probes:
                element_candidates.append([])
                for e in range(0, self.my_bbox.shape[0]):
                    if pt_in_bbox(pt, self.my_bbox[e], rel_tol=elem_percent_expansion):
                        element_candidates[i].append(e)
                i = i + 1
                if self.progress_bar:
                    pbar.update(1)
            if self.progress_bar:
                pbar.close()
        # ================================================================================
        elif use_kdtree:

            # Query the tree with the probes to reduce the bbox search
            candidate_elements = self.my_tree.query_ball_point(
                x=probes,
                r=self.my_bbox_maxdist,
                p=2.0,
                eps=elem_percent_expansion,
                workers=1,
                return_sorted=False,
                return_length=False,
            )

            element_candidates = []
            i = 0
            if self.progress_bar:
                pbar = tqdm(total=probes.shape[0])
            for pt in probes:
                element_candidates.append([])
                for e in candidate_elements[i]:
                    if pt_in_bbox(pt, self.my_bbox[e], rel_tol=elem_percent_expansion):
                        element_candidates[i].append(e)
                i = i + 1
                if self.progress_bar:
                    pbar.update(1)
            if self.progress_bar:
                pbar.close()
        # ================================================================================
        # if self.rank == tr and debug == True: print(element_candidates)

        if self.use_tensor == False and self.use_torch == False:

            if self.progress_bar:
                pbar = tqdm(total=probes.shape[0])
            for pts in range(0, probes.shape[0]):
                if err_code[pts] != 1:
                    for e in element_candidates[pts]:
                        self.ei.project_element_into_basis(
                            self.x[e, :, :, :], self.y[e, :, :, :], self.z[e, :, :, :]
                        )
                        r, s, t = self.ei.find_rst_from_xyz(
                            probes[pts, 0], probes[pts, 1], probes[pts, 2]
                        )
                        if self.ei.point_inside_element:
                            probes_rst[pts, 0] = r
                            probes_rst[pts, 1] = s
                            probes_rst[pts, 2] = t
                            el_owner[pts] = e
                            glb_el_owner[pts] = e + offset_el
                            rank_owner[pts] = rank
                            err_code[pts] = 1
                            break
                        else:

                            # Perform test interpolation and update if the results are better than previously stored
                            if use_test_pattern:
                                test_field = (
                                    self.x[e, :, :, :] ** 2
                                    + self.y[e, :, :, :] ** 2
                                    + self.z[e, :, :, :] ** 2
                                )
                                test_probe = (
                                    probes[pts, 0] ** 2
                                    + probes[pts, 1] ** 2
                                    + probes[pts, 2] ** 2
                                )
                                test_interp = self.ei.interpolate_field_at_rst(
                                    r, s, t, test_field
                                )

                                test_error = abs(test_probe - test_interp)

                                if test_error < test_pattern[pts]:
                                    probes_rst[pts, 0] = r
                                    probes_rst[pts, 1] = s
                                    probes_rst[pts, 2] = t
                                    el_owner[pts] = e
                                    glb_el_owner[pts] = e + offset_el
                                    rank_owner[pts] = rank
                                    err_code[pts] = not_found_code
                                    test_pattern[pts] = test_error

                            # Otherwise progressively update
                            else:
                                probes_rst[pts, 0] = r
                                probes_rst[pts, 1] = s
                                probes_rst[pts, 2] = t
                                el_owner[pts] = e
                                glb_el_owner[pts] = e + offset_el
                                rank_owner[pts] = rank
                                err_code[pts] = not_found_code

                if self.progress_bar:
                    pbar.update(1)
            if self.progress_bar:
                pbar.close()

        elif self.use_tensor == True or self.use_torch == True:

            # Identify variables
            max_pts = self.max_pts
            pts_n = probes.shape[0]
            max_candidate_elements = np.max(
                [len(elist) for elist in element_candidates]
            )
            iterations = np.ceil((pts_n / max_pts))
            checked_elements = [[] for i in range(0, pts_n)]

            start_time = MPI.Wtime()
            exit_flag = False
            # The following logic only works for nelems = 1
            npoints = 10000
            nelems = 1
            for e in range(0, max_candidate_elements):
                if exit_flag:
                    break
                for j in range(0, int(iterations)):
                    if npoints == 0:
                        exit_flag = True
                        break

                    # Get the index of points that have not been found
                    pt_not_found_indices = np.where(err_code != 1)[0]
                    # Get the indices of these points that still have elements remaining to check
                    pt_not_found_indices = pt_not_found_indices[
                        np.where(
                            [
                                len(checked_elements[i]) < len(element_candidates[i])
                                for i in pt_not_found_indices
                            ]
                        )[0]
                    ]
                    # Select only the maximum number of points
                    pt_not_found_indices = pt_not_found_indices[:max_pts]

                    # See which element should be checked in this iteration
                    temp_candidates = [
                        element_candidates[i] for i in pt_not_found_indices
                    ]
                    temp_checked = [checked_elements[i] for i in pt_not_found_indices]
                    temp_to_check_ = [
                        list(set(temp_candidates[i]) - set(temp_checked[i]))
                        for i in range(len(temp_candidates))
                    ]
                    # Sort them by order of closeness
                    temp_to_check = [
                        sorted(temp_to_check_[i], key=temp_candidates[i].index)
                        for i in range(len(temp_candidates))
                    ]

                    elem_to_check_per_point = [elist[0] for elist in temp_to_check]
                    # Update the checked elements
                    for i in range(0, len(pt_not_found_indices)):
                        checked_elements[pt_not_found_indices[i]].append(
                            elem_to_check_per_point[i]
                        )

                    npoints = len(pt_not_found_indices)

                    if npoints == 0:
                        exit_flag = True
                        break

                    probe_new_shape = (npoints, 1, 1, 1)
                    elem_new_shape = (
                        npoints,
                        nelems,
                        self.x.shape[1],
                        self.x.shape[2],
                        self.x.shape[3],
                    )

                    self.ei.project_element_into_basis(
                        self.x[elem_to_check_per_point].reshape(elem_new_shape),
                        self.y[elem_to_check_per_point].reshape(elem_new_shape),
                        self.z[elem_to_check_per_point].reshape(elem_new_shape),
                        use_torch=self.use_torch,
                    )
                    (
                        self.r[:npoints, :nelems],
                        self.s[:npoints, :nelems],
                        self.t[:npoints, :nelems],
                    ) = self.ei.find_rst_from_xyz(
                        probes[pt_not_found_indices, 0].reshape(probe_new_shape),
                        probes[pt_not_found_indices, 1].reshape(probe_new_shape),
                        probes[pt_not_found_indices, 2].reshape(probe_new_shape),
                        use_torch=self.use_torch,
                    )

                    # Reshape results
                    result_r = self.r[:npoints, :nelems, :, :].reshape(
                        (len(pt_not_found_indices))
                    )
                    result_s = self.s[:npoints, :nelems, :, :].reshape(
                        (len(pt_not_found_indices))
                    )
                    result_t = self.t[:npoints, :nelems, :, :].reshape(
                        (len(pt_not_found_indices))
                    )
                    result_code_bool = self.ei.point_inside_element[
                        :npoints, :nelems, :, :
                    ].reshape((len(pt_not_found_indices)))
                    # Assign the error codes
                    if not self.use_torch:

                        # Update indices of points that were found and those that were not
                        pt_found_this_it = np.where(result_code_bool)[0]
                        pt_not_found_this_it = np.where(~result_code_bool)[0]

                        # Create a list with the original indices for each of this
                        real_index_pt_found_this_it = [
                            pt_not_found_indices[pt_found_this_it[i]]
                            for i in range(0, len(pt_found_this_it))
                        ]
                        real_index_pt_not_found_this_it = [
                            pt_not_found_indices[pt_not_found_this_it[i]]
                            for i in range(0, len(pt_not_found_this_it))
                        ]

                        # Update codes for points found in this iteration
                        probes_rst[real_index_pt_found_this_it, 0] = result_r[
                            pt_found_this_it
                        ]
                        probes_rst[real_index_pt_found_this_it, 1] = result_s[
                            pt_found_this_it
                        ]
                        probes_rst[real_index_pt_found_this_it, 2] = result_t[
                            pt_found_this_it
                        ]
                        el_owner[real_index_pt_found_this_it] = np.array(
                            elem_to_check_per_point
                        )[pt_found_this_it]
                        glb_el_owner[real_index_pt_found_this_it] = (
                            el_owner[real_index_pt_found_this_it] + offset_el
                        )
                        rank_owner[real_index_pt_found_this_it] = rank
                        err_code[real_index_pt_found_this_it] = 1

                        # If user has selected to check a test pattern:
                        if use_test_pattern:

                            # Get shapes
                            ntest = len(pt_not_found_this_it)
                            test_probe_new_shape = (ntest, nelems, 1, 1)
                            test_elem_new_shape = (
                                ntest,
                                nelems,
                                self.x.shape[1],
                                self.x.shape[2],
                                self.x.shape[3],
                            )

                            # Define new arrays (On the cpu)
                            test_elems = np.array(elem_to_check_per_point)[
                                pt_not_found_this_it
                            ]
                            test_fields = (
                                self.x[test_elems, :, :, :] ** 2
                                + self.y[test_elems, :, :, :] ** 2
                                + self.z[test_elems, :, :, :] ** 2
                            )
                            test_probes = (
                                probes[real_index_pt_not_found_this_it, 0] ** 2
                                + probes[real_index_pt_not_found_this_it, 1] ** 2
                                + probes[real_index_pt_not_found_this_it, 2] ** 2
                            )

                            # Perform the test interpolation
                            self.test_interp[:ntest, :nelems] = (
                                self.ei.interpolate_field_at_rst(
                                    result_r[pt_not_found_this_it].reshape(
                                        test_probe_new_shape
                                    ),
                                    result_s[pt_not_found_this_it].reshape(
                                        test_probe_new_shape
                                    ),
                                    result_t[pt_not_found_this_it].reshape(
                                        test_probe_new_shape
                                    ),
                                    test_fields.reshape(test_elem_new_shape),
                                    use_torch=self.use_torch,
                                )
                            )
                            test_result = self.test_interp[:ntest, :nelems].reshape(
                                ntest
                            )

                            # Check if the test pattern is satisfied
                            test_error = abs(test_probes - test_result)

                            # Now assign
                            real_list = np.array(real_index_pt_not_found_this_it)
                            relative_list = np.array(pt_not_found_this_it)
                            better_test = np.where(
                                test_error
                                < test_pattern[real_index_pt_not_found_this_it]
                            )[0]

                            if len(better_test) > 0:
                                probes_rst[real_list[better_test], 0] = result_r[
                                    relative_list[better_test]
                                ]
                                probes_rst[real_list[better_test], 1] = result_s[
                                    relative_list[better_test]
                                ]
                                probes_rst[real_list[better_test], 2] = result_t[
                                    relative_list[better_test]
                                ]
                                el_owner[real_list[better_test]] = np.array(
                                    elem_to_check_per_point
                                )[relative_list[better_test]]
                                glb_el_owner[real_list[better_test]] = (
                                    el_owner[real_list[better_test]] + offset_el
                                )
                                rank_owner[real_list[better_test]] = rank
                                err_code[real_list[better_test]] = not_found_code
                                test_pattern[real_list[better_test]] = test_error[
                                    better_test
                                ]

                        else:

                            probes_rst[real_index_pt_not_found_this_it, 0] = result_r[
                                pt_not_found_this_it
                            ]
                            probes_rst[real_index_pt_not_found_this_it, 1] = result_s[
                                pt_not_found_this_it
                            ]
                            probes_rst[real_index_pt_not_found_this_it, 2] = result_t[
                                pt_not_found_this_it
                            ]
                            el_owner[real_index_pt_not_found_this_it] = np.array(
                                elem_to_check_per_point
                            )[pt_not_found_this_it]
                            glb_el_owner[real_index_pt_not_found_this_it] = (
                                el_owner[real_index_pt_not_found_this_it] + offset_el
                            )
                            rank_owner[real_index_pt_not_found_this_it] = rank
                            err_code[real_index_pt_not_found_this_it] = not_found_code

                    else:

                        result_code_bool = result_code_bool.cpu().numpy()
                        result_r = result_r.cpu().numpy()
                        result_s = result_s.cpu().numpy()
                        result_t = result_t.cpu().numpy()

                        pt_found_this_it = np.where(result_code_bool)[0]
                        pt_not_found_this_it = np.where(~result_code_bool)[0]

                        # Create a list with the original indices for each of this
                        real_index_pt_found_this_it = [
                            pt_not_found_indices[pt_found_this_it[i]]
                            for i in range(0, len(pt_found_this_it))
                        ]
                        real_index_pt_not_found_this_it = [
                            pt_not_found_indices[pt_not_found_this_it[i]]
                            for i in range(0, len(pt_not_found_this_it))
                        ]

                        # Update codes for points found in this iteration
                        probes_rst[real_index_pt_found_this_it, 0] = result_r[
                            pt_found_this_it
                        ]
                        probes_rst[real_index_pt_found_this_it, 1] = result_s[
                            pt_found_this_it
                        ]
                        probes_rst[real_index_pt_found_this_it, 2] = result_t[
                            pt_found_this_it
                        ]
                        el_owner[real_index_pt_found_this_it] = np.array(
                            elem_to_check_per_point
                        )[pt_found_this_it]
                        glb_el_owner[real_index_pt_found_this_it] = (
                            el_owner[real_index_pt_found_this_it] + offset_el
                        )
                        rank_owner[real_index_pt_found_this_it] = rank
                        err_code[real_index_pt_found_this_it] = 1

                        # If user has selected to check a test pattern:
                        if use_test_pattern:

                            # Get shapes
                            ntest = len(pt_not_found_this_it)
                            test_probe_new_shape = (ntest, nelems, 1, 1)
                            test_elem_new_shape = (
                                ntest,
                                nelems,
                                self.x.shape[1],
                                self.x.shape[2],
                                self.x.shape[3],
                            )

                            # Define new arrays (On the cpu)
                            test_elems = np.array(elem_to_check_per_point)[
                                pt_not_found_this_it
                            ]
                            test_fields = (
                                self.x[test_elems, :, :, :] ** 2
                                + self.y[test_elems, :, :, :] ** 2
                                + self.z[test_elems, :, :, :] ** 2
                            )
                            test_probes = (
                                probes[real_index_pt_not_found_this_it, 0] ** 2
                                + probes[real_index_pt_not_found_this_it, 1] ** 2
                                + probes[real_index_pt_not_found_this_it, 2] ** 2
                            )

                            # Perform the test interpolation
                            self.test_interp[:ntest, :nelems] = (
                                self.ei.interpolate_field_at_rst(
                                    result_r[pt_not_found_this_it].reshape(
                                        test_probe_new_shape
                                    ),
                                    result_s[pt_not_found_this_it].reshape(
                                        test_probe_new_shape
                                    ),
                                    result_t[pt_not_found_this_it].reshape(
                                        test_probe_new_shape
                                    ),
                                    test_fields.reshape(test_elem_new_shape),
                                    use_torch=self.use_torch,
                                )
                            )
                            test_result = self.test_interp[:ntest, :nelems].reshape(
                                ntest
                            )

                            # Check if the test pattern is satisfied
                            test_error = abs(test_probes - test_result.cpu().numpy())

                            # Now assign
                            real_list = np.array(real_index_pt_not_found_this_it)
                            relative_list = np.array(pt_not_found_this_it)
                            better_test = np.where(
                                test_error
                                < test_pattern[real_index_pt_not_found_this_it]
                            )[0]

                            if len(better_test) > 0:
                                probes_rst[real_list[better_test], 0] = result_r[
                                    relative_list[better_test]
                                ]
                                probes_rst[real_list[better_test], 1] = result_s[
                                    relative_list[better_test]
                                ]
                                probes_rst[real_list[better_test], 2] = result_t[
                                    relative_list[better_test]
                                ]
                                el_owner[real_list[better_test]] = np.array(
                                    elem_to_check_per_point
                                )[relative_list[better_test]]
                                glb_el_owner[real_list[better_test]] = (
                                    el_owner[real_list[better_test]] + offset_el
                                )
                                rank_owner[real_list[better_test]] = rank
                                err_code[real_list[better_test]] = not_found_code
                                test_pattern[real_list[better_test]] = test_error[
                                    better_test
                                ]

                        else:

                            probes_rst[real_index_pt_not_found_this_it, 0] = result_r[
                                pt_not_found_this_it
                            ]
                            probes_rst[real_index_pt_not_found_this_it, 1] = result_s[
                                pt_not_found_this_it
                            ]
                            probes_rst[real_index_pt_not_found_this_it, 2] = result_t[
                                pt_not_found_this_it
                            ]
                            el_owner[real_index_pt_not_found_this_it] = np.array(
                                elem_to_check_per_point
                            )[pt_not_found_this_it]
                            glb_el_owner[real_index_pt_not_found_this_it] = (
                                el_owner[real_index_pt_not_found_this_it] + offset_el
                            )
                            rank_owner[real_index_pt_not_found_this_it] = rank
                            err_code[real_index_pt_not_found_this_it] = not_found_code

        return (
            probes,
            probes_rst,
            el_owner,
            glb_el_owner,
            rank_owner,
            err_code,
            test_pattern,
        )

    def redistribute_probes_to_owners_from_io_rank(self, io_rank, comm):

        rank = comm.Get_rank()
        size = comm.Get_size()

        probes = self.probes
        probes_rst = self.probes_rst
        el_owner = self.el_owner
        rank_owner = self.rank_owner
        err_code = self.err_code

        # Sort the points by rank to scatter them easily
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
        sendcounts = np.zeros((size), dtype=np.intc)
        if rank == io_rank:
            for root in range(0, size):
                sendcounts[root] = len(np.where(rank_owner == root)[0])
        comm.Bcast(sendcounts, root=0)

        root = io_rank
        # Redistribute probes
        if rank == root:
            sendbuf = sorted_probes.reshape((sorted_probes.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts * 3, root, np.double, comm)
        my_probes = recvbuf.reshape((int(recvbuf.size / 3), 3))

        # Redistribute probes rst
        if rank == root:
            sendbuf = sorted_probes_rst.reshape((sorted_probes_rst.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts * 3, root, np.double, comm)
        my_probes_rst = recvbuf.reshape((int(recvbuf.size / 3), 3))

        # Redistribute err_code
        if rank == root:
            sendbuf = sorted_err_code.reshape((sorted_err_code.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        my_err_code = recvbuf

        # Redistribute el_owner
        if rank == root:
            sendbuf = sorted_el_owner.reshape((sorted_el_owner.size))
            # print(sendbuf)
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        # print(recvbuf)
        my_el_owner = recvbuf

        # Redistribute el_owner
        if rank == root:
            sendbuf = sorted_rank_owner.reshape((sorted_rank_owner.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        my_rank_owner = recvbuf

        self.my_probes = my_probes
        self.my_probes_rst = my_probes_rst
        self.my_err_code = my_err_code
        self.my_el_owner = my_el_owner
        self.my_rank_owner = my_rank_owner
        self.sendcounts = sendcounts
        self.sort_by_rank = sort_by_rank

        return

    def interpolate_field_from_rst(self, sampled_field):

        if self.use_tensor == False and self.use_torch == False:

            x = self.x
            y = self.y
            z = self.z
            my_probes = self.my_probes
            my_probes_rst = self.my_probes_rst
            my_err_code = self.my_err_code
            sampled_field_at_probe = np.empty((my_probes.shape[0]))

            i = 0  # Counter for the number of probes
            if self.progress_bar:
                pbar = tqdm(total=my_probes.shape[0])
            for e in self.my_el_owner:
                if my_err_code[i] != 0:
                    # self.ei.project_element_into_basis(x[e,:,:,:], y[e,:,:,:], z[e,:,:,:])
                    tmp = self.ei.interpolate_field_at_rst(
                        my_probes_rst[i, 0],
                        my_probes_rst[i, 1],
                        my_probes_rst[i, 2],
                        sampled_field[e, :, :, :],
                    )
                    sampled_field_at_probe[i] = tmp
                else:
                    sampled_field_at_probe[i] = 0
                i = i + 1
                if self.progress_bar:
                    pbar.update(1)
            if self.progress_bar:
                pbar.close()

        elif self.use_tensor == True or self.use_torch == True:

            max_pts = self.max_pts
            pts_n = self.my_probes.shape[0]
            iterations = np.ceil((pts_n / max_pts))

            sampled_field_at_probe = np.empty((self.my_probes.shape[0]))
            probe_interpolated = np.zeros((self.my_probes.shape[0]))

            for i in range(0, int(iterations)):

                # Check the probes to interpolate this iteration
                probes_to_interpolate = np.where(
                    (self.my_err_code != 0) & (probe_interpolated == 0)
                )[0]
                probes_to_interpolate = probes_to_interpolate[:max_pts]

                # Check the number of probes
                npoints = len(probes_to_interpolate)
                nelems = 1

                if npoints == 0:
                    break

                # Inmediately update the points that will be interpolated
                probe_interpolated[probes_to_interpolate] = 1

                rst_new_shape = (npoints, nelems, 1, 1)
                field_new_shape = (
                    npoints,
                    nelems,
                    self.x.shape[1],
                    self.x.shape[2],
                    self.x.shape[3],
                )

                self.test_interp[:npoints, :nelems] = self.ei.interpolate_field_at_rst(
                    self.my_probes_rst[probes_to_interpolate, 0].reshape(rst_new_shape),
                    self.my_probes_rst[probes_to_interpolate, 1].reshape(rst_new_shape),
                    self.my_probes_rst[probes_to_interpolate, 2].reshape(rst_new_shape),
                    sampled_field[self.my_el_owner[probes_to_interpolate]].reshape(
                        field_new_shape
                    ),
                    use_torch=self.use_torch,
                )

                # Populate the sampled field
                if not self.use_torch:
                    sampled_field_at_probe[probes_to_interpolate] = self.test_interp[
                        :npoints, :nelems
                    ].reshape(npoints)
                else:
                    sampled_field_at_probe[probes_to_interpolate] = (
                        self.test_interp[:npoints, :nelems]
                        .reshape(npoints)
                        .to("cpu")
                        .numpy()
                    )

        return sampled_field_at_probe


def pt_in_bbox(pt, bbox, rel_tol=0.01):
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

    if found_x == True and found_y == True and found_z == True:
        state = True
    else:
        state = False

    return state


def get_bbox_from_coordinates(x, y, z):

    nelv = x.shape[0]
    lx = x.shape[3]  # This is not a mistake. This is how the data is read
    ly = x.shape[2]
    lz = x.shape[1]

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

    return bbox_centroid, bbox_max_dist


def get_communication_pairs(self, global_rank_candidate_dict, comm):

    size = comm.Get_size()

    # Create a list with all the ranks
    ranks = [ii for ii in range(0, size)]

    # Get all unique pairs and colours
    if type(global_rank_candidate_dict) is NoneType:
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


def get_candidate_ranks(self, comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Find the bounding box of the rank to create a global but "sparse" kdtree
    rank_bbox = np.zeros((1, 6), dtype=np.double)
    rank_bbox[0, 0] = np.min(self.x)
    rank_bbox[0, 1] = np.max(self.x)
    rank_bbox[0, 2] = np.min(self.y)
    rank_bbox[0, 3] = np.max(self.y)
    rank_bbox[0, 4] = np.min(self.z)
    rank_bbox[0, 5] = np.max(self.z)

    rank_bbox_dist = np.zeros((1, 3), dtype=np.double)
    rank_bbox_dist[0, 0] = rank_bbox[0, 1] - rank_bbox[0, 0]
    rank_bbox_dist[0, 1] = rank_bbox[0, 3] - rank_bbox[0, 2]
    rank_bbox_dist[0, 2] = rank_bbox[0, 5] - rank_bbox[0, 4]
    rank_bbox_max_dist = np.max(
        np.sqrt(
            rank_bbox_dist[:, 0] ** 2
            + rank_bbox_dist[:, 1] ** 2
            + rank_bbox_dist[:, 2] ** 2
        )
        / 2
    )

    rank_bbox_centroid = np.zeros((1, 3))
    rank_bbox_centroid[:, 0] = rank_bbox[:, 0] + rank_bbox_dist[:, 0] / 2
    rank_bbox_centroid[:, 1] = rank_bbox[:, 2] + rank_bbox_dist[:, 1] / 2
    rank_bbox_centroid[:, 2] = rank_bbox[:, 4] + rank_bbox_dist[:, 2] / 2

    global_centroids = np.zeros((size * 3), dtype=np.double)
    global_max_dist = np.zeros((size), dtype=np.double)

    # Gather in all ranks
    comm.Allgather(
        [rank_bbox_centroid.flatten(), MPI.DOUBLE], [global_centroids, MPI.DOUBLE]
    )
    comm.Allgather([rank_bbox_max_dist, MPI.DOUBLE], [global_max_dist, MPI.DOUBLE])
    global_centroids = global_centroids.reshape((size, 3))
    # Create a tree with the rank centroids
    self.global_tree = KDTree(global_centroids)
    candidate_ranks_per_point = self.global_tree.query_ball_point(
        x=self.probe_partition,
        r=np.max(global_max_dist),
        p=2.0,
        eps=1e-8,
        workers=1,
        return_sorted=False,
        return_length=False,
    )

    flattened_list = [item for sublist in candidate_ranks_per_point for item in sublist]
    candidate_ranks = list(set(flattened_list))

    return candidate_ranks


def get_global_candidate_ranks(self, comm, candidate_ranks):

    size = comm.Get_size()

    # Get arrays with number of candidates per rank
    ## Tell how many there are per rank
    n_candidate_ranks_per_rank = np.zeros((size), dtype=np.intc)
    sendbuf = np.ones((1), dtype=np.intc) * len(candidate_ranks)
    comm.Allgather([sendbuf, MPI.INT], [n_candidate_ranks_per_rank, MPI.INT])

    ## Allocate an array in all ranks that tells which are the rank candidates in all other ranks
    nc = np.max(n_candidate_ranks_per_rank)
    rank_candidates_in_all_ranks = np.zeros((size, nc), dtype=np.intc)
    rank_candidates_in_this_rank = (
        np.ones((1, nc), dtype=np.intc) * -1
    )  # Set default to -1 to filter out easily later
    for i in range(0, len(candidate_ranks)):
        rank_candidates_in_this_rank[0, i] = candidate_ranks[i]
    comm.Allgather(
        [rank_candidates_in_this_rank, MPI.INT], [rank_candidates_in_all_ranks, MPI.INT]
    )  # This all gather can be changed for gather and broadcast

    ## Filter out the -1 entries for ranks that had less candidates than the maximun determined before
    # global_rank_candidate_dict = {i: [] for i in range(size)}
    # for i in range(0, size):
    #    for j in range(0, nc):
    #        if rank_candidates_in_all_ranks[i,j] != -1:
    #            global_rank_candidate_dict[i].append(rank_candidates_in_all_ranks[i,j])
    ## Delete the big array
    # del rank_candidates_in_all_ranks

    return rank_candidates_in_all_ranks
