""" This module contains operations to perform streaming and parallel SVD"""

import numpy as np
import scipy.optimize

NoneType = type(None)


class SVD:
    """Class used to obtain parallel and streaming SVD results"""

    def __init__(self, logger):

        self.log = logger
        self.ifget_all_modes = False
        logger.write(
            "debug",
            "ifget_all_modes is hard coded to False. This parameter applies to lcl updates. It controls if one gets all modes in the global rotation, despite keeping less modes locally. I do not see a use for this in production runs. Thus it is set to false. If needed, activate in mpi_spSVD.py module",
        )

    def lcl_to_gbl_svd(self, uii, dii, vtii, k_set, comm):
        """Perform rotations to obtain global modes from local modes.
        The number of modes that want to be kept. This is different than k because
        in local updates, each rank can have different k that will
        provide with ALL global modes.
        """
        # Get information from the communicator
        rank = comm.Get_rank()
        size = comm.Get_size()

        n = uii.shape[0]
        yii = np.diag(dii) @ vtii  # This is size KxM
        k = yii.shape[0]
        m = yii.shape[1]

        # print("rank "+repr(rank)+" has n="+repr(n)+" m="+repr(m)+" k="+repr(k))

        # yii could have different k accross ranks, so pad them to standard size
        yi = np.zeros((m, m), dtype=yii.dtype)
        if self.ifget_all_modes:
            ui = np.zeros(
                (n, m), dtype=uii.dtype
            )  # Pad the modes to be able to obtain all global rotations
        else:
            ui = np.zeros(
                (n, k), dtype=uii.dtype
            )  # Do not pad the modes, since it requires much memory
        ui[:, :k] = uii[:, :]
        yi[:k, :] = yii[:, :]

        # Gather Yi into Y in rank 0
        # prepare the buffer for recieving
        y = None
        if rank == 0:
            # Generate the buffer to gather in rank 0
            y = np.empty((m * size, m), dtype=yii.dtype)
        comm.Gather(yi, y, root=0)

        if rank == 0:
            # If tank is zero, calculate the svd of the combined eigen matrix
            # Perform the svd of the combined eigen matrix
            # tic_in = time.perf_counter()
            uy, dy, vty = np.linalg.svd(y, full_matrices=False)
            # toc_in = time.perf_counter()
            # print(f"Time for SVD of y in rank {rank}: {toc_in - tic_in:0.4f} seconds")
        else:
            # If the rank is not zero, simply create a buffer to recieve the uy dy and vty
            uy = np.empty((m * size, m), dtype=uii.dtype)
            dy = np.empty((m), dtype=dii.dtype)
            vty = np.empty((m, m), dtype=vtii.dtype)
        comm.Bcast(uy, root=0)
        comm.Bcast(dy, root=0)
        comm.Bcast(vty, root=0)

        # Perform the rotation of the local bases to obtain the global one
        if self.ifget_all_modes:
            ui_global = ui @ uy[rank * m : (rank + 1) * m, :]
        else:
            ui_global = ui @ uy[rank * m : rank * m + k, :k_set]

        string = f"ui is of shape[{ui.shape[0]},{ui.shape[1]}]"
        self.log.write("debug", string)

        string = f"ui_global is of shape[{ui_global.shape[0]},{ui_global.shape[1]}]"
        self.log.write("debug", string)

        if not self.ifget_all_modes:
            ui_global = np.ascontiguousarray(ui_global[:, :k_set])
            dy = np.ascontiguousarray(dy[:k_set])
            vty = np.ascontiguousarray(vty[:k_set, :])

        return ui_global, dy, vty

    def lcl_update(self, u_1t, d_1t, vt_1t, xi, k):
        """
        Method to update the local svds from a batch of data
        xi: new data batch

        """

        if isinstance(u_1t, NoneType):
            # Perform the distributed SVD and don't accumulate
            u_1t, d_1t, vt_1t = np.linalg.svd(xi, full_matrices=False)
        else:
            # Find the svd of the new snapshot
            u_tp1, d_tp1, vt_tp1 = np.linalg.svd(xi, full_matrices=False)
            # 2 contruct matrices to Do the updating
            v_tilde = scipy.linalg.block_diag(vt_1t.conj().T, vt_tp1.conj().T)
            w = np.append(u_1t @ np.diag(d_1t), u_tp1 @ np.diag(d_tp1), axis=1)
            uw, dw, vtw = np.linalg.svd(w, full_matrices=False)
            # 3 Update
            u_1t = uw
            d_1t = dw
            vt_1t = (v_tilde @ vtw.conj().T).conj().T

        # Truncate the matrices if needed.
        # Eliminate the lower energy mode, which should be the last ones
        if u_1t.shape[1] > k:
            u_1t = np.copy(u_1t[:, 0:k])
            d_1t = np.copy(d_1t[0:k])
            vt_1t = np.copy(vt_1t[0:k, :])

        return u_1t, d_1t, vt_1t

    def gbl_svd(self, xi, comm):
        """perform a global svd that contains all necesary rotations
        for global modes"""

        # Get information from the communicator
        rank = comm.Get_rank()
        size = comm.Get_size()
        # Get some set up data
        m = xi.shape[1]

        # Perfrom Svd in all ranks
        # tic_in = time.perf_counter()
        self.log.write("debug", "Performing individual SVD in each rank")
        ui, di, vti = np.linalg.svd(xi, full_matrices=False)
        # toc_in = time.perf_counter()
        self.log.write("debug", f"calculated eigen matrices in each rank")
        yi = np.diag(di) @ vti
        # print(f"Time for SVD of xi in rank {rank}: {toc_in - tic_in:0.4f} seconds")

        # Gather yi into y in rank 0
        # prepare the buffer for recieving
        y = None
        self.log.write("debug", "Gathering combined eigen matrix in rank 0")
        if rank == 0:
            # Generate the buffer to gather in rank 0
            y = np.empty((m * size, m), dtype=xi.dtype)
        comm.Gather(yi, y, root=0)

        self.log.write("debug", "Performing SVD of combined eigen matrix in rank 0")
        if rank == 0:
            # If tank is zero, calculate the svd of the combined eigen matrix
            # Perform the svd of the combined eigen matrix
            # tic_in = time.perf_counter()
            uy, dy, vty = np.linalg.svd(y, full_matrices=False)
            # toc_in = time.perf_counter()
            # print(f"Time for SVD of y in rank {rank}: {toc_in - tic_in:0.4f} seconds")
        else:
            # If the rank is not zero, simply create a buffer to recieve the uy dy and vty
            uy = np.empty((m * size, m), dtype=xi.dtype)
            dy = np.empty((m), dtype=np.double)
            vty = np.empty((m, m), dtype=xi.dtype)
        self.log.write("debug", "Broadcasting SVD(y) results")
        comm.Bcast(uy, root=0)
        comm.Bcast(dy, root=0)
        comm.Bcast(vty, root=0)
        # Now matrix multiply each ui by the corresponding entries in uy
        self.log.write("debug", "Performing rotations: local -> global modes")
        u_local = ui @ uy[rank * m : (rank + 1) * m, :]

        return u_local, dy, vty

    def gbl_update(self, u_1t, d_1t, vt_1t, xi, k, comm):
        """
        Method to update the global svds from a batch of data
        No need to delete xi, as this comes from a buffer that is pre allocated
        """

        if isinstance(u_1t, NoneType):
            self.log.write("info", "Initialized the updating arrays")
            # Perform the distributed SVD and don't accumulate
            u_1t, d_1t, vt_1t = self.gbl_svd(xi, comm)
        else:
            # Find the svd of the new snapshot
            self.log.write("debug", "Performing global SVD #1 in update")
            u_tp1, d_tp1, vt_tp1 = self.gbl_svd(xi, comm)
            # 2 contruct matrices to Do the updating
            self.log.write("debug", "Appending Vt matrix in update")
            v_tilde = scipy.linalg.block_diag(vt_1t.conj().T, vt_tp1.conj().T)
            self.log.write("debug", "Appending New basis to previous ones in update")
            w = np.append(u_1t @ np.diag(d_1t), u_tp1 @ np.diag(d_tp1), axis=1)
            self.log.write("debug", "Performing global SVD #2 in update")
            uw, dw, vtw = self.gbl_svd(w, comm)
            # 3 Update
            u_1t = uw
            d_1t = dw
            vt_1t = (v_tilde @ vtw.conj().T).conj().T

        # Truncate the matrices if needed.
        # Eliminate the lower energy mode, which should be the last ones
        if u_1t.shape[1] >= k:
            u_1t = np.copy(u_1t[:, 0:k])
            d_1t = np.copy(d_1t[0:k])
            vt_1t = np.copy(vt_1t[0:k, :])

        return u_1t, d_1t, vt_1t
