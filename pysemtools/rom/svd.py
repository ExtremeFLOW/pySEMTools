""" This module contains operations to perform streaming and parallel SVD"""

import sys
import numpy as np
import scipy.optimize
if 'torch' in sys.modules:
    import torch
else:
    torch = None
import sys

NoneType = type(None)


class SVD:
    """Class used to obtain parallel and streaming SVD results
    
    This is the main class of the ROM module and POD wraps arounf it.

    Parameters
    ----------
    logger : Logger
        Logger object to be used for logging.
    bckend : str, optional
        Backend to be used for the math operations. The default is "numpy". "torch"
        can be used if PyTorch is installed.
    
    Returns
    -------
    None
    
    
    """

    def __init__(self, logger, bckend="numpy"):

        self.log = logger
        self.ifget_all_modes = False
        logger.write(
            "debug",
            "ifget_all_modes is hard coded to False. This parameter applies to lcl updates. It controls if one gets all modes in the global rotation, despite keeping less modes locally. I do not see a use for this in production runs. Thus it is set to false. If needed, activate in mpi_spSVD.py module",
        )

        self.log.write('info', f"Using backend: {bckend}")

        self.bckend = bckend
        if bckend == 'torch':

            if sys.modules.get("torch") is None:
                raise ImportError("torch is not installed/imported. Please install it and import it in your driver python script to use the torch backend.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.log.write("info", f"Using device: {self.device}")

    def lcl_to_gbl_svd(self, uii, dii, vtii, k_set, comm):
        """Perform rotations to obtain global modes from local modes.

        Parameters
        ----------
        uii : np.ndarray
            Local left singular vectors.
        dii : np.ndarray
            Local singular values.
        vtii : np.ndarray
            Local right singular vectors.
        k_set : int
            Number of modes that want to be kept globally. This is different than k because
            in local updates, each rank can have different k that will
            provide with ALL global modes.
        comm : MPI communicator
            MPI communicator object.
        
        Returns
        -------

        ui_global : np.ndarray
            Global left singular vectors.
        dy : np.ndarray
            Global singular values.
        vty : np.ndarray
            Global right singular vectors.

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
        
        This method does not require communication.

        Parameters
        ----------
        u_1t : np.ndarray
            Left singular vectors from previous update.
        d_1t : np.ndarray
            Singular values from previous update.
        vt_1t : np.ndarray
            Right singular vectors from previous update.
        xi : np.ndarray
            New data to be used for the update.
        k : int
            Number of modes to be kept.
        
        Returns
        -------
        u_1t : np.ndarray
            Updated left singular vectors.
        d_1t : np.ndarray
            Updated singular values.
        vt_1t : np.ndarray
            Updated right singular vectors.

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
        """perform a global svd that contains all necesary rotations for global modes
        
        Parameters
        ----------
        xi : np.ndarray
            Data to perform the svd on.
        comm : MPI communicator
            MPI communicator object.

        Returns
        -------
        u_local : np.ndarray
            Local left singular vectors after global rotation.
            In this case local references that the modes are distributed across ranks. The modes are global.
        dy : np.ndarray
            Global singular values.
        vty : np.ndarray
            Global right singular vectors.

        """

        if self.bckend == 'numpy':
            return self.gbl_svd_numpy(xi, comm)
        elif self.bckend == 'torch':
            return self.gbl_svd_torch(xi, comm)
    
    def gbl_svd_numpy(self, xi, comm):
        """Perform a global SVD using numpy, including necessary rotations for global modes using numpy
        
        Parameters
        ----------
        xi : np.ndarray
            Data to perform the svd on.
        comm : MPI communicator
            MPI communicator object.
        
        Returns
        -------
        u_local : np.ndarray
            Local left singular vectors after global rotation.
            In this case local references that the modes are distributed across ranks. The modes are global.
        dy : np.ndarray
            Global singular values.
        vty : np.ndarray
            Global right singular vectors.


        """
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
            dy = np.empty((m), dtype=xi.real.dtype)
            vty = np.empty((m, m), dtype=xi.dtype)
        self.log.write("debug", "Broadcasting SVD(y) results")
        comm.Bcast(uy, root=0)
        comm.Bcast(dy, root=0)
        comm.Bcast(vty, root=0)
        # Now matrix multiply each ui by the corresponding entries in uy
        self.log.write("debug", "Performing rotations: local -> global modes")
        u_local = ui @ uy[rank * m : (rank + 1) * m, :]
        
        return u_local, dy, vty
    
    def gbl_svd_torch(self, xi, comm):

        """Perform a global SVD using PyTorch, including necessary rotations for global modes.
        
        Parameters
        ----------
        xi : np.ndarray or torch.Tensor
            Data to perform the svd on.
        comm : MPI communicator
            MPI communicator object.

        Returns
        -------
        u_local : torch.Tensor
            Local left singular vectors after global rotation.
            In this case local references that the modes are distributed across ranks. The modes are global.
        dy : torch.Tensor
            Global singular values.
        vty : torch.Tensor
            Global right singular vectors.


        """
        
        # Ensure xi is a torch tensor on self.device with proper dtype
        if not torch.is_tensor(xi):
            self.log.write("debug", "Converting xi to a torch tensor")
            if isinstance(xi, np.ndarray):
                xi = torch.from_numpy(xi)
            else:
                xi = torch.tensor(xi)
            xi = xi.to(device=self.device)
        
        # Get MPI rank and size
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Get dimensions
        m = xi.shape[1]

        # Perform SVD on each rank independently
        self.log.write("debug", "Performing individual SVD in each rank")
        ui, di, vti = torch.linalg.svd(xi, full_matrices=False)
        self.log.write("debug", "Calculated eigen matrices in each rank")

        yi = torch.diag(di) @ vti

        # Prepare buffer for gathering on rank 0
        y = None
        self.log.write("debug", "Gathering combined eigen matrix in rank 0")
        if rank == 0:
            y = torch.empty((m * size, m), dtype=xi.dtype, device=xi.device)
        
        # Gather yi into y at rank 0; ensure tensors are on CPU for MPI compatibility
        yi_cpu = yi.cpu()
        y_cpu = None
        if rank == 0:
            y_cpu = torch.empty((m * size, m), dtype=yi.dtype)
        
        comm.Gather(yi_cpu.numpy(), y_cpu.numpy() if y_cpu is not None else None, root=0)

        if rank == 0:
            # Move gathered tensor back to the original device
            y = y_cpu.clone().detach().to(xi.device)

        self.log.write("debug", "Performing SVD of combined eigen matrix in rank 0")
        if rank == 0:
            uy, dy, vty = torch.linalg.svd(y, full_matrices=False)
            uy = uy.cpu()
            dy = dy.cpu()
            vty = vty.cpu()
        else:
            # Prepare buffers for broadcast
            uy = torch.empty((m * size, m), dtype=xi.dtype, device='cpu')
            dy = torch.empty((m,), dtype=xi.real.dtype, device='cpu')
            vty = torch.empty((m, m), dtype=xi.dtype, device='cpu')

        self.log.write("debug", "Broadcasting SVD(y) results")
        comm.Bcast(uy.numpy(), root=0)
        comm.Bcast(dy.numpy(), root=0)
        comm.Bcast(vty.numpy(), root=0)

        # Move received tensors back to original device
        uy = torch.tensor(uy.numpy(), device=xi.device)
        dy = torch.tensor(dy.numpy(), device=xi.device)
        vty = torch.tensor(vty.numpy(), device=xi.device)

        # Compute local rotations
        self.log.write("debug", "Performing rotations: local -> global modes")
        u_local = ui @ uy[rank * m : (rank + 1) * m, :]

        return u_local, dy, vty

    def gbl_update(self, u_1t, d_1t, vt_1t, xi, k, comm):
        """
        Method to update the global svds from a batch of data

        Parameters
        ----------
        u_1t : np.ndarray
            Left singular vectors from previous update.
        d_1t : np.ndarray
            Singular values from previous update.
        vt_1t : np.ndarray
            Right singular vectors from previous update.
        xi : np.ndarray
            New data to be used for the update.
        k : int
            Number of modes to be kept.
        comm : MPI communicator
            MPI communicator object.
        
        Returns
        -------
        u_1t : np.ndarray
            Updated left singular vectors.
        d_1t : np.ndarray
            Updated singular values.
        vt_1t : np.ndarray
            Updated right singular vectors.
        
        Notes
        -----           
        No need to delete xi, as this comes from a buffer that is pre allocated
        
        """

        if self.bckend == 'numpy':
            return self.gbl_update_numpy(u_1t, d_1t, vt_1t, xi, k, comm)
        elif self.bckend == 'torch':
            return self.gbl_update_torch(u_1t, d_1t, vt_1t, xi, k, comm)

    def gbl_update_numpy(self, u_1t, d_1t, vt_1t, xi, k, comm):
        """
        Method to update the global svds from a batch of data using numpy
        
        Parameters
        ----------
        u_1t : np.ndarray
            Left singular vectors from previous update.
        d_1t : np.ndarray
            Singular values from previous update.
        vt_1t : np.ndarray
            Right singular vectors.
        xi : np.ndarray
            New data to be used for the update.
        k : int
            Number of modes to be kept.
        comm : MPI communicator
            MPI communicator object.
        
        Returns
        -------
        u_1t : np.ndarray
            Updated left singular vectors.
        d_1t : np.ndarray
            Updated singular values.
        vt_1t : np.ndarray
            Updated right singular vectors.

        Notes
        -----
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

    def gbl_update_torch(self, u_1t, d_1t, vt_1t, xi, k, comm):
        """
        Method to update the global svds from a batch of data using PyTorch

        Parameters
        ----------
        u_1t : torch.Tensor
            Left singular vectors from previous update.
        d_1t : torch.Tensor
            Singular values from previous update.
        vt_1t : torch.Tensor
            Right singular vectors.
        xi : np.ndarray or torch.Tensor
            New data to be used for the update.
        k : int
            Number of modes to be kept.
        comm : MPI communicator
            MPI communicator object.
        
        Returns
        -------
        u_1t : torch.Tensor
            Updated left singular vectors.
        d_1t : torch.Tensor
            Updated singular values.
        vt_1t : torch.Tensor
            Updated right singular vectors.

        Notes
        -----
        No need to delete xi, as this comes from a buffer that is pre allocated

        """
        
        # Check if xi is a torch tensor. If not, cast it and set a flag.
        input_was_numpy = False
        if not torch.is_tensor(xi):
            self.log.write("info", f"Backend is {self.bckend} but input xi is not a torch tensor. Converting to torch tensor.")
            self.log.write("info", "Converting input xi from numpy to torch tensor")
            input_was_numpy = True
            if isinstance(xi, np.ndarray):
                xi = torch.from_numpy(xi)
            else:
                xi = torch.tensor(xi)
            self.log.write("info", f"Transfering to device {self.device}")
            xi = xi.to(device=self.device)
        
        # Also check if u_1t, d_1t, vt_1t are provided and are tensors. If not, convert them.
        if u_1t is not None and not torch.is_tensor(u_1t):
            u_1t = torch.from_numpy(u_1t).to(device=self.device)
        if d_1t is not None and not torch.is_tensor(d_1t):
            d_1t = torch.from_numpy(d_1t).to(device=self.device)
        if vt_1t is not None and not torch.is_tensor(vt_1t):
            vt_1t = torch.from_numpy(vt_1t).to(device=self.device)

        if u_1t is None:
            self.log.write("info", "Initialized the updating arrays")
            u_1t, d_1t, vt_1t = self.gbl_svd(xi, comm)
        else:
            # Perform SVD of the new snapshot.
            self.log.write("debug", "Performing global SVD #1 in update")
            u_tp1, d_tp1, vt_tp1 = self.gbl_svd(xi, comm)
            
            # Construct the block-diagonal matrix from the conjugate transposes of vt_1t and vt_tp1.
            self.log.write("debug", "Appending Vt matrix in update")
            v_tilde = torch.block_diag(vt_1t.conj().T, vt_tp1.conj().T)
            
            # Concatenate the updated basis from previous and new snapshots.
            self.log.write("debug", "Appending New basis to previous ones in update")
            w = torch.cat((u_1t @ torch.diag(d_1t), u_tp1 @ torch.diag(d_tp1)), dim=1)
            
            # Perform SVD on the combined matrix.
            self.log.write("debug", "Performing global SVD #2 in update")
            uw, dw, vtw = self.gbl_svd(w, comm)
            
            # Update the variables.
            u_1t = uw
            d_1t = dw
            vt_1t = (v_tilde @ vtw.conj().T).conj().T

        # Truncate the matrices if the number of columns is larger than k.
        if u_1t.shape[1] >= k:
            u_1t = u_1t[:, :k].clone()
            d_1t = d_1t[:k].clone()
            vt_1t = vt_1t[:k, :].clone()

        # If the original input was a numpy array, convert outputs back to numpy.
        if input_was_numpy:
            u_1t = u_1t.cpu().numpy()
            d_1t = d_1t.cpu().numpy()
            vt_1t = vt_1t.cpu().numpy()

        return u_1t, d_1t, vt_1t
