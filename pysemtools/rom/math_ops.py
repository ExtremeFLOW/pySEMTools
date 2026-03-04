""" Module that defines some mathematical operations"""

from mpi4py import MPI
import numpy as np
from ..comm.router import Router

NoneType = type(None)


class MathOps:
    """Class that contains methods for math operations"""

    def __init__(self):
        pass

    def scale_data(self, x, bm, rows, columns, scale):
        """Method to scale the data with the given mass matrix"""
        # Scale the data with the mass matrix
        if scale == "mult":
            for i in range(0, columns):
                x[:, i] = x[:, i] * bm[:, 0]
        elif scale == "div":
            for i in range(0, columns):
                x[:, i] = x[:, i] / bm[:, 0]

    def get_perp_ratio(self, u, v):
        """Method to get the ratio of how much a vector is perpendicular
        to another one in serial execution"""
        m = u.conj().T @ v
        v_parallel = u @ m
        v_orthogonal = v - v_parallel
        nrm_o = np.linalg.norm(v_orthogonal)
        nrm = np.linalg.norm(v)
        return nrm_o / nrm

    def mpi_get_perp_ratio(self, u, v, comm):
        """Method to get the ratio of how much a vector is perpendicular
        to another one in parallel execution"""
        # Get the local partial step

        mi = u.conj().T @ v

        # Use MPI_SUM to get the global one by aggregating local
        m = np.zeros_like(mi, dtype=mi.dtype)
        comm.Allreduce(mi, m, op=MPI.SUM)

        # Get local parallel component
        v_parallel = u @ m
        # Get local orthogonal component
        v_orthogonal = v - v_parallel

        #  Do a local sum of squares
        nrmi = np.zeros((2))
        nrmi[0] = np.sum(v_orthogonal**2)
        nrmi[1] = np.sum(v**2)

        # Then do an all reduce
        nrm = np.zeros_like(nrmi, dtype=nrmi.dtype)
        comm.Allreduce(nrmi, nrm, op=MPI.SUM)

        # Get the actual norm
        nrm_o = np.sqrt(nrm[0])
        nrm = np.sqrt(nrm[1])

        return nrm_o / nrm

    def gather_modes_and_mass_at_rank0(self, u_1t, bm, n, comm):
        """Method to gather modes and mass matrix in rank zero"""
        # Get information from the communicator
        rank = comm.Get_rank()
        m = u_1t.shape[1]

        u = None  # prepare the buffer for recieving
        bm1sqrt = None  # prepare the buffer for recieving
        if rank == 0:
            # Generate the buffer to gather in rank 0
            u = np.empty((n, m), dtype=u_1t.dtype)
            bm1sqrt = np.empty((n, 1))
        comm.Gather(u_1t, u, root=0)
        comm.Gather(bm, bm1sqrt, root=0)
        return u, bm1sqrt

    def gather_modes_and_mass_at_root(self, u_1t, bm, n, comm, root=0):
        """Method to gather modes and mass matrix in a given root"""

        rt = Router(comm)

        sendbuf = u_1t.reshape((u_1t.size))
        recvbuf, _ = rt.gather_in_root(sendbuf, root, sendbuf.dtype)

        if not isinstance(recvbuf, NoneType):
            u = recvbuf.reshape((n, int(recvbuf.size / n)))
        else:
            u = None

        sendbuf = bm.reshape((bm.size))
        recvbuf, _ = rt.gather_in_root(sendbuf, root, sendbuf.dtype)

        if not isinstance(recvbuf, NoneType):
            bm1sqrt = recvbuf.reshape((n, int(recvbuf.size / n)))
        else:
            bm1sqrt = None

        return u, bm1sqrt
