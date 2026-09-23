"""Utilities for I/O operations"""

import os
import tempfile

import numpy as np
from mpi4py import MPI


class IoPathData:
    """This class stores the path, casename and index of fld inputs"""

    def __init__(self, params_file):
        self.casename = params_file["casename"]
        self.dataPath = params_file["dataPath"]
        self.index = params_file["first_index"]


def get_fld_from_ndarray(array, lx, ly, lz, nelv):
    """Reshape a 1D array obtained from fortran into a 4D array
    compliant with pySEMTools"""

    fld = array.reshape((nelv, lz, ly, lx))

    return fld



class AtomicOutput:
    """
    Context manager that writes to a temporary file and renames it into place on success.

    The output never clobbers one of the input files and a failed run never
    leaves a partial file behind: any exception unlinks the temporary file and
    the destination is not touched. In serial the file object is a regular
    Python file. With a communicator the file is an ``MPI.File`` opened
    collectively, and the rename happens on rank 0 after a barrier.

    Parameters
    ----------
    path : str
        Destination path.
    inputs : sequence of str, optional
        Paths of the input files. If ``path`` resolves to one of them a
        ``ValueError`` is raised before anything is written.
    comm : MPI.Comm, optional
        Communicator for a collective MPI-IO write. Default is a serial write.

    Examples
    --------
    >>> with AtomicOutput("mesh.nmsh", inputs=("mesh.re2",)) as f:
    ...     f.write(b"...")
    """

    def __init__(self, path, inputs=(), comm=None):
        real_path = os.path.realpath(path)
        for src in inputs:
            if os.path.exists(src) and os.path.realpath(src) == real_path:
                raise ValueError(f"Output {path} would overwrite an input file")
        self.path = path
        self.comm = comm
        directory = os.path.dirname(real_path) or "."
        if comm is None or comm.Get_rank() == 0:
            fd, self.tmp = tempfile.mkstemp(
                prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
            )
        else:
            fd, self.tmp = None, None
        if comm is None:
            self.f = os.fdopen(fd, "wb")
        else:
            if fd is not None:
                os.close(fd)
            self.tmp = comm.bcast(self.tmp, root=0)
            self.f = MPI.File.Open(comm, self.tmp, MPI.MODE_WRONLY | MPI.MODE_CREATE)

    def __enter__(self):
        return self.f

    def __exit__(self, exc_type, exc, tb):
        if self.comm is None:
            self.f.close()
            if exc_type is None:
                os.replace(self.tmp, self.path)
            else:
                try:
                    os.unlink(self.tmp)
                except OSError:
                    pass
            return False
        self.f.Close()
        failed = self.comm.allreduce(1 if exc_type is not None else 0, op=MPI.MAX)
        if self.comm.Get_rank() == 0:
            if failed:
                try:
                    os.unlink(self.tmp)
                except OSError:
                    pass
            else:
                os.replace(self.tmp, self.path)
        self.comm.Barrier()
        return False


def linear_distribution(glb_n, comm, rank=None):
    """
    Block of a linear distribution owned by a rank.

    This is the distribution of Neko's ``linear_dist_t`` and of the parallel
    field readers of this package: the first ``glb_n % size`` ranks get one
    item more than the others.

    Parameters
    ----------
    glb_n : int
        Global number of items.
    comm : MPI.Comm
        MPI communicator.
    rank : int, optional
        Rank to compute the block for. Default is the rank of ``comm``.

    Returns
    -------
    n : int
        Number of items owned by the rank.
    offset : int
        Global index of the first item owned by the rank.
    """
    size = comm.Get_size()
    if rank is None:
        rank = comm.Get_rank()
    base, rem = divmod(int(glb_n), size)
    n = base + (1 if rank < rem else 0)
    offset = rank * base + min(rank, rem)
    return n, offset


def linear_owner(index, glb_n, size):
    """
    Rank that owns a global index in a linear distribution.

    Parameters
    ----------
    index : ndarray
        0-based global indices.
    glb_n : int
        Global number of items.
    size : int
        Number of ranks.

    Returns
    -------
    ndarray
        Owner rank of each index, int64.
    """
    base, rem = divmod(int(glb_n), size)
    idx = np.asarray(index, dtype=np.int64)
    split = rem * (base + 1)
    if base == 0:
        return idx.copy()
    return np.where(idx < split, idx // (base + 1), rem + (idx - split) // base)
