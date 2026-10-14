"""
Distribution of items over the ranks of a communicator.

The package distributes elements, probes and file records linearly: the first
``n % size`` ranks get one item more than the others, and every rank owns a
contiguous block. This is the distribution of Neko's ``linear_dist_t`` and of
the parallel readers and writers of this package.
"""

import numpy as np

__all__ = ["linear_distribution", "linear_owner"]


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
