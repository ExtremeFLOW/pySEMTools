"""
Contains the base class of the mesh file data types, meshes stored as element corners.
"""

import numpy as np
from mpi4py import MPI

from ..comm.router import Router
from ..comm.distribution import linear_distribution
from .corner_mesh_geometry import to_sem_mesh

__all__ = ["CornerMesh", "deduplicate_points"]


class CornerMesh:
    """
    Base class of meshes described by the eight corners of every hexahedral element.

    This is what mesh files store, as opposed to the GLL points of every
    element that :class:`pysemtools.datatypes.msh.Mesh` stores. Concrete
    subclasses are :class:`pysemtools.datatypes.nmsh.NmshMesh` for Neko
    ``.nmsh`` files and :class:`pysemtools.datatypes.re2.Re2Mesh` for NEKTON
    ``.re2`` files.

    Like ``Mesh``, the object is distributed when it carries a communicator:
    the local records are then a contiguous block of the file in Neko's
    linear distribution. Without a communicator the object is replicated,
    that is, it holds the whole mesh on the calling process.
    :meth:`gather` and :meth:`distribute` convert between the two.

    Parameters
    ----------
    nelv : int
        Number of elements held by this rank.
    comm : MPI.Comm, optional
        Communicator over which the mesh is distributed. Default is None,
        a replicated mesh.

    Attributes
    ----------
    comm : MPI.Comm or None
        The communicator, None for a replicated mesh.
    rt : Router or None
        The router used to move records between ranks, None for a replicated mesh.
    glb_nelv : int
        Global number of elements.
    offset_el : int
        Global record position of the first local element.
    """

    def __init__(self, nelv, comm=None):
        self.comm = comm
        self.rt = None
        if comm is None:
            self.glb_nelv = int(nelv)
            self.offset_el = 0
        else:
            self.rt = Router(comm)
            self.glb_nelv = int(comm.allreduce(int(nelv), op=MPI.SUM))
            self.offset_el = int(comm.scan(int(nelv), op=MPI.SUM)) - int(nelv)

    # -- to be provided by subclasses -------------------------------------
    @property
    def corner_coordinates(self):
        """Coordinates of the local element corners, shape (nelv, 8, 3)."""
        raise NotImplementedError

    @property
    def element_ids(self):
        """Global element ids of the local elements, shape (nelv,), 1-based."""
        raise NotImplementedError

    def gather(self, root=None):
        """Collect the whole mesh on every rank, or on ``root`` only."""
        raise NotImplementedError

    def distribute(self, comm):
        """Distribute a replicated mesh linearly over a communicator."""
        raise NotImplementedError

    # -- shared behaviour -------------------------------------------------
    @property
    def nelv(self):
        """Number of elements owned by this rank."""
        return int(self.corner_coordinates.shape[0])

    @property
    def is_distributed(self):
        """True if the mesh carries a communicator and holds only local records."""
        return self.comm is not None

    def centroids(self):
        """
        Centroids of the local straight-sided elements.

        Returns
        -------
        ndarray
            Array of shape (nelv, 3).
        """
        return self.corner_coordinates.mean(axis=1)

    def bounding_box(self):
        """
        Bounding box of the whole mesh.

        Returns
        -------
        lo : ndarray
            Minimum x, y, z, shape (3,).
        hi : ndarray
            Maximum x, y, z, shape (3,).
        """
        xyz = self.corner_coordinates.reshape(-1, 3)
        if xyz.shape[0]:
            lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        else:
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        if self.comm is not None:
            lo = np.ascontiguousarray(lo, dtype=np.float64)
            hi = np.ascontiguousarray(hi, dtype=np.float64)
            self.comm.Allreduce(MPI.IN_PLACE, lo, op=MPI.MIN)
            self.comm.Allreduce(MPI.IN_PLACE, hi, op=MPI.MAX)
        return lo, hi

    def to_sem_mesh(self, comm=None, lx=3, create_connectivity=False):
        """
        Build a :class:`pysemtools.datatypes.msh.Mesh` from the straight-sided geometry.

        The GLL points of every element are obtained from the trilinear map
        of its eight corners. Curved edges are not applied. A distributed
        mesh gives a ``Mesh`` distributed the same way; a replicated one is
        distributed over ``comm`` first.

        Parameters
        ----------
        comm : MPI.Comm, optional
            MPI communicator for the ``Mesh`` object. Default is the
            communicator of a distributed mesh, or ``MPI.COMM_WORLD``.
        lx : int, optional
            Number of GLL points per direction. Default is 3.
        create_connectivity : bool, optional
            Passed on to the ``Mesh`` constructor. Default is False.

        Returns
        -------
        Mesh
            Mesh with ``x``, ``y``, ``z`` of shape (nelv, lx, lx, lx) and
            ``elmap`` set to the global element ids.
        """
        if comm is None:
            comm = self.comm if self.comm is not None else MPI.COMM_WORLD
        source = self if self.comm is not None else self.distribute(comm)
        return to_sem_mesh(source, comm, lx=lx, create_connectivity=create_connectivity)

    # -- helpers for subclasses -------------------------------------------
    def _local_block(self, comm):
        """Local block of a linear distribution of the replicated mesh over ``comm``."""
        return linear_distribution(self.nelv, comm)

    def _gather_records(self, records, root=None):
        """Gather local records to all ranks (``root`` None) or to ``root``."""
        if root is None:
            gathered, _ = self.rt.all_gather(data=records, dtype=records.dtype)
        else:
            gathered, _ = self.rt.gather_in_root(data=records, root=root, dtype=records.dtype)
        return gathered

    def _route_to_owner(self, records, owner):
        """Send every record to the rank in ``owner``; the result is ordered by source rank."""
        destinations = list(range(self.comm.Get_size()))
        _, chunks = self.rt.all_to_all(
            destination=destinations,
            data=[records[owner == r] for r in destinations],
            dtype=records.dtype,
        )
        if len(chunks) == 0:
            return np.empty(0, dtype=records.dtype)
        return np.concatenate(chunks)

    def _gather_in_file_order(self, records, index, root=None):
        """Gather records and restore their file order from their global record positions."""
        all_records = self._gather_records(records, root)
        all_index = self._gather_records(index, root)
        if all_records is None:
            return None
        return all_records[np.argsort(all_index, kind="stable")]


def deduplicate_points(xyz):
    """
    Number the distinct corner coordinates.

    Every corner gets the index of its point, 0-based and in order of first
    appearance. Coordinates are compared bit-exactly as float64 triples,
    which matches the point table of Neko's ``.re2`` reader.

    Parameters
    ----------
    xyz : ndarray
        Corner coordinates, shape (nelv, 8, 3), float64.

    Returns
    -------
    vid : ndarray
        Point index of every corner, shape (nelv, 8), int64.
    n_unique : int
        Number of distinct points.

    Notes
    -----
    The corners are sorted as triples of 64-bit integers with a stable sort,
    so equal corners become adjacent and the first appearance leads each
    group. The transient memory cost is roughly 36 bytes per corner on top
    of the input.
    """
    # The comments follow a small example with four corners and three distinct
    # points, written as letters:
    #
    #   corner i:  0  1  2  3
    #   point:     B  A  B  C
    #
    # The expected result numbers the points in order of first appearance,
    # B = 0, A = 1, C = 2, so vid = [0, 1, 0, 2].

    # One row per corner, in element order. The contiguous layout is needed
    # for the integer view below.
    n8 = xyz.shape[0] * 8
    flat = np.ascontiguousarray(xyz, dtype=np.float64).reshape(n8, 3)

    # Reinterpret the three float64 of each row as three 64-bit integers,
    # without copying. Two corners are then equal exactly when their integers
    # are, and integers sort faster than floats and need no NaN handling.
    u = flat.view(np.uint64)

    # Stable sort of the corners by (x, y, z). Equal corners become adjacent,
    # and because the sort is stable the corner that appears first in the
    # mesh leads its group. order[s] is the corner at sorted position s.
    # Example, assuming the integer order A < B < C:
    #   sorted positions 0..3 hold A, B, B, C
    #   order = [1, 0, 2, 3]
    order = np.lexsort((u[:, 2], u[:, 1], u[:, 0]))

    # new[s] is True where sorted position s starts a new group, that is
    # where the corner differs from the one before it. The sorted copy is
    # the largest temporary of the function and is dropped right away.
    # Example: new = [True, True, False, True]
    new = np.empty(n8, dtype=bool)
    new[0] = True
    sorted_u = u[order]
    np.any(sorted_u[1:] != sorted_u[:-1], axis=1, out=new[1:])
    del sorted_u

    # grp[s]: group of sorted position s, numbered in sorted order.
    # first[k]: the first corner of group k, which by stability is where the
    # point first appears in the mesh.
    # Example: grp = [0, 1, 1, 2] and first = [1, 0, 3]
    grp = np.cumsum(new) - 1
    first = order[new]

    # Sorted order is meaningless, so convert it into first-appearance
    # order. Sorting first gives the groups in order of appearance, and the
    # scatter assignment inverts that permutation: rank[k] is the appearance
    # rank of group k.
    # Example: argsort(first) = [1, 0, 2] (B, A, C), so rank = [1, 0, 2],
    # that is A has rank 1, B rank 0 and C rank 2.
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)

    # Scatter the rank of every group back from sorted positions to corners
    # and restore the (nelv, 8) layout.
    # Example: rank[grp] = [1, 0, 0, 2], written to corners [1, 0, 2, 3],
    # so vid = [0, 1, 0, 2].
    vid = np.empty(n8, dtype=np.int64)
    vid[order] = rank[grp]
    return vid.reshape(-1, 8), int(first.size)
