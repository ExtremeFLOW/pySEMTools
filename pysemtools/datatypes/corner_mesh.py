"""
Contains the base class of the mesh file data types, meshes stored as element corners.
"""

import numpy as np
from mpi4py import MPI

from ..comm.router import Router
from ..comm.distribution import linear_distribution
from .corner_mesh_geometry import to_sem_mesh

__all__ = ["CornerMesh"]


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
