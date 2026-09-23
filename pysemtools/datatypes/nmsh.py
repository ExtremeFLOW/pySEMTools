"""
Contains the NmshMesh class, the in-memory form of a Neko ``.nmsh`` mesh file.
"""

import numpy as np
from mpi4py import MPI

from ..monitoring.logger import Logger
from ..io.utils import (
    linear_distribution,
    linear_owner,
    redistribute_records,
    allgather_records,
)
from ..io.nmsh import (
    EL_DT,
    ZONE_DT,
    CURVE_DT,
    ZONE_PERIODIC,
    ZONE_LABELLED,
    NmshFormatError,
    read_nmsh,
    write_nmsh,
    validate_zones,
    validate_curves,
)

__all__ = ["NmshMesh"]


class NmshMesh:
    """
    A Neko ``.nmsh`` mesh held in memory as its raw record arrays.

    This is the topological counterpart of :class:`pysemtools.datatypes.msh.Mesh`.
    Where ``Mesh`` stores the GLL coordinates of every element,
    ``NmshMesh`` stores what the mesh file stores: eight corner vertices per
    element with their global point ids, the boundary zones and the curved
    edges. Use :meth:`to_sem_mesh` to obtain a ``Mesh`` with the GLL points
    of the straight-sided geometry.

    Like ``Mesh``, the object is distributed when it carries a communicator:
    ``elems`` are then the elements owned by the rank, a contiguous block of
    the file in Neko's linear distribution, and ``zones`` and ``curves`` are
    the records that reference those elements. Without a communicator the
    object is replicated, that is, it holds the whole mesh on the calling
    process. :meth:`gather` and :meth:`distribute` convert between the two.

    Parameters
    ----------
    elems : ndarray
        Element records of dtype :data:`EL_DT`.
    zones : ndarray
        Zone records of dtype :data:`ZONE_DT`.
    curves : ndarray
        Curve records of dtype :data:`CURVE_DT`.
    comm : MPI.Comm, optional
        Communicator over which the mesh is distributed. Default is None,
        a replicated mesh.
    trailing : int, optional
        Number of bytes found past the curve section when the mesh was read.
        Neko's MPI-IO writer does not truncate, so these can be present in a
        valid file. Default is 0.

    Attributes
    ----------
    elems : ndarray
        Element records owned by this rank.
    zones : ndarray
        Zone records owned by this rank.
    curves : ndarray
        Curve records owned by this rank.
    comm : MPI.Comm or None
        The communicator, None for a replicated mesh.
    nelv : int
        Number of elements owned by this rank.
    glb_nelv : int
        Global number of elements.
    offset_el : int
        Global record position of the first local element.
    zone_index, curve_index : ndarray or None
        Global record positions of the local zone and curve records in the
        file they were read from, used to restore file order on
        :meth:`gather`. None when unknown.

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from pysemtools.datatypes import NmshMesh
    >>> comm = MPI.COMM_WORLD
    >>> nmsh = NmshMesh.from_file("box.nmsh", comm)
    >>> nmsh.glb_nelv, nmsh.vertex_ids.shape
    (64, (16, 8))
    """

    def __init__(self, elems, zones, curves, comm=None, trailing=0):
        self.elems = np.ascontiguousarray(elems, dtype=EL_DT)
        self.zones = np.ascontiguousarray(zones, dtype=ZONE_DT)
        self.curves = np.ascontiguousarray(curves, dtype=CURVE_DT)
        self.comm = comm
        self.trailing = int(trailing)
        self.zone_index = None
        self.curve_index = None
        if comm is None:
            self.glb_nelv = self.nelv
            self.offset_el = 0
        else:
            self.glb_nelv = int(comm.allreduce(self.nelv, op=MPI.SUM))
            self.offset_el = int(comm.scan(self.nelv, op=MPI.SUM)) - self.nelv

    # -- construction -------------------------------------------------------
    @classmethod
    def from_file(cls, path, comm=None, validate=True, log_level=None):
        """
        Read a ``.nmsh`` file.

        Parameters
        ----------
        path : str
            Path of the file.
        comm : MPI.Comm, optional
            Communicator over which to distribute the elements. If None the
            whole mesh is read on the calling process and the result is
            replicated.
        validate : bool, optional
            Validate the element ids and the zone and curve records after
            reading. Default is True.
        log_level : str, optional
            Logging level passed to :class:`pysemtools.monitoring.logger.Logger`.

        Returns
        -------
        NmshMesh
            The mesh read from disk.
        """
        log = Logger(comm=comm or MPI.COMM_WORLD, module_name="NmshMesh", level=log_level)
        log.tic()
        log.write("info", f"Reading mesh file: {path}")
        data = read_nmsh(path, comm)
        mesh = cls(data.elems, data.zones, data.curves, comm=comm, trailing=data.trailing)
        mesh.zone_index = data.zone_index
        mesh.curve_index = data.curve_index
        if validate:
            mesh.validate(path=path)
        if mesh.trailing:
            log.write(
                "warning",
                f"{mesh.trailing} trailing bytes past the curve section "
                "(MPI-IO no-truncate artifact, ignored by Neko)",
            )
        nzones = mesh.zones.shape[0] if comm is None else comm.allreduce(mesh.zones.shape[0])
        ncurves = (
            mesh.curves.shape[0] if comm is None else comm.allreduce(mesh.curves.shape[0])
        )
        log.write(
            "info",
            f"Read {mesh.glb_nelv} elements, {nzones} zone records, {ncurves} curved elements",
        )
        log.toc()
        return mesh

    def write(self, path, inputs=(), comm=None, log_level=None):
        """
        Write the mesh to a ``.nmsh`` file atomically.

        A distributed mesh is written collectively, every rank writing its own
        block of elements. A replicated mesh is written by the calling
        process, or by rank 0 only if a communicator is given.

        Zones are written in Neko's order: periodic first, then labelled, then
        any records of other types. Within a type the file order is that of
        the local records, rank by rank for a distributed mesh.

        Parameters
        ----------
        path : str
            Destination path.
        inputs : sequence of str, optional
            Input files that must not be overwritten.
        comm : MPI.Comm, optional
            For a replicated mesh, the communicator whose rank 0 writes the
            file; the other ranks wait at a barrier. Ignored for a
            distributed mesh, which uses its own communicator.
        log_level : str, optional
            Logging level passed to :class:`pysemtools.monitoring.logger.Logger`.
        """
        t = self.zones["t"]
        zone_arrays = (
            self.zones[t == ZONE_PERIODIC],
            self.zones[t == ZONE_LABELLED],
            self.zones[(t != ZONE_PERIODIC) & (t != ZONE_LABELLED)],
        )
        if self.comm is not None:
            log = Logger(comm=self.comm, module_name="NmshMesh", level=log_level)
            log.write("info", f"Writing mesh file: {path}")
            write_nmsh(path, self.elems, zone_arrays, self.curves, inputs=inputs, comm=self.comm)
            return
        log = Logger(comm=comm or MPI.COMM_WORLD, module_name="NmshMesh", level=log_level)
        log.write("info", f"Writing mesh file: {path}")
        if comm is None or comm.Get_rank() == 0:
            write_nmsh(path, self.elems, zone_arrays, self.curves, inputs=inputs)
        if comm is not None:
            comm.Barrier()

    # -- distribution -------------------------------------------------------
    @property
    def is_distributed(self):
        """True if the mesh carries a communicator and holds only local records."""
        return self.comm is not None

    def gather(self):
        """
        Collect the whole mesh on every rank.

        Returns
        -------
        NmshMesh
            A replicated mesh. Elements are in file order. Zones and curves
            are restored to file order when the record positions are known
            (a mesh read with :meth:`from_file`), otherwise they are ordered
            by owner rank.
        """
        if self.comm is None:
            return self
        comm = self.comm
        elems = allgather_records(comm, self.elems)
        zones = allgather_records(comm, self.zones)
        curves = allgather_records(comm, self.curves)
        if self.zone_index is not None:
            zones = zones[np.argsort(allgather_records(comm, self.zone_index), kind="stable")]
        if self.curve_index is not None:
            curves = curves[
                np.argsort(allgather_records(comm, self.curve_index), kind="stable")
            ]
        return NmshMesh(elems, zones, curves, trailing=self.trailing)

    def distribute(self, comm):
        """
        Distribute a replicated mesh linearly over a communicator.

        Every rank keeps its block of element records and the zone and curve
        records whose element it owns, which is what :meth:`from_file` with a
        communicator produces. A distributed mesh is returned unchanged.

        Parameters
        ----------
        comm : MPI.Comm
            The communicator.

        Returns
        -------
        NmshMesh
            The distributed mesh.
        """
        if self.comm is not None:
            return self
        glb_nelv = self.nelv
        nelv, offset_el = linear_distribution(glb_nelv, comm)
        size = comm.Get_size()
        rank = comm.Get_rank()
        elems = self.elems[offset_el : offset_el + nelv]
        zone_owner = linear_owner(self.zones["e"].astype(np.int64) - 1, glb_nelv, size)
        curve_owner = linear_owner(self.curves["e"].astype(np.int64) - 1, glb_nelv, size)
        zone_sel = np.flatnonzero(zone_owner == rank)
        curve_sel = np.flatnonzero(curve_owner == rank)
        mesh = NmshMesh(
            elems, self.zones[zone_sel], self.curves[curve_sel], comm=comm, trailing=self.trailing
        )
        mesh.zone_index = zone_sel.astype(np.int64)
        mesh.curve_index = curve_sel.astype(np.int64)
        return mesh

    # -- validation ---------------------------------------------------------
    def validate(self, path=""):
        """
        Validate the element ids, zone records and curve records.

        The element ids must be a permutation of ``1..glb_nelv``. For a
        distributed mesh this is checked with one all-to-all exchange.

        Parameters
        ----------
        path : str, optional
            File name used in error messages.

        Raises
        ------
        NmshFormatError
            If any record is malformed.
        """
        if self.comm is None:
            self.element_positions()
        else:
            self._validate_ids_distributed()
        validate_zones(self.glb_nelv, self.zones, path)
        validate_curves(self.glb_nelv, self.curves, path)

    def _validate_ids_distributed(self):
        """Check that the element ids form a global permutation of 1..glb_nelv."""
        comm = self.comm
        ids = self.elems["id"].astype(np.int64)
        bad = int(comm.allreduce(int(((ids < 1) | (ids > self.glb_nelv)).any()), op=MPI.MAX))
        if bad:
            raise NmshFormatError(f"Element id out of [1,{self.glb_nelv}]")
        owner = linear_owner(ids - 1, self.glb_nelv, comm.Get_size())
        received, _ = redistribute_records(comm, ids, owner)
        expected = np.arange(self.offset_el + 1, self.offset_el + self.nelv + 1)
        ok = received.size == expected.size and np.array_equal(np.sort(received), expected)
        if not int(comm.allreduce(int(ok), op=MPI.MIN)):
            raise NmshFormatError(
                "Element ids are not a permutation of 1..nelv (duplicate or missing id)"
            )

    # -- basic properties ---------------------------------------------------
    @property
    def nelv(self):
        """Number of elements owned by this rank."""
        return int(self.elems.shape[0])

    @property
    def element_ids(self):
        """Global element ids of the local elements in record order, shape (nelv,), 1-based."""
        return self.elems["id"]

    @property
    def vertex_ids(self):
        """Global point ids of the local element corners, shape (nelv, 8), as int64."""
        return self.elems["v"]["idx"].astype(np.int64)

    @property
    def corner_coordinates(self):
        """Coordinates of the local element corners, shape (nelv, 8, 3)."""
        return self.elems["v"]["xyz"]

    @property
    def periodic_zones(self):
        """Local zone records of periodic facets."""
        return self.zones[self.zones["t"] == ZONE_PERIODIC]

    @property
    def labelled_zones(self):
        """Local zone records of labelled boundary facets. The label is in ``p_f``."""
        return self.zones[self.zones["t"] == ZONE_LABELLED]

    @property
    def legacy_zones(self):
        """Local zone records of the legacy types 1 to 4, which current Neko ignores."""
        return self.zones[(self.zones["t"] >= 1) & (self.zones["t"] <= 4)]

    def element_positions(self):
        """
        Map global element ids to record positions of a replicated mesh.

        A valid ``.nmsh`` may store its element records in any order, but the
        ids must be a permutation of ``1..nelv``.

        Returns
        -------
        ndarray
            Array of shape (nelv + 1,) such that ``pos[id]`` is the 0-based
            record position of the element with global id ``id``.

        Raises
        ------
        NmshFormatError
            If the ids are not a permutation of ``1..nelv``.
        RuntimeError
            If the mesh is distributed. Call :meth:`gather` first.
        """
        if self.comm is not None:
            raise RuntimeError(
                "element_positions() needs the whole mesh; call gather() on a distributed mesh"
            )
        nelv = self.nelv
        elids = self.elems["id"].astype(np.int64)
        if elids.size and (elids.min() < 1 or elids.max() > nelv):
            raise NmshFormatError(f"Element id out of [1,{nelv}]")
        pos = np.full(nelv + 1, -1, dtype=np.int64)
        pos[elids] = np.arange(nelv)
        if (pos[1:] < 0).any():
            raise NmshFormatError(
                "Element ids are not a permutation of 1..nelv (duplicate or missing id)"
            )
        return pos

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
        of its eight corners. Curve records are not applied. A distributed
        ``NmshMesh`` gives a ``Mesh`` distributed the same way; a replicated
        one is distributed over ``comm`` first.

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
        from .nmsh_geometry import to_sem_mesh  # pylint: disable=import-outside-toplevel

        if comm is None:
            comm = self.comm if self.comm is not None else MPI.COMM_WORLD
        source = self if self.comm is not None else self.distribute(comm)
        return to_sem_mesh(source, comm, lx=lx, create_connectivity=create_connectivity)
