"""
Contains the Re2Mesh class, the in-memory form of a NEKTON ``.re2`` mesh file.
"""

import numpy as np
from mpi4py import MPI

from ..monitoring.logger import Logger
from ..io.utils import linear_owner
from ..io.re2 import (
    RE2_EL_DT,
    RE2_CURVE_DT,
    RE2_BC_DT,
    read_re2,
    write_re2,
    validate_re2_records,
    bc_type_str,
)
from .corner_mesh import CornerMesh

__all__ = ["Re2Mesh"]


class Re2Mesh(CornerMesh):
    """
    A NEKTON ``.re2`` mesh held in memory as its raw record arrays.

    A ``.re2`` stores, for every element, a group id and the coordinates of
    the eight corners, plus curved edge records and boundary condition
    records that refer to elements by their position in the file. Unlike a
    Neko ``.nmsh`` there are no global point ids; those are created when the
    mesh is converted with the tools of ``pysemtools.nmsh``.

    Parameters
    ----------
    elems : ndarray
        Element records of dtype :data:`pysemtools.io.re2.RE2_EL_DT`.
    curves : ndarray
        Curve records of dtype :data:`pysemtools.io.re2.RE2_CURVE_DT`.
    bcs : ndarray
        Boundary condition records of dtype :data:`pysemtools.io.re2.RE2_BC_DT`.
    version : str, optional
        Format version the mesh was read from. Default is ``#v002``, which
        is also the version written.
    comm : MPI.Comm, optional
        Communicator over which the mesh is distributed. Default is None,
        a replicated mesh.
    trailing : int, optional
        Bytes found past the boundary condition section when the mesh was
        read. Default is 0.

    Attributes
    ----------
    elems : ndarray
        Element records owned by this rank.
    curves : ndarray
        Curve records owned by this rank.
    bcs : ndarray
        Boundary condition records owned by this rank.
    curve_index, bc_index : ndarray or None
        Global record positions of the local curve and boundary condition
        records in the file they were read from. None when unknown.

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from pysemtools.datatypes import Re2Mesh
    >>> re2 = Re2Mesh.from_file("hemi.re2", MPI.COMM_WORLD)
    >>> msh = re2.to_sem_mesh(lx=8)
    """

    def __init__(self, elems, curves, bcs, version="#v002", comm=None, trailing=0):
        self.elems = np.ascontiguousarray(elems, dtype=RE2_EL_DT)
        self.curves = np.ascontiguousarray(curves, dtype=RE2_CURVE_DT)
        self.bcs = np.ascontiguousarray(bcs, dtype=RE2_BC_DT)
        self.version = version
        self.trailing = int(trailing)
        self.curve_index = None
        self.bc_index = None
        super().__init__(self.elems.shape[0], comm)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_file(cls, path, comm=None, log_level=None):
        """
        Read a ``.re2`` file.

        Parameters
        ----------
        path : str
            Path of the file.
        comm : MPI.Comm, optional
            Communicator over which to distribute the elements. If None the
            whole mesh is read on the calling process and the result is
            replicated.
        log_level : str, optional
            Logging level passed to :class:`pysemtools.monitoring.logger.Logger`.

        Returns
        -------
        Re2Mesh
            The mesh read from disk, with records upcast to double precision.
        """
        log = Logger(comm=comm or MPI.COMM_WORLD, module_name="Re2Mesh", level=log_level)
        log.tic()
        log.write("info", f"Reading mesh file: {path}")
        data = read_re2(path, comm)
        mesh = cls(
            data.elems, data.curves, data.bcs, data.version, comm=comm, trailing=data.trailing
        )
        mesh.curve_index = data.curve_index
        mesh.bc_index = data.bc_index
        if mesh.trailing:
            log.write(
                "warning",
                f"{mesh.trailing} trailing bytes past the boundary condition section "
                "(further boundary condition fields are not read)",
            )
        ncurves = mesh.curves.shape[0] if comm is None else comm.allreduce(mesh.curves.shape[0])
        nbcs = mesh.bcs.shape[0] if comm is None else comm.allreduce(mesh.bcs.shape[0])
        log.write(
            "info",
            f"Read {mesh.glb_nelv} elements ({data.version}), {ncurves} curve records, "
            f"{nbcs} boundary condition records",
        )
        log.toc()
        return mesh

    def write(self, path, inputs=(), comm=None, log_level=None):
        """
        Write the mesh to a ``.re2`` file (version ``#v002``) atomically.

        A distributed mesh is written collectively, every rank writing its own
        block of elements. A replicated mesh is written by the calling
        process, or by rank 0 only if a communicator is given.

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
        if self.comm is not None:
            log = Logger(comm=self.comm, module_name="Re2Mesh", level=log_level)
            log.write("info", f"Writing mesh file: {path}")
            write_re2(path, self.elems, self.curves, self.bcs, inputs=inputs, comm=self.comm)
            return
        log = Logger(comm=comm or MPI.COMM_WORLD, module_name="Re2Mesh", level=log_level)
        log.write("info", f"Writing mesh file: {path}")
        if comm is None or comm.Get_rank() == 0:
            write_re2(path, self.elems, self.curves, self.bcs, inputs=inputs)
        if comm is not None:
            comm.Barrier()

    # -- distribution -------------------------------------------------------
    def gather(self, root=None):
        """
        Collect the whole mesh on every rank, or on one rank.

        Parameters
        ----------
        root : int, optional
            If given, only this rank receives the mesh and the others get
            None. Default is None, every rank receives it.

        Returns
        -------
        Re2Mesh or None
            A replicated mesh with elements in file order. Curve and boundary
            condition records are restored to file order when the record
            positions are known (a mesh read with :meth:`from_file`),
            otherwise they are ordered by owner rank.
        """
        if self.comm is None:
            return self
        elems = self._gather_records(self.elems, root)
        if self.curve_index is not None:
            curves = self._gather_in_file_order(self.curves, self.curve_index, root)
        else:
            curves = self._gather_records(self.curves, root)
        if self.bc_index is not None:
            bcs = self._gather_in_file_order(self.bcs, self.bc_index, root)
        else:
            bcs = self._gather_records(self.bcs, root)
        if elems is None:
            return None
        return Re2Mesh(elems, curves, bcs, self.version, trailing=self.trailing)

    def distribute(self, comm):
        """
        Distribute a replicated mesh linearly over a communicator.

        Every rank keeps its block of element records and the curve and
        boundary condition records of the elements it owns. A distributed
        mesh is returned unchanged.

        Parameters
        ----------
        comm : MPI.Comm
            The communicator.

        Returns
        -------
        Re2Mesh
            The distributed mesh.
        """
        if self.comm is not None:
            return self
        glb_nelv = self.nelv
        nelv, offset_el = self._local_block(comm)
        size, rank = comm.Get_size(), comm.Get_rank()
        curve_sel = np.flatnonzero(
            linear_owner(self.curves["e"].astype(np.int64) - 1, glb_nelv, size) == rank
        )
        bc_sel = np.flatnonzero(
            linear_owner(self.bcs["e"].astype(np.int64) - 1, glb_nelv, size) == rank
        )
        mesh = Re2Mesh(
            self.elems[offset_el : offset_el + nelv],
            self.curves[curve_sel],
            self.bcs[bc_sel],
            self.version,
            comm=comm,
            trailing=self.trailing,
        )
        mesh.curve_index = curve_sel.astype(np.int64)
        mesh.bc_index = bc_sel.astype(np.int64)
        return mesh

    # -- validation and properties ---------------------------------------------
    def validate(self):
        """
        Check the element, edge and face ranges of the curve and boundary condition records.

        Raises
        ------
        Re2FormatError
            If a record is out of range.
        """
        validate_re2_records(self.glb_nelv, self.curves, self.bcs)

    @property
    def corner_coordinates(self):
        """Coordinates of the local element corners, shape (nelv, 8, 3)."""
        return np.stack([self.elems["x"], self.elems["y"], self.elems["z"]], axis=2)

    @property
    def element_ids(self):
        """Global ids of the local elements, their 1-based position in the file."""
        return np.arange(self.offset_el + 1, self.offset_el + self.nelv + 1, dtype=np.int32)

    @property
    def group_ids(self):
        """Group id of every local element as stored in the file, shape (nelv,)."""
        return self.elems["rg"]

    def bc_types(self):
        """
        Boundary condition types of the local records as strings.

        Returns
        -------
        list of str
            One entry per record in ``bcs``, e.g. ``"W"``, ``"v"``, ``"P"``.
        """
        return [bc_type_str(t) for t in self.bcs["t"]]
