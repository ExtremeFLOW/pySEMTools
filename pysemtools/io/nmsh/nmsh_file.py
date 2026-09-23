"""
Byte layout and parallel readers and writers of Neko ``.nmsh`` mesh files.

All knowledge about the layout of the file lives in this module: the
structured dtypes of the records, the vertex ordering tables and the functions
that read, validate and write the sections. The
:class:`pysemtools.datatypes.nmsh.NmshMesh` class wraps the arrays produced
here.

Reading and writing are parallel. As in the rest of the package, and as in
Neko itself, the elements of a file are distributed linearly over the ranks
of the communicator: rank ``k`` owns the ``k``-th contiguous block of element
records, with block sizes ``glb_nelv // size`` plus one for the first
``glb_nelv % size`` ranks. Each rank reads its block with collective MPI-IO,
and zone and curve records are handed to the rank that owns their element.
"""

import os

import numpy as np
from mpi4py import MPI

from ..utils import (
    AtomicOutput,
    linear_distribution,
    linear_owner,
    record_datatype,
    redistribute_records,
)

__all__ = [
    "EL_DT",
    "ZONE_DT",
    "CURVE_DT",
    "FACE_VERTICES",
    "EDGE_VERTICES",
    "VERTEX_IJK",
    "MAX_ZONE_LABELS",
    "ZONE_PERIODIC",
    "ZONE_LABELLED",
    "NmshFormatError",
    "NmshData",
    "read_nmsh",
    "iter_nmsh_elements",
    "write_nmsh",
    "validate_zones",
    "validate_curves",
]

# ---------------------------------------------------------------------------
# Record layouts (little endian, exactly as Neko writes them)
# ---------------------------------------------------------------------------

#: One element record of a ``.nmsh``: global id and eight vertices, each with
#: a global point id and its coordinates.
EL_DT = np.dtype(
    [("id", "<i4"), ("v", [("idx", "<i4"), ("xyz", "<f8", (3,))], (8,))]
)

#: One zone record of a ``.nmsh``: element, facet, periodic partner element and
#: facet (the label of a labelled zone is stored in ``p_f``), the merged global
#: ids of the four facet corners and the zone type.
ZONE_DT = np.dtype(
    [
        ("e", "<i4"),
        ("f", "<i4"),
        ("p_e", "<i4"),
        ("p_f", "<i4"),
        ("g", "<i4", (4,)),
        ("t", "<i4"),
    ]
)

#: One curve record of a ``.nmsh``: element, five curve parameters for each of
#: the twelve edges and the curve type of each edge.
CURVE_DT = np.dtype([("e", "<i4"), ("data", "<f8", (12, 5)), ("type", "<i4", (12,))])

assert EL_DT.itemsize == 228 and ZONE_DT.itemsize == 36 and CURVE_DT.itemsize == 532

#: Size in bytes of the ``(nelv, gdim)`` header and of every section count.
HEADER_BYTES = 8
COUNT_BYTES = 4

# ---------------------------------------------------------------------------
# Vertex ordering tables
# ---------------------------------------------------------------------------

#: The four vertex slots (0-based) that make up each of the six facets, in
#: the order Neko uses when storing periodic point ids.
FACE_VERTICES = (
    np.array(
        [[1, 5, 8, 4], [2, 6, 7, 3], [1, 2, 6, 5], [4, 3, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]],
        dtype=np.int64,
    )
    - 1
)

#: The two vertex slots (0-based) that make up each of the twelve edges.
EDGE_VERTICES = (
    np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [1, 4],
            [2, 3],
            [5, 8],
            [6, 7],
            [1, 5],
            [2, 6],
            [4, 8],
            [3, 7],
        ],
        dtype=np.int64,
    )
    - 1
)

#: Position of each of the eight vertex slots in the reference cube, as
#: (i, j, k) offsets in {0, 1}.
VERTEX_IJK = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.int64,
)

#: Largest boundary label Neko accepts.
MAX_ZONE_LABELS = 20

#: Zone type of a periodic facet.
ZONE_PERIODIC = 5
#: Zone type of a labelled boundary facet.
ZONE_LABELLED = 7


class NmshFormatError(ValueError):
    """Raised when a mesh file is malformed or violates the format contract."""


class NmshData:
    """
    The raw record arrays of a ``.nmsh`` file as returned by :func:`read_nmsh`.

    Parameters
    ----------
    elems : ndarray
        Element records of dtype :data:`EL_DT` held by this rank.
    zones : ndarray
        Zone records of dtype :data:`ZONE_DT` held by this rank.
    curves : ndarray
        Curve records of dtype :data:`CURVE_DT` held by this rank.
    glb_nelv : int
        Global number of elements in the file.
    offset_el : int
        Record position of the first local element in the file.
    trailing : int, optional
        Bytes found past the curve section. Default is 0.
    zone_index, curve_index : ndarray, optional
        Record positions in the file of the local zone and curve records.
        Default is the local positions, which is correct for a serial read.
    """

    def __init__(
        self,
        elems,
        zones,
        curves,
        glb_nelv,
        offset_el,
        trailing=0,
        zone_index=None,
        curve_index=None,
    ):
        self.elems = elems
        self.zones = zones
        self.curves = curves
        self.glb_nelv = int(glb_nelv)
        self.offset_el = int(offset_el)
        self.trailing = int(trailing)
        if zone_index is None:
            zone_index = np.arange(zones.shape[0], dtype=np.int64)
        if curve_index is None:
            curve_index = np.arange(curves.shape[0], dtype=np.int64)
        self.zone_index = zone_index
        self.curve_index = curve_index


# ---------------------------------------------------------------------------
# .nmsh
# ---------------------------------------------------------------------------
def _read_count(f, what, path):
    """Read one int32 count from a Python file, refusing truncated or negative values."""
    a = np.fromfile(f, dtype="<i4", count=1)
    if a.size != 1:
        raise NmshFormatError(f"{path}: truncated file (missing {what} count)")
    n = int(a[0])
    if n < 0:
        raise NmshFormatError(f"{path}: negative {what} count ({n}), corrupt file?")
    return n


def _read_nmsh_serial(path):
    """Read a whole ``.nmsh`` on the calling process with regular file IO."""
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype="<i4", count=2)
        if hdr.size != 2:
            raise NmshFormatError(f"{path} is not a Neko .nmsh file (short header)")
        nelv, gdim = int(hdr[0]), int(hdr[1])
        if gdim != 3:
            raise NmshFormatError(
                f"{path}: only 3D (hexahedral) meshes are supported (gdim={gdim})"
            )
        if nelv < 1:
            raise NmshFormatError(f"{path}: non-positive element count ({nelv})")
        elems = np.fromfile(f, dtype=EL_DT, count=nelv)
        if elems.size != nelv:
            raise NmshFormatError(
                f"{path}: truncated element section ({elems.size} of {nelv} records)"
            )
        nzones = _read_count(f, "zone", path)
        zones = np.fromfile(f, dtype=ZONE_DT, count=nzones)
        if zones.size != nzones:
            raise NmshFormatError(
                f"{path}: truncated zone section ({zones.size} of {nzones} records)"
            )
        ncurves = _read_count(f, "curve", path)
        curves = np.fromfile(f, dtype=CURVE_DT, count=ncurves)
        if curves.size != ncurves:
            raise NmshFormatError(
                f"{path}: truncated curve section ({curves.size} of {ncurves} records)"
            )
        here = f.tell()
        f.seek(0, os.SEEK_END)
        trailing = f.tell() - here
    return NmshData(elems, zones, curves, nelv, 0, trailing=trailing)


def _read_section_distributed(fh, offset, count, dtype, glb_nelv, comm, path, what):
    """
    Read a record section collectively and hand each record to the owner of its element.

    Every rank reads a linear block of the records at ``offset``, then the
    records are redistributed by the element they reference (field ``e``,
    1-based), following Neko's reader which assumes element ids equal the
    record position in the file. Returns the local records and their global
    record indices in the file.
    """
    n, start = linear_distribution(count, comm)
    block = np.empty(n, dtype=dtype)
    rec_t = record_datatype(dtype)
    fh.Read_at_all(offset + start * dtype.itemsize, [block.view(np.uint8), n, rec_t])
    rec_t.Free()
    e = block["e"].astype(np.int64)
    if ((e < 1) | (e > glb_nelv)).any():
        raise NmshFormatError(
            f"{path}: {what} record references element outside [1,{glb_nelv}]"
        )
    owner = linear_owner(e - 1, glb_nelv, comm.Get_size())
    index = np.arange(start, start + n, dtype=np.int64)
    records, order = redistribute_records(comm, block, owner)
    index, _ = redistribute_records(comm, index[order], owner[order])
    return records, index


def _read_nmsh_distributed(path, comm):
    """Read a ``.nmsh`` collectively, distributing the elements linearly over ``comm``."""
    fh = MPI.File.Open(comm, path, MPI.MODE_RDONLY)
    try:
        file_size = fh.Get_size()
        hdr = np.zeros(2, dtype="<i4")
        if file_size >= HEADER_BYTES:
            fh.Read_at_all(0, hdr)
        else:
            fh.Read_at_all(0, np.zeros(0, dtype="<i4"))
            raise NmshFormatError(f"{path} is not a Neko .nmsh file (short header)")
        glb_nelv, gdim = int(hdr[0]), int(hdr[1])
        if gdim != 3:
            raise NmshFormatError(
                f"{path}: only 3D (hexahedral) meshes are supported (gdim={gdim})"
            )
        if glb_nelv < 1:
            raise NmshFormatError(f"{path}: non-positive element count ({glb_nelv})")

        offset = HEADER_BYTES
        el_end = offset + glb_nelv * EL_DT.itemsize
        if file_size < el_end + COUNT_BYTES:
            raise NmshFormatError(
                f"{path}: truncated element section "
                f"({(file_size - offset) // EL_DT.itemsize} of {glb_nelv} records)"
            )
        nelv, offset_el = linear_distribution(glb_nelv, comm)
        elems = np.empty(nelv, dtype=EL_DT)
        rec_t = record_datatype(EL_DT)
        fh.Read_at_all(offset + offset_el * EL_DT.itemsize, [elems.view(np.uint8), nelv, rec_t])
        rec_t.Free()
        offset = el_end

        count = np.zeros(1, dtype="<i4")
        fh.Read_at_all(offset, count)
        nzones = int(count[0])
        offset += COUNT_BYTES
        if nzones < 0 or file_size < offset + nzones * ZONE_DT.itemsize + COUNT_BYTES:
            raise NmshFormatError(f"{path}: truncated or corrupt zone section")
        zones, zone_index = _read_section_distributed(
            fh, offset, nzones, ZONE_DT, glb_nelv, comm, path, "zone"
        )
        offset += nzones * ZONE_DT.itemsize

        fh.Read_at_all(offset, count)
        ncurves = int(count[0])
        offset += COUNT_BYTES
        if ncurves < 0 or file_size < offset + ncurves * CURVE_DT.itemsize:
            raise NmshFormatError(f"{path}: truncated or corrupt curve section")
        curves, curve_index = _read_section_distributed(
            fh, offset, ncurves, CURVE_DT, glb_nelv, comm, path, "curve"
        )
        offset += ncurves * CURVE_DT.itemsize
        trailing = file_size - offset
    finally:
        fh.Close()
    return NmshData(
        elems,
        zones,
        curves,
        glb_nelv,
        offset_el,
        trailing=trailing,
        zone_index=zone_index,
        curve_index=curve_index,
    )


def read_nmsh(path, comm=None):
    """
    Read a ``.nmsh`` file.

    Parameters
    ----------
    path : str
        Path of the ``.nmsh`` file.
    comm : MPI.Comm, optional
        If given, the elements are distributed linearly over the ranks and
        read with collective MPI-IO. If None, the calling process reads the
        whole file.

    Returns
    -------
    NmshData
        The raw records. They are in file order when read serially; with a
        communicator ``elems`` is the block of the rank and the zone and
        curve records are those of the local elements.

    Raises
    ------
    NmshFormatError
        If the file is not a 3D ``.nmsh`` or a section is truncated.
    """
    if comm is None:
        return _read_nmsh_serial(path)
    return _read_nmsh_distributed(path, comm)


def iter_nmsh_elements(path, chunk=2**21):
    """
    Iterate over the element section of a ``.nmsh`` in chunks on one process.

    This is the low-memory alternative to :func:`read_nmsh` for reductions
    that never need the whole mesh at once, such as bounding boxes or
    Jacobian scans.

    Parameters
    ----------
    path : str
        Path of the ``.nmsh`` file.
    chunk : int, optional
        Number of elements per chunk. Default is 2**21.

    Yields
    ------
    start : int
        Record position of the first element of the chunk.
    elems : ndarray
        Structured array of dtype :data:`EL_DT` with the elements of the chunk.
    """
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype="<i4", count=2)
        if hdr.size != 2 or int(hdr[1]) != 3:
            raise NmshFormatError(f"{path} is not a 3D Neko .nmsh file")
        nelv = int(hdr[0])
        done = 0
        while done < nelv:
            n = min(chunk, nelv - done)
            e = np.fromfile(f, dtype=EL_DT, count=n)
            if e.size != n:
                raise NmshFormatError(
                    f"{path}: truncated element section ({done + e.size} of {nelv} records)"
                )
            yield done, e
            done += n


def _write_block_collective(fh, offset, records, comm):
    """
    Write the records of every rank contiguously at ``offset``, ordered by rank.

    Returns the offset past the written section.
    """
    n = int(records.shape[0])
    counts = np.array(comm.allgather(n), dtype=np.int64)
    start = int(counts[: comm.Get_rank()].sum())
    rec_t = record_datatype(records.dtype)
    fh.Write_at_all(
        offset + start * records.dtype.itemsize,
        [np.ascontiguousarray(records).view(np.uint8), n, rec_t],
    )
    rec_t.Free()
    return offset + int(counts.sum()) * records.dtype.itemsize


def write_nmsh(path, elems, zone_arrays, curves, inputs=(), comm=None):
    """
    Write a ``.nmsh`` file atomically.

    Parameters
    ----------
    path : str
        Destination path.
    elems : ndarray
        Element records of dtype :data:`EL_DT`. With a communicator these
        are the local elements, written as the contiguous block of the rank.
    zone_arrays : sequence of ndarray
        Zone record arrays of dtype :data:`ZONE_DT`, written in the given
        order. Neko writes periodic zones first, then labelled zones. With a
        communicator every array is written as one section ordered by rank.
    curves : ndarray
        Curve records of dtype :data:`CURVE_DT`.
    inputs : sequence of str, optional
        Input files that must not be overwritten.
    comm : MPI.Comm, optional
        Communicator for a collective write. Default is a serial write by
        the calling process.
    """
    if comm is None:
        nzones = sum(int(z.shape[0]) for z in zone_arrays)
        with AtomicOutput(path, inputs) as f:
            np.array([elems.shape[0], 3], dtype="<i4").tofile(f)
            np.ascontiguousarray(elems, dtype=EL_DT).tofile(f)
            np.array([nzones], dtype="<i4").tofile(f)
            for z in zone_arrays:
                np.ascontiguousarray(z, dtype=ZONE_DT).tofile(f)
            np.array([curves.shape[0]], dtype="<i4").tofile(f)
            np.ascontiguousarray(curves, dtype=CURVE_DT).tofile(f)
        return

    glb_nelv = comm.allreduce(int(elems.shape[0]), op=MPI.SUM)
    nzones = comm.allreduce(sum(int(z.shape[0]) for z in zone_arrays), op=MPI.SUM)
    ncurves = comm.allreduce(int(curves.shape[0]), op=MPI.SUM)
    root = comm.Get_rank() == 0
    with AtomicOutput(path, inputs, comm=comm) as fh:
        if root:
            fh.Write_at(0, np.array([glb_nelv, 3], dtype="<i4"))
        offset = _write_block_collective(fh, HEADER_BYTES, elems, comm)
        if root:
            fh.Write_at(offset, np.array([nzones], dtype="<i4"))
        offset += COUNT_BYTES
        for z in zone_arrays:
            offset = _write_block_collective(fh, offset, z, comm)
        if root:
            fh.Write_at(offset, np.array([ncurves], dtype="<i4"))
        offset += COUNT_BYTES
        _write_block_collective(fh, offset, curves, comm)


def validate_zones(nelv, zones, path=""):
    """
    Check the zone records of a mesh and refuse malformed ones.

    Checks the zone type, the element and facet ranges, the periodic partner
    ranges and the label range of labelled zones. A violation raises instead
    of being silently repaired.

    Parameters
    ----------
    nelv : int
        Global number of elements in the mesh.
    zones : ndarray
        Zone records of dtype :data:`ZONE_DT`.
    path : str, optional
        File name used in error messages.

    Raises
    ------
    NmshFormatError
        If any record is malformed.
    """
    where = f" in {path}" if path else ""
    if zones.size == 0:
        return
    t = zones["t"]
    if ((t < 1) | (t > 7)).any():
        raise NmshFormatError(
            f"Zone record with implausible type (valid: 1..7){where}, "
            "corrupt or mis-framed zone section?"
        )
    bad = (zones["e"] < 1) | (zones["e"] > nelv) | (zones["f"] < 1) | (zones["f"] > 6)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise NmshFormatError(
            f"Zone record {i + 1} references element {int(zones['e'][i])} facet "
            f"{int(zones['f'][i])}, outside [1,{nelv}]x[1,6]{where}"
        )
    z5 = zones[t == ZONE_PERIODIC]
    if z5.size:
        bad = (z5["p_e"] < 1) | (z5["p_e"] > nelv) | (z5["p_f"] < 1) | (z5["p_f"] > 6)
        if bad.any():
            raise NmshFormatError(
                f"Periodic zone record references partner element/facet out of range{where}"
            )
    z7 = zones[t == ZONE_LABELLED]
    if z7.size:
        lbl = z7["p_f"]
        if ((lbl < 1) | (lbl > MAX_ZONE_LABELS)).any():
            raise NmshFormatError(
                f"Labelled zone with label outside [1,{MAX_ZONE_LABELS}]{where}"
            )


def validate_curves(nelv, curves, path=""):
    """
    Check the curve records of a mesh and refuse malformed ones.

    Parameters
    ----------
    nelv : int
        Global number of elements in the mesh.
    curves : ndarray
        Curve records of dtype :data:`CURVE_DT`.
    path : str, optional
        File name used in error messages.

    Raises
    ------
    NmshFormatError
        If a record references an element out of range or an unknown curve type.
    """
    where = f" in {path}" if path else ""
    if curves.size == 0:
        return
    if ((curves["e"] < 1) | (curves["e"] > nelv)).any():
        raise NmshFormatError(f"Curve record references element outside [1,{nelv}]{where}")
    ct = curves["type"]
    if ((ct != 0) & (ct != 3) & (ct != 4)).any():
        raise NmshFormatError(
            f"Curve record with unknown edge type (valid: 0, 3 = circle, 4 = midside){where}"
        )
