"""
Byte layout and parallel reader and writer of NEKTON ``.re2`` mesh files.

A ``.re2`` file holds an 80 character header, a float32 endianness tag, the
element records (a group id and the coordinates of the eight corners), the
curved edge records and the boundary condition records, the last two
preceded by their counts. Versions ``#v001`` (single precision) to
``#v004`` are read; ``#v002`` is written. Only three-dimensional meshes are
supported.

As for ``.nmsh`` files, reading and writing are collective when a
communicator is given: the elements are distributed linearly over the
ranks and the curve and boundary condition records are handed to the rank
that owns their element.
"""

import os

import numpy as np
from mpi4py import MPI

from ...comm.router import Router, record_datatype
from ...comm.distribution import linear_distribution, linear_owner
from ..utils import AtomicOutput

__all__ = [
    "RE2_EL_DT",
    "RE2_CURVE_DT",
    "RE2_BC_DT",
    "RE2_TO_NEKO_FACET",
    "Re2Data",
    "Re2FormatError",
    "read_re2",
    "write_re2",
    "validate_re2_records",
    "bc_type_str",
]

#: Element record of a ``.re2`` (double precision): group id and the x, y, z
#: coordinates of the eight corners.
RE2_EL_DT = np.dtype(
    [("rg", "<f8"), ("x", "<f8", (8,)), ("y", "<f8", (8,)), ("z", "<f8", (8,))]
)
#: Curve record of a ``.re2`` (double precision): element, edge, five
#: parameters and the curve type.
RE2_CURVE_DT = np.dtype([("e", "<f8"), ("edge", "<f8"), ("d", "<f8", (5,)), ("t", "S8")])
#: Boundary condition record of a ``.re2`` (double precision): element, face,
#: five parameters and the type.
RE2_BC_DT = np.dtype([("e", "<f8"), ("f", "<f8"), ("d", "<f8", (5,)), ("t", "S8")])

# Single precision layouts of version #v001, upcast on reading.
_RE2_EL_DT_SP = np.dtype(
    [("rg", "<f4"), ("x", "<f4", (8,)), ("y", "<f4", (8,)), ("z", "<f4", (8,))]
)
_RE2_CURVE_DT_SP = np.dtype([("e", "<i4"), ("edge", "<i4"), ("d", "<f4", (5,)), ("t", "S4")])
_RE2_BC_DT_SP = np.dtype([("e", "<i4"), ("f", "<i4"), ("d", "<f4", (5,)), ("t", "S4")])

RE2_ENDIAN_TEST = 6.54321
HEADER_BYTES = 80
TAG_BYTES = 4
WRITE_VERSION = "#v002"

#: Map from a NEKTON ``.re2`` face number (1-based index) to the Neko facet
#: number (1-based value).
RE2_TO_NEKO_FACET = np.array([3, 2, 4, 1, 5, 6], dtype=np.int64)


class Re2FormatError(ValueError):
    """Raised when a ``.re2`` file is malformed or not supported."""


class Re2Data:
    """
    The raw record arrays of a ``.re2`` file as returned by :func:`read_re2`.

    Records are always held in the double precision layouts :data:`RE2_EL_DT`,
    :data:`RE2_CURVE_DT` and :data:`RE2_BC_DT`; single precision files are
    upcast on reading.

    Parameters
    ----------
    elems : ndarray
        Element records held by this rank.
    curves : ndarray
        Curve records held by this rank.
    bcs : ndarray
        Boundary condition records held by this rank.
    version : str
        Format version of the file, e.g. ``#v002``.
    glb_nelv : int
        Global number of elements in the file.
    offset_el : int
        Record position of the first local element in the file.
    trailing : int, optional
        Bytes found past the boundary condition section. Default is 0.
    curve_index, bc_index : ndarray, optional
        Record positions in the file of the local curve and boundary
        condition records. Default is the local positions.
    """

    def __init__(
        self,
        elems,
        curves,
        bcs,
        version,
        glb_nelv,
        offset_el,
        trailing=0,
        curve_index=None,
        bc_index=None,
    ):
        self.elems = elems
        self.curves = curves
        self.bcs = bcs
        self.version = version
        self.glb_nelv = int(glb_nelv)
        self.offset_el = int(offset_el)
        self.trailing = int(trailing)
        if curve_index is None:
            curve_index = np.arange(curves.shape[0], dtype=np.int64)
        if bc_index is None:
            bc_index = np.arange(bcs.shape[0], dtype=np.int64)
        self.curve_index = curve_index
        self.bc_index = bc_index

    @property
    def nelv(self):
        """Number of elements held by this rank."""
        return int(self.elems.shape[0])

    @property
    def xyz(self):
        """Corner coordinates of the local elements, shape (nelv, 8, 3)."""
        return np.stack([self.elems["x"], self.elems["y"], self.elems["z"]], axis=2)


def _parse_header(hdr, path):
    """Version, dimension and element count from the 80 byte header."""
    ver = hdr[:5].decode("latin-1")
    try:
        if ver == "#v004":
            ndim = int(hdr[21:24])
            nelv = int(hdr[24:40])
        elif ver in ("#v001", "#v002", "#v003"):
            ndim = int(hdr[14:17])
            nelv = int(hdr[17:26])
        else:
            raise Re2FormatError(f"Unknown re2 version {ver!r} in {path}")
    except ValueError as ex:
        raise Re2FormatError(f"Cannot parse re2 header of {path}") from ex
    if ndim != 3:
        raise Re2FormatError("Only 3D (hexahedral) meshes are supported")
    if nelv < 1:
        raise Re2FormatError(f"{path}: non-positive element count ({nelv})")
    return ver, nelv


def _layouts(version):
    """Record dtypes and count dtype of a version."""
    if version == "#v001":
        return _RE2_EL_DT_SP, _RE2_CURVE_DT_SP, _RE2_BC_DT_SP, np.dtype("<i4")
    return RE2_EL_DT, RE2_CURVE_DT, RE2_BC_DT, np.dtype("<f8")


def _upcast(records, dtype):
    """Convert records to the double precision layout (a no-op if already there)."""
    if records.dtype == dtype:
        return records
    return records.astype(dtype)


def _check_tag(tag):
    if tag.size != 1 or abs(float(tag[0]) - RE2_ENDIAN_TEST) > 1e-4:
        raise Re2FormatError("Byte-swapped or corrupt re2 (endian tag)")


def _read_re2_serial(path, chunk):
    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
        if len(hdr) != HEADER_BYTES:
            raise Re2FormatError(f"{path} is not a .re2 file (short header)")
        ver, nelv = _parse_header(hdr, path)
        el_dt, curve_dt, bc_dt, count_dt = _layouts(ver)
        _check_tag(np.fromfile(f, dtype="<f4", count=1))

        elems = np.empty(nelv, dtype=RE2_EL_DT)
        done = 0
        while done < nelv:
            n = min(chunk, nelv - done)
            rec = np.fromfile(f, dtype=el_dt, count=n)
            if rec.size != n:
                raise Re2FormatError(
                    "Truncated or corrupt .re2 file "
                    f"(element section, record {done + rec.size} of {nelv})"
                )
            elems[done : done + n] = _upcast(rec, RE2_EL_DT)
            done += n

        ncurve = _read_count(f, count_dt, "curve")
        curves = np.fromfile(f, dtype=curve_dt, count=ncurve)
        if curves.size != ncurve:
            raise Re2FormatError("Truncated or corrupt .re2 file (curve section)")
        nbc = _read_count(f, count_dt, "boundary-condition")
        bcs = np.fromfile(f, dtype=bc_dt, count=nbc)
        if bcs.size != nbc:
            raise Re2FormatError("Truncated or corrupt .re2 file (BC section)")
        here = f.tell()
        f.seek(0, os.SEEK_END)
        trailing = f.tell() - here
    return Re2Data(
        elems, _upcast(curves, RE2_CURVE_DT), _upcast(bcs, RE2_BC_DT), ver, nelv, 0, trailing
    )


def _read_count(f, count_dt, what):
    a = np.fromfile(f, dtype=count_dt, count=1)
    if a.size != 1:
        raise Re2FormatError(f"Truncated or corrupt .re2 file (missing {what} count)")
    n = int(a[0])
    if n < 0:
        raise Re2FormatError(f"Negative {what} count in .re2")
    return n


def _route_to_owner(comm, records, owner):
    """
    Send every record to the rank in ``owner`` and return what this rank receives.

    The received records are ordered by source rank and, within a source, in
    the order they were sent, so a companion array routed with the same
    ``owner`` stays aligned.
    """
    rt = Router(comm)
    destinations = list(range(comm.Get_size()))
    _, chunks = rt.all_to_all(
        destination=destinations,
        data=[records[owner == r] for r in destinations],
        dtype=records.dtype,
    )
    if len(chunks) == 0:
        return np.empty(0, dtype=records.dtype)
    return np.concatenate(chunks)


def _read_section_distributed(fh, offset, count, dtype, out_dtype, glb_nelv, comm, what):
    """Read a record section in linear blocks and route each record to its element's owner."""
    n, start = linear_distribution(count, comm)
    block = np.empty(n, dtype=dtype)
    rec_t = record_datatype(dtype)
    fh.Read_at_all(offset + start * dtype.itemsize, [block.view(np.uint8), n, rec_t])
    rec_t.Free()
    block = _upcast(block, out_dtype)
    e = block["e"].astype(np.int64)
    if ((e < 1) | (e > glb_nelv)).any():
        raise Re2FormatError(f"{what} record references element outside [1,{glb_nelv}]")
    owner = linear_owner(e - 1, glb_nelv, comm.Get_size())
    index = np.arange(start, start + n, dtype=np.int64)
    records = _route_to_owner(comm, block, owner)
    index = _route_to_owner(comm, index, owner)
    return records, index


def _read_re2_distributed(path, comm):
    fh = MPI.File.Open(comm, path, MPI.MODE_RDONLY)
    try:
        file_size = fh.Get_size()
        if file_size < HEADER_BYTES + TAG_BYTES:
            fh.Read_at_all(0, np.zeros(0, dtype=np.uint8))
            raise Re2FormatError(f"{path} is not a .re2 file (short header)")
        hdr = np.zeros(HEADER_BYTES, dtype=np.uint8)
        fh.Read_at_all(0, hdr)
        ver, glb_nelv = _parse_header(hdr.tobytes(), path)
        el_dt, curve_dt, bc_dt, count_dt = _layouts(ver)
        tag = np.zeros(1, dtype="<f4")
        fh.Read_at_all(HEADER_BYTES, tag)
        _check_tag(tag)

        offset = HEADER_BYTES + TAG_BYTES
        el_end = offset + glb_nelv * el_dt.itemsize
        if file_size < el_end + count_dt.itemsize:
            raise Re2FormatError(f"{path}: truncated element section")
        nelv, offset_el = linear_distribution(glb_nelv, comm)
        block = np.empty(nelv, dtype=el_dt)
        rec_t = record_datatype(el_dt)
        fh.Read_at_all(offset + offset_el * el_dt.itemsize, [block.view(np.uint8), nelv, rec_t])
        rec_t.Free()
        elems = _upcast(block, RE2_EL_DT)
        offset = el_end

        count = np.zeros(1, dtype=count_dt)
        fh.Read_at_all(offset, count)
        ncurve = int(count[0])
        offset += count_dt.itemsize
        if ncurve < 0 or file_size < offset + ncurve * curve_dt.itemsize + count_dt.itemsize:
            raise Re2FormatError(f"{path}: truncated or corrupt curve section")
        curves, curve_index = _read_section_distributed(
            fh, offset, ncurve, curve_dt, RE2_CURVE_DT, glb_nelv, comm, "Curve"
        )
        offset += ncurve * curve_dt.itemsize

        fh.Read_at_all(offset, count)
        nbc = int(count[0])
        offset += count_dt.itemsize
        if nbc < 0 or file_size < offset + nbc * bc_dt.itemsize:
            raise Re2FormatError(f"{path}: truncated or corrupt boundary condition section")
        bcs, bc_index = _read_section_distributed(
            fh, offset, nbc, bc_dt, RE2_BC_DT, glb_nelv, comm, "BC"
        )
        offset += nbc * bc_dt.itemsize
        trailing = file_size - offset
    finally:
        fh.Close()
    return Re2Data(
        elems, curves, bcs, ver, glb_nelv, offset_el, trailing, curve_index, bc_index
    )


def read_re2(path, comm=None, chunk=2**21):
    """
    Read a NEKTON ``.re2`` file (versions ``#v001`` to ``#v004``, little endian, 3D).

    Parameters
    ----------
    path : str
        Path of the ``.re2`` file.
    comm : MPI.Comm, optional
        If given, the elements are distributed linearly over the ranks and
        read with collective MPI-IO, and the curve and boundary condition
        records are handed to the rank owning their element. If None, the
        calling process reads the whole file.
    chunk : int, optional
        Number of elements read per chunk in a serial read. Default is 2**21.

    Returns
    -------
    Re2Data
        The records, upcast to double precision.

    Raises
    ------
    Re2FormatError
        If the file is not a supported ``.re2`` or a section is malformed.

    Examples
    --------
    >>> from pysemtools.io.re2 import read_re2
    >>> re2 = read_re2("hemi.re2")
    >>> re2.nelv, re2.xyz.shape
    (2042, (2042, 8, 3))
    """
    if comm is None:
        data = _read_re2_serial(path, chunk)
    else:
        data = _read_re2_distributed(path, comm)
    validate_re2_records(data.glb_nelv, data.curves, data.bcs)
    return data


def validate_re2_records(nelv, curves, bcs):
    """
    Check the ranges of curve and boundary condition records.

    Parameters
    ----------
    nelv : int
        Global number of elements.
    curves : ndarray
        Curve records of dtype :data:`RE2_CURVE_DT`.
    bcs : ndarray
        Boundary condition records of dtype :data:`RE2_BC_DT`.

    Raises
    ------
    Re2FormatError
        If a record references an element, edge or face out of range.
    """
    if curves.size:
        ce = curves["e"].astype(np.int64)
        cz = curves["edge"].astype(np.int64)
        if ((ce < 1) | (ce > nelv)).any():
            raise Re2FormatError("Curve record references element out of range")
        if ((cz < 1) | (cz > 12)).any():
            raise Re2FormatError("Curve record edge index out of [1,12]")
    if bcs.size:
        be = bcs["e"].astype(np.int64)
        bf = bcs["f"].astype(np.int64)
        if ((be < 1) | (be > nelv)).any():
            raise Re2FormatError("BC record references element out of range")
        if ((bf < 1) | (bf > 6)).any():
            raise Re2FormatError("BC record face out of [1,6]")


def _header(glb_nelv):
    return f"{WRITE_VERSION}{glb_nelv:9d}{3:3d}{glb_nelv:9d} this is the hdr".ljust(
        HEADER_BYTES
    ).encode("ascii")


def _write_block_collective(fh, offset, records, comm):
    """Write the records of every rank contiguously at ``offset``, ordered by rank."""
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


def write_re2(path, elems, curves, bcs, inputs=(), comm=None):
    """
    Write a ``.re2`` file (version ``#v002``, double precision) atomically.

    Parameters
    ----------
    path : str
        Destination path.
    elems : ndarray
        Element records of dtype :data:`RE2_EL_DT`. With a communicator these
        are the local elements, written as the contiguous block of the rank.
    curves : ndarray
        Curve records of dtype :data:`RE2_CURVE_DT`.
    bcs : ndarray
        Boundary condition records of dtype :data:`RE2_BC_DT`.
    inputs : sequence of str, optional
        Input files that must not be overwritten.
    comm : MPI.Comm, optional
        Communicator for a collective write. Default is a serial write by
        the calling process.
    """
    elems = np.ascontiguousarray(elems, dtype=RE2_EL_DT)
    curves = np.ascontiguousarray(curves, dtype=RE2_CURVE_DT)
    bcs = np.ascontiguousarray(bcs, dtype=RE2_BC_DT)
    tag = np.array([RE2_ENDIAN_TEST], dtype="<f4")
    if comm is None:
        with AtomicOutput(path, inputs) as f:
            f.write(_header(elems.shape[0]))
            tag.tofile(f)
            elems.tofile(f)
            np.array([curves.shape[0]], dtype="<f8").tofile(f)
            curves.tofile(f)
            np.array([bcs.shape[0]], dtype="<f8").tofile(f)
            bcs.tofile(f)
        return

    glb_nelv = comm.allreduce(int(elems.shape[0]), op=MPI.SUM)
    ncurves = comm.allreduce(int(curves.shape[0]), op=MPI.SUM)
    nbcs = comm.allreduce(int(bcs.shape[0]), op=MPI.SUM)
    root = comm.Get_rank() == 0
    with AtomicOutput(path, inputs, comm=comm) as fh:
        if root:
            fh.Write_at(0, np.frombuffer(_header(glb_nelv), dtype=np.uint8))
            fh.Write_at(HEADER_BYTES, tag)
        offset = _write_block_collective(fh, HEADER_BYTES + TAG_BYTES, elems, comm)
        if root:
            fh.Write_at(offset, np.array([ncurves], dtype="<f8"))
        offset = _write_block_collective(fh, offset + 8, curves, comm)
        if root:
            fh.Write_at(offset, np.array([nbcs], dtype="<f8"))
        _write_block_collective(fh, offset + 8, bcs, comm)


def bc_type_str(raw):
    """
    Decode a boundary condition or curve type field as Neko sees it.

    Parameters
    ----------
    raw : bytes
        The raw field.

    Returns
    -------
    str
        The type with NUL padding and surrounding blanks stripped.
    """
    return raw.decode("latin-1").replace("\x00", " ").strip()
