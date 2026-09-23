"""
Byte layout and reader of NEKTON ``.re2`` mesh files.

The reader supports the little-endian versions ``#v001`` to ``#v004`` of
three-dimensional meshes. The file is read on the calling process; converting
a mesh to the Neko format needs the whole mesh for the point numbering, see
:func:`pysemtools.nmsh.convert.re2_to_nmsh`.
"""

import numpy as np

__all__ = ["Re2Data", "Re2FormatError", "RE2_TO_NEKO_FACET", "read_re2", "bc_type_str"]

# ``.re2`` records for double (version >= #v002) and single (#v001) precision.
RE2_EL_DT = {
    True: np.dtype(
        [("rg", "<f8"), ("x", "<f8", (8,)), ("y", "<f8", (8,)), ("z", "<f8", (8,))]
    ),
    False: np.dtype(
        [("rg", "<f4"), ("x", "<f4", (8,)), ("y", "<f4", (8,)), ("z", "<f4", (8,))]
    ),
}
RE2_CURVE_DT = {
    True: np.dtype([("e", "<f8"), ("edge", "<f8"), ("d", "<f8", (5,)), ("t", "S8")]),
    False: np.dtype([("e", "<i4"), ("edge", "<i4"), ("d", "<f4", (5,)), ("t", "S4")]),
}
RE2_BC_DT = {
    True: np.dtype([("e", "<f8"), ("f", "<f8"), ("d", "<f8", (5,)), ("t", "S8")]),
    False: np.dtype([("e", "<i4"), ("f", "<i4"), ("d", "<f4", (5,)), ("t", "S4")]),
}
RE2_ENDIAN_TEST = 6.54321

#: Map from a NEKTON ``.re2`` face number (1-based index) to the Neko facet
#: number (1-based value).
RE2_TO_NEKO_FACET = np.array([3, 2, 4, 1, 5, 6], dtype=np.int64)


class Re2FormatError(ValueError):
    """Raised when a ``.re2`` file is malformed or not supported."""


class Re2Data:
    """
    A NEKTON ``.re2`` mesh in memory (3D only).

    Parameters
    ----------
    nelv : int
        Number of elements.
    version : str
        Format version string, e.g. ``#v002``.
    xyz : ndarray
        Corner coordinates, shape (nelv, 8, 3), float64.
    curves : ndarray
        Curve records as read (dtype depends on the version).
    bcs : ndarray
        Boundary condition records as read (dtype depends on the version).
    """

    def __init__(self, nelv, version, xyz, curves, bcs):
        self.nelv = int(nelv)
        self.version = version
        self.xyz = xyz
        self.curves = curves
        self.bcs = bcs


def _re2_count(f, double_precision, what):
    a = np.fromfile(f, dtype="<f8" if double_precision else "<i4", count=1)
    if a.size != 1:
        raise Re2FormatError(f"Truncated or corrupt .re2 file (missing {what} count)")
    n = int(a[0])
    if n < 0:
        raise Re2FormatError(f"Negative {what} count in .re2")
    return n


def read_re2(path, chunk=1 << 21):
    """
    Read a NEKTON ``.re2`` file (versions ``#v001`` to ``#v004``, little endian, 3D).

    Parameters
    ----------
    path : str
        Path of the ``.re2`` file.
    chunk : int, optional
        Number of elements read per chunk. Default is 2**21.

    Returns
    -------
    Re2Data
        The mesh data with coordinates upcast to float64.

    Raises
    ------
    Re2FormatError
        If the file is not a supported ``.re2`` or a section is malformed.

    Examples
    --------
    >>> from pysemtools.io.re2 import read_re2
    >>> re2 = read_re2("hemi.re2")
    >>> re2.nelv, re2.xyz.shape
    (1000, (1000, 8, 3))
    """
    with open(path, "rb") as f:
        hdr = f.read(80)
        if len(hdr) != 80:
            raise Re2FormatError(f"{path} is not a .re2 file (short header)")
        ver = hdr[:5].decode("latin-1")
        try:
            if ver == "#v004":
                ndim = int(hdr[21:24])
                nelv = int(hdr[24:40])
            elif ver in ("#v001", "#v002", "#v003"):
                ndim = int(hdr[14:17])
                nelv = int(hdr[17:26])
            else:
                raise Re2FormatError(f"Unknown re2 version {ver!r}")
        except ValueError as ex:
            raise Re2FormatError(f"Cannot parse re2 header of {path}") from ex
        double_precision = ver != "#v001"
        endian = np.fromfile(f, dtype="<f4", count=1)
        if endian.size != 1 or abs(float(endian[0]) - RE2_ENDIAN_TEST) > 1e-4:
            raise Re2FormatError("Byte-swapped or corrupt re2 (endian tag)")
        if ndim != 3:
            raise Re2FormatError("Only 3D (hexahedral) meshes are supported")

        xyz = np.empty((nelv, 8, 3), dtype=np.float64)
        done = 0
        while done < nelv:
            n = min(chunk, nelv - done)
            rec = np.fromfile(f, dtype=RE2_EL_DT[double_precision], count=n)
            if rec.size != n:
                raise Re2FormatError(
                    "Truncated or corrupt .re2 file "
                    f"(element section, record {done + rec.size} of {nelv})"
                )
            xyz[done : done + n, :, 0] = rec["x"]
            xyz[done : done + n, :, 1] = rec["y"]
            xyz[done : done + n, :, 2] = rec["z"]
            done += n
        if not np.isfinite(xyz).all():
            raise Re2FormatError(f"Non-finite coordinate in {path}")

        ncurve = _re2_count(f, double_precision, "curve")
        curves = np.fromfile(f, dtype=RE2_CURVE_DT[double_precision], count=ncurve)
        if curves.size != ncurve:
            raise Re2FormatError("Truncated or corrupt .re2 file (curve section)")
        nbc = _re2_count(f, double_precision, "boundary-condition")
        bcs = np.fromfile(f, dtype=RE2_BC_DT[double_precision], count=nbc)
        if bcs.size != nbc:
            raise Re2FormatError("Truncated or corrupt .re2 file (BC section)")

    ce = curves["e"].astype(np.int64)
    cz = curves["edge"].astype(np.int64)
    if curves.size and ((ce < 1) | (ce > nelv)).any():
        raise Re2FormatError("Curve record references element out of range")
    if curves.size and ((cz < 1) | (cz > 12)).any():
        raise Re2FormatError("Curve record edge index out of [1,12]")
    be = bcs["e"].astype(np.int64)
    bf = bcs["f"].astype(np.int64)
    if bcs.size and ((be < 1) | (be > nelv)).any():
        raise Re2FormatError("BC record references element out of range")
    if bcs.size and ((bf < 1) | (bf > 6)).any():
        raise Re2FormatError("BC record face out of [1,6]")
    return Re2Data(nelv, ver, xyz, curves, bcs)


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
