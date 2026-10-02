"""
Conversion of NEKTON ``.re2`` meshes to the Neko ``.nmsh`` format.

This reproduces Neko's ``rea2nbin`` tool byte for byte: the same first
appearance point numbering, the same two-pass boundary condition
classification and the same fixed three-sweep periodic id merge.
"""

import os

import numpy as np
from mpi4py import MPI

from ..monitoring.logger import Logger
from ..comm.utils import require_single_rank
from ..datatypes.nmsh import NmshMesh
from ..datatypes.re2 import Re2Mesh
from ..datatypes.corner_mesh import deduplicate_points
from ..io.nmsh import (
    EL_DT,
    ZONE_DT,
    CURVE_DT,
    FACE_VERTICES,
    MAX_ZONE_LABELS,
    ZONE_PERIODIC,
    ZONE_LABELLED,
    NmshFormatError,
)
from ..io.re2 import RE2_TO_NEKO_FACET, bc_type_str
from .registry import Conversion, Option, register

__all__ = [
    "re2_to_nmsh",
    "classify_boundary_conditions",
    "merge_periodic",
    "aggregate_curves",
    "default_periodic_tolerance",
]

#: Internal slot of each named NEKTON boundary condition type. Named types
#: sharing a slot get the same label.
NAMED_SLOT = {
    "W": 1,
    "v": 2,
    "V": 2,
    "O": 3,
    "o": 3,
    "SYM": 4,
    "sym": 4,
    "ON": 5,
    "on": 5,
    "s": 6,
    "sl": 6,
    "sh": 6,
    "shl": 6,
    "S": 6,
    "SL": 6,
    "SH": 6,
    "SHL": 6,
}
#: Boundary condition types that carry a user label in the fifth parameter.
MSH_TYPES = ("MSH", "msh", "EXO", "exo")


def default_periodic_tolerance():
    """
    Tolerance for matching periodic corners.

    Reads the ``NEKO_PERIODIC_TOL`` environment variable, as Neko does,
    and defaults to ``1e-7``.

    Returns
    -------
    float
        The tolerance.
    """
    s = os.environ.get("NEKO_PERIODIC_TOL", "")
    if not s:
        return 1e-7
    try:
        tol = float(s)
    except ValueError as ex:
        raise ValueError(f"Invalid NEKO_PERIODIC_TOL value: {s}") from ex
    if tol <= 0.0:
        raise ValueError(f"Invalid NEKO_PERIODIC_TOL value: {s}")
    return tol


def classify_boundary_conditions(nelv, bcs):
    """
    Classify ``.re2`` boundary condition records the way Neko's reader does.

    User labelled types (``MSH``/``EXO``) come first and keep their label.
    Named types (wall, inlet, outlet, ...) are assigned labels after the
    highest user label, in order of first appearance. Periodic records are
    collected as facet pairs.

    Parameters
    ----------
    nelv : int
        Number of elements.
    bcs : ndarray
        Boundary condition records as read by
        :func:`pysemtools.io.re2.read_re2`.

    Returns
    -------
    zone_e : ndarray
        Element of each labelled facet, in emission order.
    zone_f : ndarray
        Neko facet number of each labelled facet.
    zone_label : ndarray
        Label of each labelled facet.
    pairs : list of tuple
        Periodic pairs ``(e, facet, partner_e, partner_facet)`` in file order.
    """
    types = [bc_type_str(t) for t in bcs["t"]]
    e = bcs["e"].astype(np.int64)
    fc = bcs["f"].astype(np.int64)
    d1 = bcs["d"][:, 0].astype(np.float64)
    d2 = bcs["d"][:, 1].astype(np.float64)
    d5 = bcs["d"][:, 4].astype(np.float64)

    z_e, z_f, z_lbl = [], [], []
    pairs = []

    def add_zone(el, facet, lbl):
        if lbl < 1 or lbl > MAX_ZONE_LABELS:
            raise NmshFormatError(f"Boundary label out of [1,{MAX_ZONE_LABELS}]: {lbl}")
        z_e.append(el)
        z_f.append(facet)
        z_lbl.append(lbl)

    user_off = 0
    for i, t in enumerate(types):
        if t in MSH_TYPES:
            user_off = max(user_off, int(d5[i]))
    for i, t in enumerate(types):
        if t in MSH_TYPES:
            add_zone(int(e[i]), int(RE2_TO_NEKO_FACET[fc[i] - 1]), int(d5[i]))

    named_map = {}

    def named_label(slot):
        if slot not in named_map:
            named_map[slot] = len(named_map) + 1
        return user_off + named_map[slot]

    for i, t in enumerate(types):
        if t in MSH_TYPES or t in ("E", "e") or t == "":
            continue
        if t in NAMED_SLOT:
            add_zone(int(e[i]), int(RE2_TO_NEKO_FACET[fc[i] - 1]), named_label(NAMED_SLOT[t]))
        elif t == "P":
            pe, pf = int(d1[i]), int(d2[i])
            if pe < 1 or pe > nelv or pf < 1 or pf > 6:
                raise NmshFormatError(
                    "Periodic BC references out-of-range partner element/face"
                )
            pairs.append(
                (int(e[i]), int(RE2_TO_NEKO_FACET[fc[i] - 1]), pe, int(RE2_TO_NEKO_FACET[pf - 1]))
            )
    return (
        np.array(z_e, dtype=np.int64),
        np.array(z_f, dtype=np.int64),
        np.array(z_lbl, dtype=np.int64),
        pairs,
    )


def merge_periodic(pid, vid, coords, pairs, tol):
    """
    Merge the point ids of periodic facet pairs in place.

    This follows Neko's ``mesh_create_periodic_ids`` line for line: three
    sweeps over the facet pairs, each setting ``pid = min(pid_i, pid_j)`` in
    place per matching corner. The fixed sweep count and the sequential
    in-place minimum are what make the result byte identical to Neko's.

    Parameters
    ----------
    pid : ndarray
        Merged 0-based point index of each point, shape (npts,), modified in
        place.
    vid : ndarray
        0-based point index of the element corners, shape (nelv, 8), as
        returned by :func:`pysemtools.datatypes.corner_mesh.deduplicate_points`.
    coords : ndarray
        Coordinates of each point, shape (npts, 3).
    pairs : list of tuple
        Periodic pairs from :func:`classify_boundary_conditions`.
    tol : float
        Corner matching tolerance.
    """
    for _ in range(3):
        for el, f, pe, pf in pairs:
            si = FACE_VERTICES[f - 1]
            sj = FACE_VERTICES[pf - 1]
            ii = vid[el - 1, si]
            jj = vid[pe - 1, sj]
            a = coords[ii]
            b = coords[jj]
            shift = (a - b).mean(axis=0)
            d = np.linalg.norm(a[:, None, :] - b[None, :, :] - shift, axis=2)
            for k in range(4):
                hits = np.flatnonzero(d[k] < tol)
                if hits.size != 1:
                    raise NmshFormatError(
                        f"Periodic facet corner has {hits.size} matches (expected 1); "
                        "malformed periodic pairing"
                    )
                j = int(hits[0])
                m = min(pid[ii[k]], pid[jj[j]])
                pid[ii[k]] = m
                pid[jj[j]] = m


def aggregate_curves(curves):
    """
    Convert ``.re2`` curve records to ``.nmsh`` curve records.

    One record is produced per curved element, in ascending element order,
    with the edge slots filled per record (``C`` becomes type 3, ``m`` type
    4). A single unsupported type makes Neko treat the whole mesh as
    non-curved, which is reproduced here.

    Parameters
    ----------
    curves : ndarray
        Curve records as read by :func:`pysemtools.io.re2.read_re2`.

    Returns
    -------
    curves_out : ndarray
        Records of dtype :data:`pysemtools.io.nmsh.CURVE_DT`.
    skipped : bool
        True if an unsupported type was found and the curves were dropped.
    """
    if curves.size == 0:
        return np.empty(0, dtype=CURVE_DT), False
    first = np.array([t[:1] for t in curves["t"]])
    ctype = np.zeros(curves.size, dtype=np.int32)
    ctype[first == b"C"] = 3
    ctype[first == b"m"] = 4
    if (ctype == 0).any():
        return np.empty(0, dtype=CURVE_DT), True
    el = curves["e"].astype(np.int64)
    edge = curves["edge"].astype(np.int64)
    uniq = np.unique(el)
    out = np.zeros(uniq.size, dtype=CURVE_DT)
    out["e"] = uniq.astype(np.int32)
    row = np.searchsorted(uniq, el)
    out["data"][row, edge - 1, :] = curves["d"].astype(np.float64)
    out["type"][row, edge - 1] = ctype
    return out, False


def re2_to_nmsh(re2, nmsh_fname=None, periodic_tol=None, comm=None):
    """
    Convert a NEKTON ``.re2`` mesh to a Neko ``.nmsh`` mesh.

    The point numbering follows the order of first appearance in the file
    and the periodic merge is sequential, so the conversion works on the
    whole mesh and runs on a single rank.

    Parameters
    ----------
    re2 : str or Re2Mesh
        Input ``.re2`` file, or a replicated :class:`pysemtools.datatypes.re2.Re2Mesh`.
    nmsh_fname : str, optional
        Output ``.nmsh`` file. Default replaces the ``.re2`` extension of an
        input file; nothing is written when None is given with an ``Re2Mesh``.
    periodic_tol : float, optional
        Tolerance for matching periodic corners. Default is read from the
        ``NEKO_PERIODIC_TOL`` environment variable or ``1e-7``.
    comm : MPI.Comm, optional
        Communicator used for logging, which must have a single rank.
        Default is ``MPI.COMM_WORLD``.

    Returns
    -------
    NmshMesh
        The converted mesh, which has also been written to ``nmsh_fname``.

    Examples
    --------
    >>> from pysemtools.convert import re2_to_nmsh
    >>> nmsh = re2_to_nmsh("hemi.re2", "hemi.nmsh")
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    require_single_rank(comm, "The re2 -> nmsh conversion")
    log = Logger(comm=comm, module_name="re2_to_nmsh")
    if periodic_tol is None:
        periodic_tol = default_periodic_tolerance()

    log.tic()
    if isinstance(re2, str):
        if nmsh_fname is None:
            base, ext = os.path.splitext(re2)
            nmsh_fname = (base if ext == ".re2" else re2) + ".nmsh"
        re2 = Re2Mesh.from_file(re2)
    elif re2.is_distributed:
        re2 = re2.gather()
    nelv = re2.nelv
    xyz = re2.corner_coordinates
    log.write("info", f"{nelv} hex elements (format {re2.version})")

    log.write("info", "De-duplicating points")
    vid, nuniq = deduplicate_points(xyz)
    log.write("info", f"{nuniq} unique points")
    if nuniq > np.iinfo(np.int32).max:
        raise NmshFormatError(
            "More than 2**31 unique points, the .nmsh format stores 32-bit point ids"
        )
    coords = np.empty((nuniq, 3), dtype=np.float64)
    coords[vid.reshape(-1)] = xyz.reshape(-1, 3)

    log.write("info", "Classifying boundary conditions and merging periodic points")
    z_e, z_f, z_lbl, pairs = classify_boundary_conditions(nelv, re2.bcs)
    pid = np.arange(nuniq, dtype=np.int64)
    if pairs:
        merge_periodic(pid, vid, coords, pairs, periodic_tol)
    curves_out, curve_skip = aggregate_curves(re2.curves)
    if curve_skip:
        log.write(
            "warning",
            "Unsupported curve type (s/e/other); mesh treated as non-curved, as Neko does",
        )

    # The .nmsh format stores 1-based ids, the numbering above is 0-based
    elems = np.empty(nelv, dtype=EL_DT)
    elems["id"] = np.arange(1, nelv + 1, dtype=np.int32)
    elems["v"]["idx"] = (vid + 1).astype(np.int32)
    elems["v"]["xyz"] = xyz

    zp = np.zeros(len(pairs), dtype=ZONE_DT)
    for i, (el, f, pe, pf) in enumerate(pairs):
        zp["e"][i], zp["f"][i] = el, f
        zp["p_e"][i], zp["p_f"][i] = pe, pf
        zp["g"][i] = pid[vid[el - 1, FACE_VERTICES[f - 1]]] + 1
    zp["t"] = ZONE_PERIODIC

    zl = np.zeros(z_e.size, dtype=ZONE_DT)
    order = np.argsort(z_lbl, kind="stable") if z_e.size else []
    zl["e"] = z_e[order]
    zl["f"] = z_f[order]
    zl["p_f"] = z_lbl[order]
    zl["t"] = ZONE_LABELLED

    nmsh = NmshMesh(elems, np.concatenate([zp, zl]), curves_out)
    if nmsh_fname is not None:
        nmsh.write(nmsh_fname)
    log.write(
        "info",
        f"{len(pairs)} periodic + {z_e.size} labelled boundary facets, "
        f"{curves_out.shape[0]} curved elements",
    )
    log.toc()
    return nmsh


register(
    Conversion(
        "re2",
        "nmsh",
        re2_to_nmsh,
        "convert a NEKTON .re2 mesh to Neko .nmsh, like Neko's rea2nbin",
        parallel=False,
        options=(
            Option(
                "periodic_tol",
                float,
                "tolerance for matching periodic corners",
                default="NEKO_PERIODIC_TOL or 1e-7",
            ),
        ),
    )
)
