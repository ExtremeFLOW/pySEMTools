"""
Topology of hexahedral meshes given by the corners of their elements.

This module fixes the canonical topology of the corner mesh data types,
:class:`pysemtools.datatypes.corner_mesh.CornerMesh` and its subclasses, and
provides the topological operations on it. The conventions are:

- **Corners**: corner ``c`` of an element sits at ``CORNER_IJK[c]`` in the
  reference cube.
- **Facets**: facet ``f`` is made of the corners ``FACET_CORNERS[f]``. The
  facets are, in order, the faces r = -1, r = +1, s = -1, s = +1, t = -1 and
  t = +1 of the reference cube.
- **Edges**: edge ``e`` joins the corners ``EDGE_CORNERS[e]``.
- **Element indices**: an element is referred to by its 0-based global
  index, its element id minus one.
- **Point ids**: integer labels of the corners. Two corners have the same id
  exactly when they are the same point; the ids carry no order and need not
  be dense. :func:`compress_ids` makes them dense.

These are the conventions of Nek5000 and Neko mesh files, so the data types
of both formats translate their records with little or no work. The
functions work on plain arrays and do not depend on a file format.
Everything is vectorised numpy and sort based, so the memory footprint
scales with the number of element corners.
"""

import numpy as np

from ..comm.router import Router

__all__ = [
    "CORNER_IJK",
    "FACET_CORNERS",
    "EDGE_CORNERS",
    "deduplicate_points",
    "periodic_min_merge",
    "periodic_replace_merge",
    "compress_ids",
    "face_multiplicity",
    "count_edges",
    "skin",
    "dual_graph",
]

#: Position of each of the eight corners of an element in the reference cube,
#: as (i, j, k) offsets in {0, 1} along (r, s, t). The first four corners go
#: counter-clockwise round the bottom face, starting at the origin, and the
#: last four repeat that on the top face.
CORNER_IJK = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.int64,
)

#: The four corners of each of the six facets. The facets are the faces
#: r = -1, r = +1, s = -1, s = +1, t = -1 and t = +1 of the reference cube.
#: The order of the corners within a facet matters wherever a facet carries
#: data per corner, such as the target point ids of a periodic facet.
FACET_CORNERS = np.array(
    [[0, 4, 7, 3], [1, 5, 6, 2], [0, 1, 5, 4], [3, 2, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]],
    dtype=np.int64,
)

#: The two corners of each of the twelve edges, in the edge numbering of
#: Nek5000 and Neko curve records: edges 0 to 3 go round the bottom face,
#: edges 4 to 7 round the top face and edges 8 to 11 are the vertical ones.
EDGE_CORNERS = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
    dtype=np.int64,
)


def deduplicate_points(xyz, comm=None):
    """
    Number the distinct corner coordinates.

    Coordinates are compared bit-exactly as float64 triples, which matches
    the point table of Neko's ``.re2`` reader. The numbering is independent
    of any file format: a format that stores point ids, like ``.nmsh``,
    applies its own offset and integer width when the records are written.

    Without a communicator the corners are numbered 0-based in order of
    first appearance. With a communicator every rank passes its local
    corners and the numbering is global: coinciding corners get the same id
    on every rank, the ids are dense from 0, and their order is unspecified.
    On a single rank both give the same result.

    Parameters
    ----------
    xyz : ndarray
        Corner coordinates, shape (nelv, 8, 3), float64.
    comm : MPI.Comm, optional
        Communicator over which the corners are distributed. Default is
        None, a numbering of the given corners alone. The call is collective
        when a communicator is given.

    Returns
    -------
    vid : ndarray
        Point id of every corner, shape (nelv, 8), int64.
    n_unique : int
        Number of distinct points, over all ranks when a communicator is
        given.

    Notes
    -----
    The corners are sorted as triples of 64-bit integers with a stable sort,
    so equal corners become adjacent and the first appearance leads each
    group. The transient memory cost is roughly 36 bytes per corner on top
    of the input. With a communicator every corner is first sent to an owner
    rank chosen by a hash of its 24 bytes, so coinciding corners from all
    ranks meet on the same rank. The owner numbers what it received, an
    exclusive scan of the counts makes the ids global, and the ids are sent
    back. This takes two all-to-all exchanges and memory proportional to the
    local corners.
    """
    # One row per corner, in element order. The contiguous layout is needed
    # for the integer view below.
    n8 = xyz.shape[0] * 8
    flat = np.ascontiguousarray(xyz, dtype=np.float64).reshape(n8, 3)

    # Reinterpret the three float64 of each row as three 64-bit integers,
    # without copying. Two corners are then equal exactly when their integers
    # are, and integers sort faster than floats and need no NaN handling.
    u = flat.view(np.uint64)

    if comm is None:
        vid, n_unique = _number_rows(u)
    else:
        vid, n_unique = _number_rows_distributed(u, comm)
    return vid.reshape(-1, 8), n_unique


def _number_rows(u):
    """
    Number the distinct rows of an (n, 3) uint64 array in order of first appearance.

    Returns the 0-based id of every row, int64, and the number of distinct rows.
    """
    # The comments follow a small example with four corners and three distinct
    # points, written as letters:
    #
    #   corner i:  0  1  2  3
    #   point:     B  A  B  C
    #
    # The expected result numbers the points in order of first appearance,
    # B = 0, A = 1, C = 2, so vid = [0, 1, 0, 2].

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
    n8 = u.shape[0]
    new = np.empty(n8, dtype=bool)
    if n8 == 0:
        return np.empty(0, dtype=np.int64), 0
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

    # Scatter the rank of every group back from sorted positions to corners.
    # Example: rank[grp] = [1, 0, 0, 2], written to corners [1, 0, 2, 3],
    # so vid = [0, 1, 0, 2].
    vid = np.empty(n8, dtype=np.int64)
    vid[order] = rank[grp]
    return vid, int(first.size)


def _hash_rows(u):
    """Mix the three 64-bit integers of every row into one well spread 64-bit hash."""
    with np.errstate(over="ignore"):
        h = u[:, 0] * np.uint64(0x9E3779B97F4A7C15)
        h ^= u[:, 1] + np.uint64(0x632BE59BD9B4E019) + (h << np.uint64(6)) + (h >> np.uint64(2))
        h ^= u[:, 2] + np.uint64(0x8CB92BA72F3D8DD7) + (h << np.uint64(6)) + (h >> np.uint64(2))
        # Finaliser of splitmix64, so that rows differing in only a few bits
        # still land on different ranks
        h ^= h >> np.uint64(30)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(27)
        h *= np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return h


def _number_rows_distributed(u, comm):
    """
    Number the distinct rows of (n, 3) uint64 arrays held by the ranks of a communicator.

    Returns the global 0-based id of every local row, int64, and the global
    number of distinct rows.
    """
    size = comm.Get_size()
    rt = Router(comm)
    ranks = list(range(size))

    # Every row goes to the owner rank given by its hash, so that equal rows
    # from all ranks meet on one rank. The rows are grouped by owner, and the
    # stable sort keeps their order within a group.
    owner = (_hash_rows(u) % np.uint64(size)).astype(np.int64)
    order = np.argsort(owner, kind="stable")
    bounds = np.concatenate(([0], np.cumsum(np.bincount(owner, minlength=size))))
    grouped = u[order]
    sources, received = rt.all_to_all(
        destination=ranks,
        data=[grouped[bounds[r] : bounds[r + 1]].ravel() for r in ranks],
        dtype=np.uint64,
    )
    del grouped

    # The owner numbers the rows it received, which the router returns ordered
    # by source rank. An exclusive scan of the counts turns the local numbers
    # into global ones.
    counts = [chunk.size // 3 for chunk in received]
    rows = (
        np.concatenate([chunk.reshape(-1, 3) for chunk in received])
        if received
        else np.empty((0, 3), dtype=np.uint64)
    )
    ids, n_local = _number_rows(rows)
    offset = comm.exscan(n_local)
    if comm.Get_rank() > 0:
        ids += offset
    n_unique = int(comm.allreduce(n_local))

    # Every source gets the ids of its rows back in the order it sent them.
    # They arrive ordered by owner rank, which is the order of the grouped
    # rows, so scattering them through the grouping permutation restores the
    # order of the local corners.
    split = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    _, returned = rt.all_to_all(
        destination=[int(s) for s in sources],
        data=[ids[split[i] : split[i + 1]] for i in range(len(counts))],
        dtype=np.int64,
    )
    vid = np.empty(u.shape[0], dtype=np.int64)
    vid[order] = np.concatenate(returned) if returned else np.empty(0, dtype=np.int64)
    return vid, n_unique


def periodic_min_merge(vidx, elem, facet, targets):
    """
    Merge periodic points by taking the minimum of the own and the target id.

    Every corner of a periodic facet takes ``min(own id, target id)``,
    applied over the whole set at once. This is the merge used by the
    partitioners to build a periodic dual graph.

    Parameters
    ----------
    vidx : ndarray
        Point ids, shape (nelv, 8), int64.
    elem : ndarray
        0-based element position of each periodic facet, shape (n,).
    facet : ndarray
        0-based facet index of each periodic facet, shape (n,).
    targets : ndarray
        Point ids that the four corners ``FACET_CORNERS[facet]`` of each
        periodic facet are identified with, shape (n, 4).

    Returns
    -------
    ndarray
        Merged point ids, shape (nelv, 8).
    """
    elem = np.asarray(elem, dtype=np.int64)
    if not elem.size:
        return vidx
    nelv = vidx.shape[0]
    uniq = np.unique(vidx)
    merge = uniq.copy()
    slots = FACET_CORNERS[np.asarray(facet, dtype=np.int64)]
    raw = vidx[elem[:, None], slots]
    ridx = np.searchsorted(uniq, raw.ravel())
    np.minimum.at(merge, ridx, np.asarray(targets, dtype=np.int64).ravel())
    return merge[np.searchsorted(uniq, vidx.ravel())].reshape(nelv, 8)


def periodic_replace_merge(vidx, elem, facet, targets):
    """
    Merge periodic points by replacing them with the target ids.

    Corner ``k`` of a periodic facet takes ``targets[k]`` directly, with the
    last facet winning when a point is named several times. With the targets
    of a ``.nmsh`` file this is what Neko's ``apply_periodic_facet`` does
    when reading the mesh, so it reproduces the sizes Neko reports.

    Parameters
    ----------
    vidx : ndarray
        Point ids, shape (nelv, 8), int64.
    elem : ndarray
        0-based element position of each periodic facet, shape (n,).
    facet : ndarray
        0-based facet index of each periodic facet, shape (n,).
    targets : ndarray
        Point ids that the four corners ``FACET_CORNERS[facet]`` of each
        periodic facet are identified with, shape (n, 4). The order of the
        facets decides which target wins.

    Returns
    -------
    ndarray
        Merged point ids, shape (nelv, 8).
    """
    elem = np.asarray(elem, dtype=np.int64)
    if not elem.size:
        return vidx
    nelv = vidx.shape[0]
    remap = {}
    for pos, f, tgt in zip(elem.tolist(), np.asarray(facet).tolist(), np.asarray(targets).tolist()):
        slots = FACET_CORNERS[f]
        for k in range(4):
            remap[int(vidx[pos, slots[k]])] = int(tgt[k])
    keys = np.fromiter(remap.keys(), dtype=np.int64, count=len(remap))
    vals = np.fromiter(remap.values(), dtype=np.int64, count=len(remap))
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]
    idx = np.searchsorted(keys, vidx.ravel())
    idx[idx >= keys.size] = 0
    hit = keys[idx] == vidx.ravel()
    out = vidx.ravel().copy()
    out[hit] = vals[idx[hit]]
    return out.reshape(nelv, 8)


def compress_ids(vidx):
    """
    Compress arbitrary point ids to dense 0-based ids.

    Parameters
    ----------
    vidx : ndarray
        Point ids, shape (nelv, 8).

    Returns
    -------
    ndarray
        Dense ids ``0..n-1`` in sorted-id order, same shape, int64.
    """
    _, cell = np.unique(vidx.ravel(), return_inverse=True)
    return cell.reshape(vidx.shape).astype(np.int64)


def _pack_faces(vidx):
    """Canonical 16-byte key per (element, facet): the sorted corner ids packed in two uint64."""
    f = vidx[:, FACET_CORNERS].reshape(-1, 4).astype(np.uint64)
    f.sort(axis=1)
    keys = np.empty((f.shape[0], 2), dtype=np.uint64)
    keys[:, 0] = (f[:, 0] << np.uint64(32)) | f[:, 1]
    keys[:, 1] = (f[:, 2] << np.uint64(32)) | f[:, 3]
    return keys


def face_multiplicity(vidx):
    """
    Count how many times the face of each (element, facet) pair occurs in the mesh.

    Parameters
    ----------
    vidx : ndarray
        Point ids, shape (nelv, 8).

    Returns
    -------
    counts : ndarray
        Multiplicity of each facet, shape (nelv, 6). A value of 1 marks an
        external face.
    n_unique : int
        Number of unique faces in the mesh.
    """
    keys = _pack_faces(vidx).view([("", "V16")]).ravel()
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inverse].reshape(-1, 6), int(counts.size)


def count_edges(vidx):
    """
    Count the unique edges of the mesh.

    Parameters
    ----------
    vidx : ndarray
        Point ids, shape (nelv, 8).

    Returns
    -------
    int
        Number of unique edges.
    """
    e = vidx[:, EDGE_CORNERS].reshape(-1, 2).astype(np.uint64)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    return int(np.unique((lo << np.uint64(32)) | hi).size)


def skin(vidx):
    """
    Find the external surface of the mesh.

    Parameters
    ----------
    vidx : ndarray
        Point ids, shape (nelv, 8). With raw (unmerged) ids the periodic
        boundaries are part of the skin, with merged ids they are not.

    Returns
    -------
    elem_pos : ndarray
        Record positions of the elements with an external facet.
    facet : ndarray
        The 0-based facet index of each entry in ``elem_pos``.
    """
    mult, _ = face_multiplicity(vidx)
    epos, fct = np.nonzero(mult == 1)
    return epos.astype(np.int64), fct.astype(np.int64)


def dual_graph(cell):
    """
    Build the vertex-weighted element dual graph.

    The graph is ``A = E E^T`` with the diagonal removed, where ``E`` is the
    element to vertex incidence matrix, so ``A[i, j]`` is the number of
    vertices shared by elements ``i`` and ``j``.

    Parameters
    ----------
    cell : ndarray
        Dense 0-based point ids, shape (nelv, 8), see :func:`compress_ids`.

    Returns
    -------
    scipy.sparse.csr_matrix
        The symmetric weighted adjacency matrix of shape (nelv, nelv).
    """
    import scipy.sparse as sp  # pylint: disable=import-outside-toplevel

    nelv = cell.shape[0]
    npts = int(cell.max()) + 1
    rows = np.repeat(np.arange(nelv, dtype=np.int64), 8)
    incidence = sp.csr_matrix(
        (np.ones(nelv * 8), (rows, cell.ravel())), shape=(nelv, npts)
    )
    adjacency = (incidence @ incidence.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency
