"""
Geometry of the straight-sided elements described by a Neko ``.nmsh`` mesh.

The eight corners of an element define a trilinear map from the reference
cube to physical space. This module evaluates that map and its Jacobian at
the GLL points of a chosen polynomial order and bridges to the GLL based
:class:`pysemtools.datatypes.msh.Mesh` used by the rest of the package.
"""

import numpy as np

from .coef import GLL_pwts
from .msh import Mesh
from ..io.nmsh import VERTEX_IJK

__all__ = [
    "gll_nodes",
    "trilinear_shape_functions",
    "gll_coordinates",
    "min_jacobian",
    "facet_gll_mask",
    "to_sem_mesh",
]

_shape_cache = {}


def gll_nodes(n):
    """
    GLL nodes on [-1, 1] in ascending order.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    ndarray
        The nodes, shape (n,).
    """
    xi, _ = GLL_pwts(n)
    xi = np.sort(np.asarray(xi, dtype=np.float64))
    # Pin the end points and the symmetric midpoint so that the corner
    # coordinates are reproduced exactly by the trilinear map.
    xi[0], xi[-1] = -1.0, 1.0
    if n % 2 == 1:
        xi[n // 2] = 0.0
    return xi


def trilinear_shape_functions(n=3):
    """
    Trilinear shape functions and their derivatives at the GLL points.

    The GLL points are ordered with the r index fastest, then s, then t, so
    that reshaping to ``(n, n, n)`` gives the ``(lz, ly, lx)`` ordering used
    by :class:`pysemtools.datatypes.msh.Mesh`.

    Parameters
    ----------
    n : int, optional
        Number of GLL points per direction. Default is 3.

    Returns
    -------
    shape : ndarray
        Shape function values, shape (n**3, 8).
    dshape : ndarray
        Derivatives with respect to r, s and t, shape (3, n**3, 8).
    """
    if n in _shape_cache:
        return _shape_cache[n]
    rst_corner = 2.0 * VERTEX_IJK.astype(np.float64) - 1.0
    xi = gll_nodes(n)
    t, s, r = np.meshgrid(xi, xi, xi, indexing="ij")
    pts = np.stack([r.ravel(), s.ravel(), t.ravel()], axis=1)
    one = 1.0 + pts[:, None, :] * rst_corner[None, :, :]
    shape = 0.125 * one.prod(axis=2)
    dshape = np.empty((3, n**3, 8))
    for c in range(3):
        f = one.copy()
        f[:, :, c] = rst_corner[None, :, c]
        dshape[c] = 0.125 * f.prod(axis=2)
    _shape_cache[n] = (shape, dshape)
    return shape, dshape


def gll_coordinates(corners, n=3):
    """
    Coordinates of the GLL points of straight-sided elements.

    Parameters
    ----------
    corners : ndarray
        Corner coordinates, shape (m, 8, 3).
    n : int, optional
        Number of GLL points per direction. Default is 3.

    Returns
    -------
    ndarray
        Coordinates, shape (m, n**3, 3), with the point ordering of
        :func:`trilinear_shape_functions`.
    """
    shape, _ = trilinear_shape_functions(n)
    return np.einsum("pk,mkc->mpc", shape, corners)


def min_jacobian(corners, n=3, chunk=1 << 20):
    """
    Minimum Jacobian determinant of the trilinear map over the GLL points.

    Parameters
    ----------
    corners : ndarray
        Corner coordinates, shape (m, 8, 3).
    n : int, optional
        Number of GLL points per direction. Default is 3.
    chunk : int, optional
        Number of elements processed at once. Default is 2**20.

    Returns
    -------
    ndarray
        Minimum determinant per element, shape (m,). A value ``<= 0`` flags
        an inverted or degenerate element.
    """
    _, dshape = trilinear_shape_functions(n)
    out = np.empty(corners.shape[0], dtype=np.float64)
    for s in range(0, corners.shape[0], chunk):
        jac = np.einsum("dpk,mkc->mpdc", dshape, corners[s : s + chunk])
        det = (
            jac[..., 0, 0] * (jac[..., 1, 1] * jac[..., 2, 2] - jac[..., 1, 2] * jac[..., 2, 1])
            - jac[..., 0, 1] * (jac[..., 1, 0] * jac[..., 2, 2] - jac[..., 1, 2] * jac[..., 2, 0])
            + jac[..., 0, 2] * (jac[..., 1, 0] * jac[..., 2, 1] - jac[..., 1, 1] * jac[..., 2, 0])
        )
        out[s : s + chunk] = det.min(axis=1)
    return out


def facet_gll_mask(n=3):
    """
    Mask of the GLL points that lie on each facet.

    Parameters
    ----------
    n : int, optional
        Number of GLL points per direction. Default is 3.

    Returns
    -------
    ndarray
        Boolean array of shape (6, n**3), row ``f`` marks the points on
        Neko facet ``f + 1``.
    """
    idx = np.arange(n**3)
    ir, is_, it = idx % n, (idx // n) % n, idx // (n * n)
    mask = np.zeros((6, n**3), dtype=bool)
    mask[0], mask[1] = ir == 0, ir == n - 1
    mask[2], mask[3] = is_ == 0, is_ == n - 1
    mask[4], mask[5] = it == 0, it == n - 1
    return mask


def to_sem_mesh(nmsh, comm, lx=3, create_connectivity=False):
    """
    Build a :class:`pysemtools.datatypes.msh.Mesh` from a :class:`pysemtools.datatypes.nmsh.NmshMesh`.

    Parameters
    ----------
    nmsh : NmshMesh
        The topological mesh.
    comm : MPI.Comm
        MPI communicator for the ``Mesh`` object.
    lx : int, optional
        Number of GLL points per direction. Default is 3.
    create_connectivity : bool, optional
        Passed on to the ``Mesh`` constructor. Default is False.

    Returns
    -------
    Mesh
        Mesh with coordinates of shape (nelv, lx, lx, lx) and ``elmap`` set
        to the global element ids in record order.
    """
    xyz = gll_coordinates(nmsh.corner_coordinates, lx)
    nelv = xyz.shape[0]
    x = np.ascontiguousarray(xyz[:, :, 0].reshape(nelv, lx, lx, lx))
    y = np.ascontiguousarray(xyz[:, :, 1].reshape(nelv, lx, lx, lx))
    z = np.ascontiguousarray(xyz[:, :, 2].reshape(nelv, lx, lx, lx))
    elmap = nmsh.element_ids.astype(np.int32).copy()
    return Mesh(
        comm, x=x, y=y, z=z, elmap=elmap, create_connectivity=create_connectivity
    )
