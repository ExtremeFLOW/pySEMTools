import numpy as np
import h5py
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkHexahedron, VtkQuad

def ijk_to_id(i, j, k, ny, nz):
    return i*(ny*nz) + j*nz + k

def ij_to_id(i, j, ny):
    return i*ny + j

def build_cells_with_hole(x, y, z, center=(0.0, 0.0), R=1.0, yc=0.0, tol=0.0):
    """
    Build unstructured cell connectivity from a structured point grid:
      - if nz > 1: hexahedra (3D)
      - if nz == 1: quads (2D slice)

    Removes:
      1) cells that "bridge" across the cut y=yc (some vertices below and some above)
      2) cells whose center is inside cylinder r < R+tol
    """
    nx, ny, nz = x.shape
    xc, yc0 = center

    if nz == 1:
        # ---- 2D quads ----
        conn_list = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                v0 = ij_to_id(i,   j,   ny)
                v1 = ij_to_id(i+1, j,   ny)
                v2 = ij_to_id(i+1, j+1, ny)
                v3 = ij_to_id(i,   j+1, ny)

                xverts = np.array([x[i, j, 0], x[i+1, j, 0], x[i+1, j+1, 0], x[i, j+1, 0]])
                yverts = np.array([y[i, j, 0], y[i+1, j, 0], y[i+1, j+1, 0], y[i, j+1, 0]])

                # remove bridge cells across cut
                if (yverts.min() < yc) and (yverts.max() > yc) and (xverts.min() < xc - R) and (xverts.max() > xc + R):
                    continue

                # remove inside-cylinder cells via cell center
                xcen = 0.25 * (x[i, j, 0] + x[i+1, j, 0] + x[i+1, j+1, 0] + x[i, j+1, 0])
                ycen = 0.25 * (y[i, j, 0] + y[i+1, j, 0] + y[i+1, j+1, 0] + y[i, j+1, 0])
                r = np.sqrt((xcen - xc)**2 + (ycen - yc0)**2)
                if r < (R + tol):
                    continue

                conn_list.append([v0, v1, v2, v3])

        conn = np.asarray(conn_list, dtype=np.int64).ravel()
        ncells = len(conn_list)
        offsets = (4 * np.arange(1, ncells + 1, dtype=np.int64))
        celltypes = np.full(ncells, VtkQuad.tid, dtype=np.uint8)
        return conn, offsets, celltypes

    else:
        # ---- 3D hexes ----
        conn_list = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    v0 = ijk_to_id(i,   j,   k,   ny, nz)
                    v1 = ijk_to_id(i+1, j,   k,   ny, nz)
                    v2 = ijk_to_id(i+1, j+1, k,   ny, nz)
                    v3 = ijk_to_id(i,   j+1, k,   ny, nz)
                    v4 = ijk_to_id(i,   j,   k+1, ny, nz)
                    v5 = ijk_to_id(i+1, j,   k+1, ny, nz)
                    v6 = ijk_to_id(i+1, j+1, k+1, ny, nz)
                    v7 = ijk_to_id(i,   j+1, k+1, ny, nz)

                    yverts = np.array([
                        y[i,   j,   k],   y[i+1, j,   k],   y[i+1, j+1, k],   y[i,   j+1, k],
                        y[i,   j,   k+1], y[i+1, j,   k+1], y[i+1, j+1, k+1], y[i,   j+1, k+1],
                    ])

                    # remove bridge cells across cut
                    if (yverts.min() < yc) and (yverts.max() > yc):
                        continue

                    # remove inside-cylinder cells via cell center (x,y only)
                    xcen = 0.125 * (
                        x[i, j, k] + x[i+1, j, k] + x[i+1, j+1, k] + x[i, j+1, k] +
                        x[i, j, k+1] + x[i+1, j, k+1] + x[i+1, j+1, k+1] + x[i, j+1, k+1]
                    )
                    ycen = 0.125 * (
                        y[i, j, k] + y[i+1, j, k] + y[i+1, j+1, k] + y[i, j+1, k] +
                        y[i, j, k+1] + y[i+1, j, k+1] + y[i+1, j+1, k+1] + y[i, j+1, k+1]
                    )
                    r = np.sqrt((xcen - xc)**2 + (ycen - yc0)**2)
                    if r < (R + tol):
                        continue

                    conn_list.append([v0, v1, v2, v3, v4, v5, v6, v7])

        conn = np.asarray(conn_list, dtype=np.int64).ravel()
        ncells = len(conn_list)
        offsets = (8 * np.arange(1, ncells + 1, dtype=np.int64))
        celltypes = np.full(ncells, VtkHexahedron.tid, dtype=np.uint8)
        return conn, offsets, celltypes


# -----------------------
# -----------------------
mesh_fname = "./meshgrid_data/points.hdf5"
with h5py.File(mesh_fname, 'r') as f:
    x = f["x"][:]  # (nx, ny, nz)
    y = f["y"][:]
    z = f["z"][:]

center = (0.0, 0.0)
R = 1.0
yc = 0.0

nx, ny, nz = x.shape

# Flatten points: for nz==1, flatten (nx,ny) slice; for nz>1 flatten full 3D
if nz == 1:
    x1 = np.ascontiguousarray(x[:, :, 0].ravel(order="C"))
    y1 = np.ascontiguousarray(y[:, :, 0].ravel(order="C"))
    z1 = np.ascontiguousarray(z[:, :, 0].ravel(order="C"))
else:
    x1 = np.ascontiguousarray(x.ravel(order="C"))
    y1 = np.ascontiguousarray(y.ravel(order="C"))
    z1 = np.ascontiguousarray(z.ravel(order="C"))

# Build connectivity once (reuse per timestep)
conn, offsets, celltypes = build_cells_with_hole(
    x, y, z, center=center, R=R, yc=yc, tol=0.0
)

for it in range(0, 20):
    field_fname = "field" + str(it+1).zfill(5) + ".hdf5"
    with h5py.File(field_fname, 'r') as f:
        field_dict = {key: np.ascontiguousarray(f[key][:]) for key in f.keys()}

    if nz == 1:
        pointData = {k: np.ascontiguousarray(v[:, :, 0].ravel(order="C")) for k, v in field_dict.items()}
    else:
        pointData = {k: np.ascontiguousarray(v.ravel(order="C")) for k, v in field_dict.items()}

    unstructuredGridToVTK(
        "field" + str(it+1).zfill(5),
        x1, y1, z1,
        connectivity=conn,
        offsets=offsets,
        cell_types=celltypes,
        pointData=pointData
    )
