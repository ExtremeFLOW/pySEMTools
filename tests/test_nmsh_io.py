import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Import general modules
import numpy as np
import pytest

# Import relevant modules
from pysemtools.io.nmsh import (
    EL_DT,
    ZONE_DT,
    CURVE_DT,
    VERTEX_IJK,
    ZONE_PERIODIC,
    ZONE_LABELLED,
    NmshFormatError,
    read_nmsh,
    write_nmsh,
    iter_nmsh_elements,
)
from pysemtools.io.re2 import read_re2, Re2FormatError, RE2_EL_DT
from pysemtools.datatypes import NmshMesh, Re2Mesh, Coef
from pysemtools.datatypes.corner_mesh_geometry import gll_nodes, gll_coordinates, min_jacobian

#==============================================================================

NEL = (3, 2, 2)
BOUNDS = (0.0, 3.0, 0.0, 2.0, 0.0, 1.0)


def build_box():
    """
    Build the record arrays of a box mesh by hand: lexicographic point ids,
    a labelled zone on the x-min face and a periodic pair in z.
    """
    nx, ny, nz = NEL
    gx = np.linspace(BOUNDS[0], BOUNDS[1], nx + 1)
    gy = np.linspace(BOUNDS[2], BOUNDS[3], ny + 1)
    gz = np.linspace(BOUNDS[4], BOUNDS[5], nz + 1)
    nelv = nx * ny * nz
    elems = np.zeros(nelv, dtype=EL_DT)
    zones = []
    for e in range(nelv):
        ez, ey, ex = e // (nx * ny), (e // nx) % ny, e % nx
        elems["id"][e] = e + 1
        for sl in range(8):
            a, b, c = ex + VERTEX_IJK[sl, 0], ey + VERTEX_IJK[sl, 1], ez + VERTEX_IJK[sl, 2]
            elems["v"]["idx"][e, sl] = 1 + a + b * (nx + 1) + c * (nx + 1) * (ny + 1)
            elems["v"]["xyz"][e, sl] = (gx[a], gy[b], gz[c])
        if ex == 0:
            zones.append((e + 1, 1, 0, 4, (0, 0, 0, 0), ZONE_LABELLED))
        if ez == 0:
            partner = 1 + ex + ey * nx + (nz - 1) * nx * ny
            zones.append((e + 1, 5, partner, 6, (0, 0, 0, 0), ZONE_PERIODIC))
    zones = np.array(zones, dtype=ZONE_DT)
    curves = np.zeros(1, dtype=CURVE_DT)
    curves["e"] = 2
    curves["type"][0, 0] = 3
    curves["data"][0, 0, :] = (1.0, 2.0, 3.0, 4.0, 5.0)
    return elems, zones, curves


def test_write_read_roundtrip(tmp_path):

    elems, zones, curves = build_box()
    fname = str(tmp_path / "box.nmsh")
    write_nmsh(fname, elems, (zones[zones["t"] == 5], zones[zones["t"] == 7]), curves)

    data = read_nmsh(fname)
    assert data.glb_nelv == elems.shape[0]
    assert np.array_equal(data.elems, elems)
    assert data.zones.shape[0] == zones.shape[0]
    assert np.array_equal(data.curves, curves)
    assert data.trailing == 0

    chunks = list(iter_nmsh_elements(fname, chunk=5))
    assert [s for s, _ in chunks] == [0, 5, 10]
    assert np.array_equal(np.concatenate([e for _, e in chunks]), elems)

    # A distributed read with the world communicator, then a collective write
    nmsh = NmshMesh.from_file(fname, comm)
    assert nmsh.is_distributed and nmsh.glb_nelv == elems.shape[0]
    assert comm.allreduce(nmsh.nelv) == elems.shape[0]
    copy = comm.bcast(str(tmp_path / "copy.nmsh"), root=0)
    nmsh.write(copy)
    back = read_nmsh(copy)
    assert np.array_equal(back.elems, elems)
    assert np.array_equal(np.sort(back.zones.view("V36")), np.sort(zones.view("V36")))

    # gather restores file order, distribute goes back
    gathered = nmsh.gather()
    assert not gathered.is_distributed
    assert np.array_equal(gathered.elems, elems)
    in_file_order = np.concatenate([zones[zones["t"] == 5], zones[zones["t"] == 7]])
    assert np.array_equal(gathered.zones, in_file_order)
    again = gathered.distribute(comm)
    assert np.array_equal(again.elems, nmsh.elems)


def test_mesh_properties():

    elems, zones, curves = build_box()
    nmsh = NmshMesh(elems, zones, curves)
    nmsh.validate()
    assert nmsh.nelv == nmsh.glb_nelv == np.prod(NEL)
    assert nmsh.vertex_ids.shape == (nmsh.nelv, 8)
    assert nmsh.periodic_zones.shape[0] == NEL[0] * NEL[1]
    assert nmsh.labelled_zones.shape[0] == NEL[1] * NEL[2]
    assert (nmsh.labelled_zones["p_f"] == 4).all()
    lo, hi = nmsh.bounding_box()
    assert np.allclose(lo, BOUNDS[0::2]) and np.allclose(hi, BOUNDS[1::2])
    assert np.allclose(nmsh.centroids()[0], (0.5, 0.5, 0.25))
    pos = nmsh.element_positions()
    assert np.array_equal(pos[1:], np.arange(nmsh.nelv))


def test_validation_errors(tmp_path):

    elems, zones, curves = build_box()

    bad = NmshMesh(elems, zones.copy(), curves)
    bad.zones["f"][0] = 9
    with pytest.raises(NmshFormatError):
        bad.validate()

    bad = NmshMesh(elems, zones, curves.copy())
    bad.curves["type"][0, 0] = 2
    with pytest.raises(NmshFormatError):
        bad.validate()

    bad = NmshMesh(elems.copy(), zones, curves)
    bad.elems["id"][0] = 3
    with pytest.raises(NmshFormatError):
        bad.element_positions()

    fname = str(tmp_path / "box.nmsh")
    write_nmsh(fname, elems, (zones,), curves)
    with open(fname, "rb") as f:
        payload = f.read()
    truncated = str(tmp_path / "trunc.nmsh")
    with open(truncated, "wb") as f:
        f.write(payload[: len(payload) // 2])
    with pytest.raises(NmshFormatError):
        read_nmsh(truncated)


def test_geometry_and_sem_mesh():

    assert np.allclose(gll_nodes(3), [-1.0, 0.0, 1.0])
    elems, zones, curves = build_box()
    nmsh = NmshMesh(elems, zones, curves)

    lx = 4
    xyz = gll_coordinates(nmsh.corner_coordinates, lx)
    assert xyz.shape == (nmsh.nelv, lx**3, 3)
    # the first and last GLL points are the corners of the first and last vertex
    assert np.allclose(xyz[:, 0], nmsh.corner_coordinates[:, 0])
    assert np.allclose(xyz[:, -1], nmsh.corner_coordinates[:, 6])
    assert (min_jacobian(nmsh.corner_coordinates) > 0).all()

    msh = nmsh.to_sem_mesh(comm, lx=lx)
    assert comm.allreduce(msh.nelv) == nmsh.glb_nelv
    assert msh.x.shape[1:] == (lx, lx, lx)
    coef = Coef(msh, comm)
    volume = np.prod(np.array(BOUNDS[1::2]) - np.array(BOUNDS[0::2]))
    assert np.isclose(comm.allreduce(float(coef.B.sum())), volume)


def test_re2_reader(tmp_path):

    # A minimal #v002 file with one element and one wall boundary condition
    fname = str(tmp_path / "one.re2")
    hdr = f"#v002{1:9d}{3:3d}{1:9d}".ljust(80).encode("ascii")
    el_dt = np.dtype([("rg", "<f8"), ("x", "<f8", (8,)), ("y", "<f8", (8,)), ("z", "<f8", (8,))])
    bc_dt = np.dtype([("e", "<f8"), ("f", "<f8"), ("d", "<f8", (5,)), ("t", "S8")])
    el = np.zeros(1, dtype=el_dt)
    el["x"][0], el["y"][0], el["z"][0] = VERTEX_IJK[:, 0], VERTEX_IJK[:, 1], VERTEX_IJK[:, 2]
    bc = np.array([(1.0, 1.0, (0, 0, 0, 0, 0), b"W")], dtype=bc_dt)
    with open(fname, "wb") as f:
        f.write(hdr)
        np.float32(6.54321).tofile(f)
        el.tofile(f)
        np.array([0.0]).tofile(f)
        np.array([1.0]).tofile(f)
        bc.tofile(f)

    re2 = read_re2(fname)
    assert re2.nelv == 1 and re2.version == "#v002"
    assert np.array_equal(re2.xyz[0], VERTEX_IJK)
    assert re2.bcs.shape[0] == 1
    assert re2.curves.shape[0] == 0
    assert re2.elems.dtype == RE2_EL_DT

    with open(fname, "wb") as f:
        f.write(hdr[:40])
    with pytest.raises(Re2FormatError):
        read_re2(fname)


def test_read_neko_hemi_mesh():

    # hemi.nmsh was written by Neko's own rea2nbin from hemi.re2 (both from the Neko repository)
    fname = "examples/data/hemi.nmsh"
    data = read_nmsh(fname)
    assert data.glb_nelv == 2042
    assert data.zones.shape[0] == 1232
    assert data.curves.shape[0] == 0  # hemi has an 's' curve, so Neko treats it as non-curved
    assert (data.zones["t"] == ZONE_LABELLED).all()

    nmsh = NmshMesh.from_file(fname, comm)
    assert nmsh.glb_nelv == 2042
    gathered = nmsh.gather()
    assert np.array_equal(gathered.elems, data.elems)
    assert np.array_equal(gathered.zones, data.zones)
    lo, hi = nmsh.bounding_box()
    assert np.all(np.isfinite(lo)) and np.all(hi > lo)
    assert (min_jacobian(gathered.corner_coordinates) > 0).all()

    re2 = read_re2("examples/data/hemi.re2")
    assert re2.nelv == 2042 and re2.version == "#v002"
    assert np.array_equal(re2.xyz, gathered.corner_coordinates)


def test_re2_mesh(tmp_path):

    # Every rank gets its own tmp_path from pytest; share the one of rank 0
    tmp_path = comm.bcast(tmp_path, root=0)
    fname = "examples/data/hemi.re2"
    serial = Re2Mesh.from_file(fname)
    assert not serial.is_distributed
    assert serial.nelv == serial.glb_nelv == 2042
    assert serial.curves.shape[0] == 700 and serial.bcs.shape[0] == 1232
    assert set(serial.bc_types()) == {"O", "SYM", "W", "v"}
    assert np.array_equal(serial.element_ids, np.arange(1, 2043))
    serial.validate()

    # Distributed read agrees with the serial one, gather restores file order
    dist = Re2Mesh.from_file(fname, comm)
    assert dist.is_distributed and dist.glb_nelv == 2042
    assert comm.allreduce(dist.nelv) == 2042
    assert np.array_equal(dist.elems, serial.elems[dist.offset_el : dist.offset_el + dist.nelv])
    owned = (dist.bcs["e"] - 1 >= dist.offset_el) & (dist.bcs["e"] - 1 < dist.offset_el + dist.nelv)
    assert owned.all()
    gathered = dist.gather()
    assert np.array_equal(gathered.elems, serial.elems)
    assert np.array_equal(gathered.curves, serial.curves)
    assert np.array_equal(gathered.bcs, serial.bcs)
    on_root = dist.gather(root=0)
    if comm.Get_rank() == 0:
        assert np.array_equal(on_root.bcs, serial.bcs)
    else:
        assert on_root is None
    again = serial.distribute(comm)
    assert np.array_equal(again.elems, dist.elems) and np.array_equal(again.bcs, dist.bcs)

    # Writing hemi back reproduces the file byte for byte (it is a #v002 file)
    out = str(tmp_path / "hemi_copy.re2")
    serial.write(out, comm=comm)
    with open(fname, "rb") as f1, open(out, "rb") as f2:
        assert f1.read() == f2.read()
    out_dist = str(tmp_path / "hemi_dist.re2")
    dist.write(out_dist)
    back = Re2Mesh.from_file(out_dist)
    assert np.array_equal(back.elems, serial.elems)
    assert np.array_equal(np.sort(back.bcs.view("V64")), np.sort(serial.bcs.view("V64")))

    # The GLL mesh of the re2 equals the one of the nmsh written by Neko from it
    lx = 4
    msh_re2 = dist.to_sem_mesh(lx=lx)
    msh_nmsh = NmshMesh.from_file("examples/data/hemi.nmsh", comm).to_sem_mesh(lx=lx)
    assert np.allclose(msh_re2.x, msh_nmsh.x) and np.allclose(msh_re2.z, msh_nmsh.z)
    lo, hi = dist.bounding_box()
    assert np.all(np.isfinite(lo))
    assert np.all(hi > lo)
    coef = Coef(msh_re2, comm)
    assert comm.allreduce(float(coef.B.sum())) > 0
