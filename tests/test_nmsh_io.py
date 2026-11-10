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
    FACE_VERTICES,
    EDGE_VERTICES,
    ZONE_PERIODIC,
    ZONE_LABELLED,
    NmshFormatError,
    read_nmsh,
    write_nmsh,
    iter_nmsh_elements,
)
from pysemtools.io.re2 import (
    read_re2,
    Re2FormatError,
    RE2_EL_DT,
    RE2_CURVE_DT,
    RE2_BC_DT,
    RE2_TO_NEKO_FACET,
)
from pysemtools.comm.distribution import linear_distribution
from pysemtools.datatypes import NmshMesh, Re2Mesh, Coef
from pysemtools.datatypes.corner_mesh_topology import (
    deduplicate_points,
    CORNER_IJK,
    FACET_CORNERS,
    EDGE_CORNERS,
)
from pysemtools.datatypes.corner_mesh_geometry import (
    gll_nodes,
    gll_coordinates,
    min_jacobian,
    facet_gll_mask,
)

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


def test_deduplicate_points():

    # The worked example of the comments: corners B, A, B, C are numbered 0, 1, 0, 2
    A, B, C = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]
    rest = [[float(k), 1.0, 0.0] for k in range(4)]
    vid, nuniq = deduplicate_points(np.array([[B, A, B, C] + rest]))
    assert vid.dtype == np.int64 and vid.shape == (1, 8)
    assert vid[0].tolist() == [0, 1, 0, 2, 3, 4, 5, 6] and nuniq == 7

    # Random corners drawn from a small pool, checked against a brute force numbering
    rng = np.random.default_rng(7)
    pool = rng.random((50, 3))
    pick = rng.integers(0, pool.shape[0], size=(40, 8))
    xyz = pool[pick]
    vid, nuniq = deduplicate_points(xyz)
    seen, expected = {}, []
    for row in xyz.reshape(-1, 3):
        key = row.tobytes()
        expected.append(seen.setdefault(key, len(seen)))
    assert vid.reshape(-1).tolist() == expected
    assert nuniq == len(seen)

    # Bit-exact comparison: -0.0 and 0.0 are different points
    two = np.zeros((1, 8, 3))
    two[0, 1, 0] = -0.0
    assert deduplicate_points(two)[1] == 2


def test_corner_convention():

    # Both mesh file formats store corners in the corner convention, so their data types
    # return the records unchanged
    assert np.array_equal(CORNER_IJK, VERTEX_IJK)

    # With two GLL points per direction the GLL points are the corners themselves, numbered
    # i + 2 j + 4 k; the trilinear map must put corner c at its reference position
    cube = CORNER_IJK.astype(np.float64)[None]
    xyz = gll_coordinates(cube, n=2)[0]
    for c, (i, j, k) in enumerate(CORNER_IJK):
        assert np.array_equal(xyz[i + 2 * j + 4 * k], cube[0, c])
    assert min_jacobian(cube, n=3)[0] == pytest.approx(0.125)

    # Swapping two corners breaks the convention, which the Jacobian exposes
    twisted = cube.copy()
    twisted[0, [2, 3]] = twisted[0, [3, 2]]
    assert min_jacobian(twisted, n=3)[0] <= 0.0


def test_canonical_tables():

    # The canonical corners and facets are those of the .nmsh format
    assert np.array_equal(CORNER_IJK, VERTEX_IJK)
    assert np.array_equal(FACET_CORNERS, FACE_VERTICES)

    # Facet f is the face where reference direction f // 2 is at its low (even f) or high end
    for f, corners in enumerate(FACET_CORNERS):
        ref = CORNER_IJK[corners]
        fixed = np.flatnonzero((ref == ref[0]).all(axis=0))
        assert fixed.tolist() == [f // 2] and ref[0, f // 2] == f % 2

    # The GLL points of each facet, derived from the table, are the expected index ranges
    for n in range(2, 6):
        idx = np.arange(n**3)
        ijk = (idx % n, (idx // n) % n, idx // (n * n))
        mask = facet_gll_mask(n)
        for f in range(6):
            assert np.array_equal(mask[f], ijk[f // 2] == (f % 2) * (n - 1))

    # The edges are the twelve edges of the cube, numbered like the curve records: round the
    # bottom face, round the top face, then the vertical edges
    assert {tuple(sorted(e)) for e in EDGE_CORNERS} == {tuple(sorted(e)) for e in EDGE_VERTICES}
    for e, (a, b) in enumerate(EDGE_CORNERS):
        assert np.abs(CORNER_IJK[a] - CORNER_IJK[b]).sum() == 1
        if e < 8:
            base = 4 * (e // 4)
            assert (a, b) == (base + e % 4, base + (e + 1) % 4)
        else:
            assert (a, b) == (e - 8, e - 4)


def test_deduplicate_points_distributed():

    rank, size = comm.Get_rank(), comm.Get_size()
    rng = np.random.default_rng(11)
    pool = rng.random((60, 3))
    pool[0] = 0.0
    pool[1] = (-0.0, 0.0, 0.0)  # bitwise different from pool[0]
    for glb in (50, 1):
        xyz = pool[rng.integers(0, pool.shape[0], size=(glb, 8))]
        serial, n_serial = deduplicate_points(xyz)

        # On a single rank the parallel numbering is the serial one
        single, n_single = deduplicate_points(xyz, MPI.COMM_SELF)
        assert np.array_equal(single, serial) and n_single == n_serial

        # Every rank holds a block of the elements, some ranks may hold none
        n, off = linear_distribution(glb, comm)
        local, n_unique = deduplicate_points(xyz[off : off + n], comm)
        assert local.shape == (n, 8) and local.dtype == np.int64
        assert n_unique == n_serial
        dist = np.concatenate(comm.allgather(local)).reshape(-1)

        # The same points get the same ids on all ranks: a bijection with the serial ids,
        # and the ids are dense
        pairs = np.unique(np.stack([serial.reshape(-1), dist], axis=1), axis=0)
        assert pairs.shape[0] == n_serial
        assert np.unique(pairs[:, 1]).size == n_serial
        assert np.array_equal(np.unique(dist), np.arange(n_serial))


def test_point_ids():

    # Neko's rea2nbin numbered the points of hemi.nmsh in order of first appearance, which is
    # also the numbering of a replicated Re2Mesh; hemi has no periodic facets
    re2 = Re2Mesh.from_file("examples/data/hemi.re2", log_level="hide")
    nmsh = NmshMesh.from_file("examples/data/hemi.nmsh", log_level="hide")
    assert np.array_equal(nmsh.point_ids(), nmsh.vertex_ids - 1)
    assert np.array_equal(re2.point_ids(), nmsh.point_ids())

    # Distributed: the nmsh ids come from the file, the re2 ids are numbered in parallel and
    # must identify the same points
    nmsh_dist = NmshMesh.from_file("examples/data/hemi.nmsh", comm, log_level="hide")
    lo, hi = nmsh_dist.offset_el, nmsh_dist.offset_el + nmsh_dist.nelv
    assert np.array_equal(nmsh_dist.point_ids(), nmsh.point_ids()[lo:hi])
    re2_dist = Re2Mesh.from_file("examples/data/hemi.re2", comm, log_level="hide")
    ids = np.concatenate(comm.allgather(re2_dist.point_ids())).reshape(-1)
    pairs = np.unique(np.stack([re2.point_ids().reshape(-1), ids], axis=1), axis=0)
    nuniq = np.unique(re2.point_ids()).size
    assert pairs.shape[0] == nuniq and np.unique(pairs[:, 1]).size == nuniq


def re2_from_nmsh_records(elems, zones):
    """The records of an Re2Mesh with the corners and zones of an nmsh box."""
    neko_to_re2 = {int(RE2_TO_NEKO_FACET[k]): k + 1 for k in range(6)}
    re2_elems = np.zeros(elems.shape[0], dtype=RE2_EL_DT)
    for c, name in enumerate("xyz"):
        re2_elems[name] = elems["v"]["xyz"][:, :, c]
    bcs = np.zeros(zones.shape[0], dtype=RE2_BC_DT)
    bcs["e"] = zones["e"]
    bcs["f"] = [neko_to_re2[int(f)] for f in zones["f"]]
    periodic = zones["t"] == ZONE_PERIODIC
    bcs["t"] = np.where(periodic, b"P       ", b"MSH     ")
    bcs["d"][periodic, 0] = zones["p_e"][periodic]
    bcs["d"][periodic, 1] = [neko_to_re2[int(f)] for f in zones["p_f"][periodic]]
    bcs["d"][~periodic, 4] = zones["p_f"][~periodic]
    return re2_elems, np.zeros(0, dtype=RE2_CURVE_DT), bcs


def test_periodic_pairs_and_boundary_facets():

    elems, zones, curves = build_box()
    nmsh = NmshMesh(elems, zones, curves)
    re2 = Re2Mesh(*re2_from_nmsh_records(elems, zones))

    # Both data types give the same canonical pairs and boundary facets
    nx, ny, nz = NEL
    pairs = np.stack(nmsh.periodic_pairs(), axis=1)
    assert pairs.shape == (nx * ny, 4)
    assert (pairs[:, 1] == 4).all() and (pairs[:, 3] == 5).all()  # t = -1 paired with t = +1
    assert np.array_equal(np.stack(re2.periodic_pairs(), axis=1), pairs)
    e7, f7, label = nmsh.boundary_facets()
    assert (f7 == 0).all() and (label == 4).all() and e7.size == ny * nz
    e_re2, f_re2, tag = re2.boundary_facets()
    assert np.array_equal(e_re2, e7) and np.array_equal(f_re2, f7)
    assert set(tag.tolist()) == {"MSH"}

    # Distributed: every rank lists the facets of its own elements, and together they are the
    # facets of the replicated mesh
    for mesh in (nmsh, re2):
        dist = mesh.distribute(comm)
        local = np.stack(dist.periodic_pairs(), axis=1)
        own = (local[:, 0] >= dist.offset_el) & (local[:, 0] < dist.offset_el + dist.nelv)
        assert own.all()
        glb = np.concatenate(comm.allgather(local))
        assert np.array_equal(glb[np.lexsort(glb.T[::-1])], pairs[np.lexsort(pairs.T[::-1])])
        e_loc, f_loc, _ = dist.boundary_facets()
        e_glb = np.concatenate(comm.allgather(e_loc))
        assert np.array_equal(np.sort(e_glb), np.sort(e7))
