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
from pysemtools.convert import (
    convert,
    conversions,
    get_conversion,
    detect_format,
    register,
    Conversion,
    re2_to_nmsh,
    mesh_to_fld,
)
from pysemtools.convert.registry import _CONVERSIONS
from pysemtools.datatypes import NmshMesh, Mesh
from pysemtools.datatypes.corner_mesh_geometry import min_jacobian
from pysemtools.io.nmsh import VERTEX_IJK, EL_DT, ZONE_DT
from pysemtools.io.re2 import RE2_TO_NEKO_FACET
from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.cli.convert import main as convert_cli

#==============================================================================

serial_only = pytest.mark.skipif(comm.Get_size() > 1, reason="serial conversion")

BOUNDS = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
NEL = (4, 3, 3)  # at least 3 elements per direction so periodic faces stay distinct
HEMI_RE2 = "examples/data/hemi.re2"
HEMI_NMSH = "examples/data/hemi.nmsh"


@pytest.fixture
def shared_tmp(tmp_path):
    """tmp_path of rank 0, so all ranks work on the same files."""
    return comm.bcast(tmp_path, root=0)


def write_re2(fname, bounds, nel):
    """
    Write a NEKTON .re2 (#v002) of a box with periodic BCs between the x-min and x-max faces,
    wall, inlet and outlet BCs and a user labelled (MSH) BC on the other faces. The type
    fields are blank padded, as the NEKTON tools write them.
    """
    x0, x1, y0, y1, z0, z1 = bounds
    nx, ny, nz = nel
    nelv = nx * ny * nz
    gx = np.linspace(x0, x1, nx + 1)
    gy = np.linspace(y0, y1, ny + 1)
    gz = np.linspace(z0, z1, nz + 1)
    hdr = f"#v002{nelv:9d}{3:3d}{nelv:9d} hdr".ljust(80).encode("ascii")
    el_dt = np.dtype([("rg", "<f8"), ("x", "<f8", (8,)), ("y", "<f8", (8,)), ("z", "<f8", (8,))])
    bc_dt = np.dtype([("e", "<f8"), ("f", "<f8"), ("d", "<f8", (5,)), ("t", "S8")])
    elems = np.zeros(nelv, dtype=el_dt)
    for e in range(nelv):
        ez, ey, ex = e // (nx * ny), (e // nx) % ny, e % nx
        for sl in range(8):
            elems["x"][e, sl] = gx[ex + VERTEX_IJK[sl, 0]]
            elems["y"][e, sl] = gy[ey + VERTEX_IJK[sl, 1]]
            elems["z"][e, sl] = gz[ez + VERTEX_IJK[sl, 2]]
    bcs = []
    # re2 face numbering: RE2_TO_NEKO_FACET maps re2 face k+1 -> neko facet.
    neko_to_re2 = {int(RE2_TO_NEKO_FACET[k]): k + 1 for k in range(6)}
    for e in range(nelv):
        ez, ey, ex = e // (nx * ny), (e // nx) % ny, e % nx
        if ex == 0:
            partner = 1 + (nx - 1) + ey * nx + ez * nx * ny
            bcs.append((e + 1, neko_to_re2[1], (partner, neko_to_re2[2], 0, 0, 0), b"P"))
        if ex == nx - 1:
            partner = 1 + 0 + ey * nx + ez * nx * ny
            bcs.append((e + 1, neko_to_re2[2], (partner, neko_to_re2[1], 0, 0, 0), b"P"))
        if ey == 0:
            bcs.append((e + 1, neko_to_re2[3], (0, 0, 0, 0, 0), b"W"))
        if ey == ny - 1:
            bcs.append((e + 1, neko_to_re2[4], (0, 0, 0, 0, 0), b"v"))
        if ez == 0:
            bcs.append((e + 1, neko_to_re2[5], (0, 0, 0, 0, 7), b"MSH"))
        if ez == nz - 1:
            bcs.append((e + 1, neko_to_re2[6], (0, 0, 0, 0, 0), b"O"))
    bc_arr = np.array([(e, f, d, t.ljust(8)) for e, f, d, t in bcs], dtype=bc_dt)
    with open(fname, "wb") as f:
        f.write(hdr)
        np.float32(6.54321).tofile(f)
        elems.tofile(f)
        np.array([0.0], dtype="<f8").tofile(f)  # no curves
        np.array([float(bc_arr.size)], dtype="<f8").tofile(f)
        bc_arr.tofile(f)
    return nelv, bc_arr.size


def read_fld_mesh(fname):
    """Read the mesh of a field file, distributed over the ranks."""
    msh = Mesh(comm, create_connectivity=False)
    pynekread(fname, comm, data_dtype=np.single, msh=msh)
    return msh

#==============================================================================


def test_registry():

    assert detect_format("a/b/mesh.re2") == "re2"
    assert detect_format("mesh.nmsh") == "nmsh"
    assert detect_format("field0.f00012") == "fld"
    assert detect_format("field.fld") == "fld"
    with pytest.raises(ValueError):
        detect_format("mesh.txt")

    keys = {c.key for c in conversions()}
    assert {("re2", "nmsh"), ("re2", "fld"), ("nmsh", "fld")} <= keys
    assert get_conversion("re2", "nmsh").func is re2_to_nmsh
    assert get_conversion("nmsh", "fld").func is mesh_to_fld
    assert not get_conversion("re2", "nmsh").parallel
    assert get_conversion("nmsh", "fld").parallel
    with pytest.raises(ValueError):
        get_conversion("nmsh", "re2")

    # Registering the same pair twice or an unknown format is an error
    with pytest.raises(ValueError):
        register(get_conversion("re2", "nmsh"))
    with pytest.raises(ValueError):
        register(Conversion("re2", "vtk", lambda *a, **k: None, "unknown"))

    # A new conversion is picked up by convert()
    calls = []
    func = lambda i, o, comm=None, **k: calls.append((i, o, k))
    register(Conversion("fld", "re2", func, "test", parallel=True))
    try:
        with pytest.raises(FileNotFoundError):
            convert("missing0.f00000", "out.re2", comm=comm)
        convert(HEMI_NMSH, "out.re2", comm=comm, source="fld")
        assert calls == [(HEMI_NMSH, "out.re2", {})]
    finally:
        del _CONVERSIONS[("fld", "re2")]


def test_convert_argument_errors(shared_tmp):

    out = str(shared_tmp / "hemi0.f00000")
    with pytest.raises(TypeError):
        convert(HEMI_NMSH, out, comm=comm)  # order is required
    with pytest.raises(TypeError):
        convert(HEMI_NMSH, out, comm=comm, order=3, bogus=1)
    with pytest.raises(ValueError):
        convert(HEMI_NMSH, HEMI_NMSH, comm=comm, source="re2")  # output is the input
    with pytest.raises(ValueError):
        mesh_to_fld(HEMI_NMSH, out, order=0, comm=comm)
    with pytest.raises(ValueError):
        mesh_to_fld(HEMI_NMSH, out, order=3, wdsz=2, comm=comm)
    if comm.Get_size() > 1:
        with pytest.raises(RuntimeError):
            convert(HEMI_RE2, str(shared_tmp / "hemi.nmsh"), comm=comm)


@serial_only
def test_re2_to_nmsh(tmp_path):

    re2 = str(tmp_path / "box.re2")
    out = str(tmp_path / "box.nmsh")
    nelv, nbc = write_re2(re2, BOUNDS, NEL)

    nmsh = convert(re2, out, comm=comm)
    assert os.path.isfile(out)
    assert nmsh.nelv == nelv
    assert nmsh.periodic_zones.shape[0] == 2 * NEL[1] * NEL[2]
    assert nmsh.periodic_zones.shape[0] + nmsh.labelled_zones.shape[0] == nbc
    # user label 7 from MSH, then named labels W, v, O in order of appearance
    labels = np.unique(nmsh.labelled_zones["p_f"])
    assert set(labels.tolist()) == {7, 8, 9, 10}
    # Periodic merge leaves one point per box vertex, minus the merged x faces
    assert np.unique(nmsh.periodic_zones["g"]).size <= (NEL[1] + 1) * (NEL[2] + 1) * NEL[0]

    back = NmshMesh.from_file(out, comm=comm)
    back.validate(out)
    assert np.array_equal(back.elems, nmsh.elems)
    assert np.array_equal(back.zones, nmsh.zones)
    assert (min_jacobian(back.corner_coordinates) > 0).all()

    # Default output name and the periodic tolerance option
    nmsh2 = re2_to_nmsh(re2, periodic_tol=1e-9, comm=comm)
    assert os.path.isfile(out)
    assert np.array_equal(nmsh2.zones, nmsh.zones)


@serial_only
def test_re2_to_nmsh_matches_neko(tmp_path):

    # hemi.nmsh was written by Neko's Fortran rea2nbin from hemi.re2. The two files must agree
    # byte for byte except in the partner element and point ids of labelled zones, which Neko
    # leaves uninitialised and this converter writes as zeros.
    out = str(tmp_path / "hemi.nmsh")
    convert(HEMI_RE2, out, comm=comm)

    with open(out, "rb") as f1, open(HEMI_NMSH, "rb") as f2:
        mine, neko = f1.read(), f2.read()
    assert len(mine) == len(neko)

    nelv = int(np.frombuffer(mine[:4], "<i4")[0])
    off = 8 + nelv * EL_DT.itemsize
    assert mine[: off + 4] == neko[: off + 4]  # header and element section
    nzones = int(np.frombuffer(mine[off : off + 4], "<i4")[0])
    z_mine = np.frombuffer(mine[off + 4 : off + 4 + nzones * ZONE_DT.itemsize], ZONE_DT)
    z_neko = np.frombuffer(neko[off + 4 : off + 4 + nzones * ZONE_DT.itemsize], ZONE_DT)
    for key in ("e", "f", "p_f", "t"):
        assert np.array_equal(z_mine[key], z_neko[key])
    tail = off + 4 + nzones * ZONE_DT.itemsize
    assert mine[tail:] == neko[tail:]

    back = NmshMesh.from_file(out, comm=comm)
    back.validate(out)
    assert back.nelv == 2042 and back.labelled_zones.shape[0] == 1232


@pytest.mark.parametrize("order", [1, 4])
def test_mesh_to_fld(shared_tmp, order):

    lx = order + 1
    out_nmsh = str(shared_tmp / f"nmsh{order}0.f00000")
    out_re2 = str(shared_tmp / f"re2_{order}0.f00000")
    msh = convert(HEMI_NMSH, out_nmsh, comm=comm, order=order)
    convert(HEMI_RE2, out_re2, comm=comm, order=order)
    comm.Barrier()

    # Both inputs hold the same corners, so the field files are identical
    if comm.Get_rank() == 0:
        with open(out_nmsh, "rb") as f1, open(out_re2, "rb") as f2:
            assert f1.read() == f2.read()

    # The file holds the GLL points of the trilinear geometry and nothing else
    back = read_fld_mesh(out_nmsh)
    assert (back.lx, back.ly, back.lz) == (lx, lx, lx)
    assert back.glb_nelv == 2042
    assert back.nelv == msh.nelv
    assert np.allclose(back.x, msh.x, atol=1e-6) and np.allclose(back.z, msh.z, atol=1e-6)

    expected = NmshMesh.from_file(HEMI_NMSH, comm).to_sem_mesh(comm, lx=lx)
    assert np.allclose(back.y, expected.y, atol=1e-6)

    # Double precision coordinates are exact
    out_dp = str(shared_tmp / f"dp{order}0.f00000")
    mesh_to_fld(NmshMesh.from_file(HEMI_NMSH, comm), out_dp, order=order, wdsz=8, comm=comm)
    comm.Barrier()
    dp = Mesh(comm, create_connectivity=False)
    pynekread(out_dp, comm, data_dtype=np.double, msh=dp)
    assert np.array_equal(dp.x, expected.x)


def test_cli(shared_tmp):

    assert convert_cli(["--list"]) == 0

    out = str(shared_tmp / "cli0.f00000")
    assert convert_cli([HEMI_NMSH, out, "--order", "2"]) == 0
    comm.Barrier()
    assert read_fld_mesh(out).lx == 3
    assert convert_cli([HEMI_RE2, out, "--order", "2", "--wdsz", "8"]) == 0

    # Errors are reported, not raised
    assert convert_cli([HEMI_NMSH, out]) == 1  # missing --order
    assert convert_cli([HEMI_NMSH, str(shared_tmp / "cli.txt")]) == 1  # unknown format
    assert convert_cli([HEMI_NMSH, str(shared_tmp / "cli.re2")]) == 1  # no conversion
    assert convert_cli([str(shared_tmp / "missing.nmsh"), out, "--order", "2"]) == 1
    # An option of another conversion
    assert convert_cli([HEMI_RE2, str(shared_tmp / "cli.nmsh"), "--order", "2"]) == 1

    if comm.Get_size() == 1:
        re2 = str(shared_tmp / "cli.re2")
        write_re2(re2, BOUNDS, NEL)
        assert convert_cli([re2, str(shared_tmp / "cli.nmsh")]) == 0
        assert os.path.isfile(str(shared_tmp / "cli.nmsh"))
        # Explicit formats override the names
        assert convert_cli([re2, str(shared_tmp / "cli.bin"), "--to", "nmsh"]) == 0
    else:
        assert convert_cli([HEMI_RE2, str(shared_tmp / "cli.nmsh")]) == 1
