from mpi4py import MPI
import numpy as np

from pysemtools.datatypes.field import FieldRegistry
from pysemtools.datatypes.msh import Mesh
from pysemtools.interpolation.wrappers import interpolate_along_line


comm = MPI.COMM_WORLD


def _single_element_cube(comm):
    nodes = np.array([-1.0, 1.0], dtype=np.double)
    x = np.zeros((1, 2, 2, 2), dtype=np.double)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    for k in range(2):
        for j in range(2):
            for i in range(2):
                x[0, k, j, i] = nodes[i]
                y[0, k, j, i] = nodes[j]
                z[0, k, j, i] = nodes[k]

    return Mesh(comm, x=x, y=y, z=z, create_connectivity=False)


def test_interpolate_along_line_uses_probes():
    msh = _single_element_cube(comm)

    fld = FieldRegistry(comm)
    fld.add_field(
        comm,
        field_name="sum_xyz",
        field=msh.x + msh.y + msh.z,
        dtype=np.double,
    )

    start_point = np.array([-0.5, -0.25, 0.0])
    end_point = np.array([0.5, 0.25, 0.0])
    n_points = 5

    line_points, interpolated_data = interpolate_along_line(
        comm,
        msh,
        fld,
        start_point,
        end_point,
        n_points,
        fields_to_interpolate=["sum_xyz"],
    )

    if comm.Get_rank() != 0:
        assert line_points is None
        assert interpolated_data == {}
        return

    expected_xyz = np.linspace(start_point, end_point, n_points)
    expected_d = np.linspace(0.0, np.linalg.norm(end_point - start_point), n_points)

    assert np.allclose(line_points[:, :3], expected_xyz)
    assert np.allclose(line_points[:, 3], expected_d)
    assert set(interpolated_data.keys()) == {"sum_xyz"}
    assert np.allclose(interpolated_data["sum_xyz"], np.sum(expected_xyz, axis=1))
