"""
Conversion of corner meshes to Nek5000 field files.

The GLL points of every element are evaluated from the trilinear map of its
eight corners at the requested polynomial order and written as the
coordinates of a field file without any data fields. Both the ``.re2`` and
the ``.nmsh`` mesh formats are supported, and the conversion runs in
parallel: the input is read distributed over the ranks and the field file
is written collectively.
"""

from mpi4py import MPI

from ..monitoring.logger import Logger
from ..datatypes.corner_mesh import CornerMesh
from ..datatypes.nmsh import NmshMesh
from ..datatypes.re2 import Re2Mesh
from ..datatypes.field import Field
from ..io.ppymech.neksuite import pynekwrite
from .registry import Conversion, Option, register, detect_format

__all__ = ["mesh_to_fld"]


def mesh_to_fld(mesh, fld_fname, order, wdsz=4, comm=None):
    """
    Write the GLL points of a corner mesh to a Nek5000 field file at
    arbitrary polynomial order.

    Curved edges are not applied: every element is written with the
    straight-sided trilinear geometry of its corners.

    Parameters
    ----------
    mesh : str or CornerMesh
        Input ``.re2`` or ``.nmsh`` file, or a mesh object such as
        :class:`pysemtools.datatypes.nmsh.NmshMesh`.
    fld_fname : str
        Output field file, for example ``mesh0.f00000``.
    order : int
        Polynomial order of the elements. The file holds ``order + 1`` GLL
        points per direction.
    wdsz : int, optional
        Word size of the coordinates in bytes, 4 or 8. Default is 4.
    comm : MPI.Comm, optional
        MPI communicator. Default is ``MPI.COMM_WORLD``.

    Returns
    -------
    Mesh
        The :class:`pysemtools.datatypes.msh.Mesh` that was written.

    Examples
    --------
    >>> from pysemtools.convert import mesh_to_fld
    >>> msh = mesh_to_fld("hemi.nmsh", "hemi0.f00000", order=5)
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if order < 1:
        raise ValueError(f"The polynomial order must be at least 1, got {order}")
    if wdsz not in (4, 8):
        raise ValueError(f"The word size must be 4 or 8, got {wdsz}")
    log = Logger(comm=comm, module_name="mesh_to_fld")
    log.tic()

    if isinstance(mesh, str):
        fmt = detect_format(mesh)
        if fmt == "nmsh":
            mesh = NmshMesh.from_file(mesh, comm)
        elif fmt == "re2":
            mesh = Re2Mesh.from_file(mesh, comm)
        else:
            raise ValueError(f"{mesh} is a {fmt} file, expected a re2 or nmsh mesh")
    elif not isinstance(mesh, CornerMesh):
        raise TypeError(f"Expected a file name or a corner mesh, got {type(mesh).__name__}")

    ncurves = int(mesh.curves.shape[0])
    if mesh.is_distributed:
        ncurves = comm.allreduce(ncurves, op=MPI.SUM)
    if ncurves > 0:
        log.write(
            "warning",
            f"{ncurves} curve records ignored: elements are written with straight edges",
        )

    lx = order + 1
    log.write("info", f"Evaluating the GLL points of {mesh.glb_nelv} elements at order {order}")
    msh = mesh.to_sem_mesh(comm, lx=lx)
    pynekwrite(fld_fname, comm, msh=msh, fld=Field(comm), wdsz=wdsz)
    log.toc()
    return msh


_OPTIONS = (
    Option("order", int, "polynomial order of the elements (order + 1 GLL points per direction)"),
    Option("wdsz", int, "word size of the coordinates in bytes", default="4", choices=(4, 8)),
)

for _source in ("re2", "nmsh"):
    register(
        Conversion(
            _source,
            "fld",
            mesh_to_fld,
            f"write the GLL points of a {_source} mesh to a Nek5000 field file without data",
            parallel=True,
            options=_OPTIONS,
        )
    )
