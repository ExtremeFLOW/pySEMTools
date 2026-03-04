import json
import numpy as np
import os

from ...monitoring.logger import Logger
from ...datatypes.msh import Mesh
from ...datatypes.field import FieldRegistry
from ...datatypes.coef import Coef
from ...io.ppymech.neksuite import preadnek, pwritenek, pynekread, pynekwrite


def tke_budgets_cartesian(comm, msh_fname="", mean_fname="", stats_fname=""):
    """
    tke budgets for incompressible NS in Cartesian coordinates.

    Get the tke for the statistics files.


    Parameters
    ----------
    comm : MPI.COMM
        MPI communicator.
    msh_fname : str
        File name of a field file that contains the mesh. Include path.
    mean_fname : str
        File name of the mean field file. Include path.
    stats_fname : str
        File name of the statistics field file. Include path.

    Returns
    -------
    None
        Writes the data to a file.
    """

    logger = Logger(comm=comm, module_name="tke_budgets_cartesian")

    logger.write("info", f"Using mean field file: {mean_fname}")
    logger.write("info", f"Using statistics field file: {stats_fname}")

    # Read the mean field file
    msh = Mesh(comm, create_connectivity=False)
    fld = FieldRegistry(comm)

    # Read the data
    pynekread(msh_fname, comm, msh=msh, data_dtype=np.single)

    # Initialize coef
    coef = Coef(msh, comm)

    # Get reynolds stress
    logger.write("info", "Obtaining Reynolds stress tensor")
    logger.tic()
    fld.add_field(
        comm,
        field_name="U",
        file_type="fld",
        file_name=mean_fname,
        file_key="vel_0",
        dtype=np.single,
    )
    fld.add_field(
        comm,
        field_name="V",
        file_type="fld",
        file_name=mean_fname,
        file_key="vel_1",
        dtype=np.single,
    )
    fld.add_field(
        comm,
        field_name="W",
        file_type="fld",
        file_name=mean_fname,
        file_key="vel_2",
        dtype=np.single,
    )
    fld.add_field(
        comm,
        field_name="uu",
        file_type="fld",
        file_name=stats_fname,
        file_key="vel_0",
        dtype=np.single,
    )
    fld.add_field(
        comm,
        field_name="vv",
        file_type="fld",
        file_name=stats_fname,
        file_key="vel_1",
        dtype=np.single,
    )
    fld.add_field(
        comm,
        field_name="ww",
        file_type="fld",
        file_name=stats_fname,
        file_key="vel_2",
        dtype=np.single,
    )
    fld.add_field(
        comm,
        field_name="uv",
        file_type="fld",
        file_name=stats_fname,
        file_key="temp",
        dtype=np.single,
    )
    fld.add_field(
        comm,
        field_name="uw",
        file_type="fld",
        file_name=stats_fname,
        file_key="scal_0",
        dtype=np.single,
    )
    fld.add_field(
        comm,
        field_name="vw",
        file_type="fld",
        file_name=stats_fname,
        file_key="scal_1",
        dtype=np.single,
    )

    logger.write("info", "Calculating components of Reynolds stress tensor")
    logger.write("info", "Note that density is not multiplied in the calculation")

    u_u_ = fld.registry["uu"] - fld.registry["U"] * fld.registry["U"]
    v_v_ = fld.registry["vv"] - fld.registry["V"] * fld.registry["V"]
    w_w_ = fld.registry["ww"] - fld.registry["W"] * fld.registry["W"]
    u_v_ = fld.registry["uv"] - fld.registry["U"] * fld.registry["V"]
    u_w_ = fld.registry["uw"] - fld.registry["U"] * fld.registry["W"]
    v_w_ = fld.registry["vw"] - fld.registry["V"] * fld.registry["W"]

    # Write the data
    fname = "./rij0.f00000"
    logger.write("info", f"Writing components to {fname}")
    fld.clear()
    fld.add_field(comm, field_name="u_u_", field=u_u_, dtype=np.single)
    fld.add_field(comm, field_name="v_v_", field=v_v_, dtype=np.single)
    fld.add_field(comm, field_name="w_w_", field=w_w_, dtype=np.single)
    fld.add_field(comm, field_name="u_v_", field=u_v_, dtype=np.single)
    fld.add_field(comm, field_name="u_w_", field=u_w_, dtype=np.single)
    fld.add_field(comm, field_name="v_w_", field=v_w_, dtype=np.single)
    pynekwrite(fname, comm, msh=msh, fld=fld, wdsz=4, write_mesh=True)

    logger.write("info", "Obtaining Reynolds stress tensor: done")
    logger.toc()
