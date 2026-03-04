import json
import numpy as np
import os

from ...monitoring.logger import Logger
from mpi4py import MPI
from ...datatypes.msh import Mesh
from ...datatypes.field import FieldRegistry
from ...datatypes.coef import Coef
from ...io.ppymech.neksuite import preadnek, pwritenek, pynekread, pynekwrite


def generate_augmented_field(
    comm: MPI.Comm = None,
    msh: Mesh = None,
    fld: FieldRegistry = None,
    coef: Coef = None,
    dtype: np.dtype = np.single,
):
    """
    """

    # Initialize a new field registry
    stat_fld = FieldRegistry(comm, bckend=fld.bckend)

    stat_fld.add_field(comm, field_name="u", field=fld.registry["u"], dtype=dtype) #1
    stat_fld.add_field(comm, field_name="v", field=fld.registry["v"], dtype=dtype) 
    stat_fld.add_field(comm, field_name="w", field=fld.registry["w"], dtype=dtype)
    stat_fld.add_field(comm, field_name="p", field=fld.registry["p"], dtype=dtype) #4

    # Get products
    ## From here on, they will be added as scalars. Keep the neko order
    temp = fld.registry["p"] * fld.registry["p"]
    stat_fld.add_field(comm, field_name="t", field=temp, dtype=dtype) #5
    temp = fld.registry["u"] * fld.registry["u"]
    stat_fld.add_field(comm, field_name="uu", field=temp, dtype=dtype)
    temp = fld.registry["v"] * fld.registry["v"]
    stat_fld.add_field(comm, field_name="vv", field=temp, dtype=dtype)
    temp = fld.registry["w"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="ww", field=temp, dtype=dtype)
    temp = fld.registry["u"] * fld.registry["v"]
    stat_fld.add_field(comm, field_name="uv", field=temp, dtype=dtype)
    temp = fld.registry["u"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="uw", field=temp, dtype=dtype)
    temp = fld.registry["v"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="vw", field=temp, dtype=dtype) #11

    # Get triple products
    temp = fld.registry["u"] * fld.registry["u"] * fld.registry["u"]
    stat_fld.add_field(comm, field_name="uuu", field=temp, dtype=dtype) #12
    temp = fld.registry["v"] * fld.registry["v"] * fld.registry["v"]
    stat_fld.add_field(comm, field_name="vvv", field=temp, dtype=dtype) #13
    temp = fld.registry["w"] * fld.registry["w"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="www", field=temp, dtype=dtype) #14
    temp = fld.registry["u"] * fld.registry["u"] * fld.registry["v"]
    stat_fld.add_field(comm, field_name="uuv", field=temp, dtype=dtype) #15
    temp = fld.registry["u"] * fld.registry["u"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="uuw", field=temp, dtype=dtype) #16
    temp = fld.registry["u"] * fld.registry["v"] * fld.registry["v"]  
    stat_fld.add_field(comm, field_name="uvv", field=temp, dtype=dtype) #17
    temp = fld.registry["u"] * fld.registry["v"] * fld.registry["w"] 
    stat_fld.add_field(comm, field_name="uvw", field=temp, dtype=dtype) #18
    temp = fld.registry["v"] * fld.registry["v"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="vvw", field=temp, dtype=dtype) #19
    temp = fld.registry["u"] * fld.registry["w"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="uww", field=temp, dtype=dtype) #20
    temp = fld.registry["v"] * fld.registry["w"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="vww", field=temp, dtype=dtype) #21

    # Get quadruple products and presure products
    temp = fld.registry["u"] * fld.registry["u"] * fld.registry["u"] * fld.registry["u"]
    stat_fld.add_field(comm, field_name="uuuu", field=temp, dtype=dtype) #22
    temp = fld.registry["v"] * fld.registry["v"] * fld.registry["v"] * fld.registry["v"]
    stat_fld.add_field(comm, field_name="vvvv", field=temp, dtype=dtype) #23
    temp = fld.registry["w"] * fld.registry["w"] * fld.registry["w"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="wwww", field=temp, dtype=dtype) #24
    temp = fld.registry["p"] * fld.registry["p"] * fld.registry["p"]
    stat_fld.add_field(comm, field_name="ppp", field=temp, dtype=dtype) #25
    temp = fld.registry["p"] * fld.registry["p"] * fld.registry["p"] * fld.registry["p"]
    stat_fld.add_field(comm, field_name="pppp", field=temp, dtype=dtype) #26
    temp = fld.registry["p"] * fld.registry["u"]
    stat_fld.add_field(comm, field_name="pu", field=temp, dtype=dtype) #27
    temp = fld.registry["p"] * fld.registry["v"]
    stat_fld.add_field(comm, field_name="pv", field=temp, dtype=dtype) #28
    temp = fld.registry["p"] * fld.registry["w"]
    stat_fld.add_field(comm, field_name="pw", field=temp, dtype=dtype) #29

    # Get the gradient tensor
    dudx = coef.dudxyz(fld.registry["u"], coef.drdx, coef.dsdx, coef.dtdx)
    dudy = coef.dudxyz(fld.registry["u"], coef.drdy, coef.dsdy, coef.dtdy)
    dudz = coef.dudxyz(fld.registry["u"], coef.drdz, coef.dsdz, coef.dtdz)
    dvdx = coef.dudxyz(fld.registry["v"], coef.drdx, coef.dsdx, coef.dtdx)
    dvdy = coef.dudxyz(fld.registry["v"], coef.drdy, coef.dsdy, coef.dtdy)
    dvdz = coef.dudxyz(fld.registry["v"], coef.drdz, coef.dsdz, coef.dtdz)
    dwdx = coef.dudxyz(fld.registry["w"], coef.drdx, coef.dsdx, coef.dtdx)
    dwdy = coef.dudxyz(fld.registry["w"], coef.drdy, coef.dsdy, coef.dtdy)
    dwdz = coef.dudxyz(fld.registry["w"], coef.drdz, coef.dsdz, coef.dtdz)

    # Add pressure*gradient terms
    temp = fld.registry["p"] * dudx
    stat_fld.add_field(comm, field_name="pdudx", field=temp, dtype=dtype) #30
    temp = fld.registry["p"] * dudy
    stat_fld.add_field(comm, field_name="pdudy", field=temp, dtype=dtype) #31
    temp = fld.registry["p"] * dudz
    stat_fld.add_field(comm, field_name="pdudz", field=temp, dtype=dtype) #32
    temp = fld.registry["p"] * dvdx
    stat_fld.add_field(comm, field_name="pdvdx", field=temp, dtype=dtype) #33
    temp = fld.registry["p"] * dvdy
    stat_fld.add_field(comm, field_name="pdvdy", field=temp, dtype=dtype) #34
    temp = fld.registry["p"] * dvdz
    stat_fld.add_field(comm, field_name="pdvdz", field=temp, dtype=dtype) #35
    temp = fld.registry["p"] * dwdx
    stat_fld.add_field(comm, field_name="pdwdx", field=temp, dtype=dtype) #36
    temp = fld.registry["p"] * dwdy
    stat_fld.add_field(comm, field_name="pdwdy", field=temp, dtype=dtype) #37
    temp = fld.registry["p"] * dwdz
    stat_fld.add_field(comm, field_name="pdwdz", field=temp, dtype=dtype) #38
    
    # Now add strain rate entries
    temp = dudx * dudx + dudy * dudy + dudz * dudz
    stat_fld.add_field(comm, field_name="e11", field=temp, dtype=dtype) #39
    temp = dvdx * dvdx + dvdy * dvdy + dvdz * dvdz
    stat_fld.add_field(comm, field_name="e22", field=temp, dtype=dtype) #40
    temp = dwdx * dwdx + dwdy * dwdy + dwdz * dwdz
    stat_fld.add_field(comm, field_name="e33", field=temp, dtype=dtype) #41
    temp = dudx*dvdx + dudy*dvdy + dudz*dvdz
    stat_fld.add_field(comm, field_name="e12", field=temp, dtype=dtype) #42
    temp = dudx*dwdx + dudy*dwdy + dudz*dwdz
    stat_fld.add_field(comm, field_name="e13", field=temp, dtype=dtype) #43
    temp = dvdx*dwdx + dvdy*dwdy + dvdz*dwdz
    stat_fld.add_field(comm, field_name="e23", field=temp, dtype=dtype) #44

    return stat_fld