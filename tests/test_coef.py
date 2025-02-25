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
# Import relevant modules
from pysemtools.io.ppymech.neksuite import preadnek
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef

NoneType = type(None)

#==============================================================================

def test_coef_2d():

    # Read the original mesh data
    fname = 'examples/data/mixlay0.f00001'
    data     = preadnek(fname, comm)
    msh      = Mesh(comm, data = data)
    del data

    coef = Coef(msh, comm)

    # Test the derivation
    ## The derivatives for each direction their same direction should be equal to 1
    dxdx = coef.dudxyz(msh.x, coef.drdx, coef.dsdx)
    dydy = coef.dudxyz(msh.y, coef.drdy, coef.dsdy)

    t1 = np.allclose(dxdx,np.ones_like(dxdx))
    t2 = np.allclose(dydy,np.ones_like(dydy))

    # Test the mass matrix
    ## Calculate the Area
    xx = np.max(msh.x)
    yy = np.max(msh.y)
    area_1 = xx*yy
    area_2 = coef.glsum(coef.B, comm)
    ## Check if the areas are okay
    t3 = np.allclose(area_1, area_2)
    
    # Check if all tests passed
    passed = np.all([t1, t2, t3])

    assert passed

#==============================================================================

def test_coef_3d():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data     = preadnek(fname, comm)
    msh      = Mesh(comm, data = data)
    del data

    coef = Coef(msh, comm)

    # Test the derivation
    ## The derivatives for each direction their same direction should be equal to 1
    dxdx = coef.dudxyz(msh.x, coef.drdx, coef.dsdx, coef.dtdx)
    dydy = coef.dudxyz(msh.y, coef.drdy, coef.dsdy, coef.dtdy)
    dzdz = coef.dudxyz(msh.z, coef.drdz, coef.dsdz, coef.dtdz)

    t1 = np.allclose(dxdx,np.ones_like(dxdx))
    t2 = np.allclose(dydy,np.ones_like(dydy))
    t3 = np.allclose(dzdz,np.ones_like(dzdz))


    # Test the mass matrix
    ## Calculate the volume
    rr = np.max(msh.x)
    zz = np.max(msh.z)
    vol_1 = np.pi*rr**2*zz
    vol_2 = coef.glsum(coef.B, comm)
    ## Check if volumes are within 1 percent difference
    ## Should not expect them exactly equal due to the mesh quality not being so good
    t4 = np.allclose(vol_1, vol_2, rtol=1e-2)

    # Check if all tests passed
    passed = np.all([t1, t2, t3, t4])

    assert passed
