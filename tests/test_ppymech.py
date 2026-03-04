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
from pysemtools.io.ppymech.neksuite import preadnek, pwritenek
from pymech.neksuite import readnek, writenek

NoneType = type(None)

#==============================================================================

def test_read_double():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data_pynek     = preadnek(fname, comm)
    data_pymech = readnek(fname)
    
    for e in range(data_pymech.nel):

        t1 = np.allclose(data_pynek.elem[e].pos[0], data_pymech.elem[e].pos[0])
        t2 = np.allclose(data_pynek.elem[e].pos[1], data_pymech.elem[e].pos[1])
        t3 = np.allclose(data_pynek.elem[e].pos[2], data_pymech.elem[e].pos[2])
        
        t4 = np.allclose(data_pynek.elem[e].vel[0], data_pymech.elem[e].vel[0])
        t5 = np.allclose(data_pynek.elem[e].vel[1], data_pymech.elem[e].vel[1])
        t6 = np.allclose(data_pynek.elem[e].vel[2], data_pymech.elem[e].vel[2])
        
        t7 = np.allclose(data_pynek.elem[e].pres[0], data_pymech.elem[e].pres[0])
        t8 = np.allclose(data_pynek.elem[e].temp[0], data_pymech.elem[e].temp[0])

        passed = np.all([t1, t2, t3, t4, t5, t6, t7, t8])

        if not passed:
            break
      
    assert passed

#==============================================================================

def test_read_single():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data_pynek     = preadnek(fname, comm, data_dtype=np.single)
    data_pymech = readnek(fname, dtype=np.single)
    
    for e in range(data_pymech.nel):

        t1 = np.allclose(data_pynek.elem[e].pos[0], data_pymech.elem[e].pos[0])
        t2 = np.allclose(data_pynek.elem[e].pos[1], data_pymech.elem[e].pos[1])
        t3 = np.allclose(data_pynek.elem[e].pos[2], data_pymech.elem[e].pos[2])
        
        t4 = np.allclose(data_pynek.elem[e].vel[0], data_pymech.elem[e].vel[0])
        t5 = np.allclose(data_pynek.elem[e].vel[1], data_pymech.elem[e].vel[1])
        t6 = np.allclose(data_pynek.elem[e].vel[2], data_pymech.elem[e].vel[2])
        
        t7 = np.allclose(data_pynek.elem[e].pres[0], data_pymech.elem[e].pres[0])
        t8 = np.allclose(data_pynek.elem[e].temp[0], data_pymech.elem[e].temp[0])

        passed = np.all([t1, t2, t3, t4, t5, t6, t7, t8])

        if not passed:
            break
      
    assert passed

#==============================================================================

def test_write_double():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data_pynek     = preadnek(fname, comm)
    
    fname2 = 'examples/data/rbc0.f00002'
    pwritenek(fname2, data_pynek, comm)
    data_pymech = readnek(fname2)
    
    for e in range(data_pymech.nel):

        t1 = np.allclose(data_pynek.elem[e].pos[0], data_pymech.elem[e].pos[0])
        t2 = np.allclose(data_pynek.elem[e].pos[1], data_pymech.elem[e].pos[1])
        t3 = np.allclose(data_pynek.elem[e].pos[2], data_pymech.elem[e].pos[2])
        
        t4 = np.allclose(data_pynek.elem[e].vel[0], data_pymech.elem[e].vel[0])
        t5 = np.allclose(data_pynek.elem[e].vel[1], data_pymech.elem[e].vel[1])
        t6 = np.allclose(data_pynek.elem[e].vel[2], data_pymech.elem[e].vel[2])
        
        t7 = np.allclose(data_pynek.elem[e].pres[0], data_pymech.elem[e].pres[0])
        t8 = np.allclose(data_pynek.elem[e].temp[0], data_pymech.elem[e].temp[0])

        passed = np.all([t1, t2, t3, t4, t5, t6, t7, t8])

        if not passed:
            break
      
    assert passed

#==============================================================================

def test_write_single():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data_pynek     = preadnek(fname, comm, data_dtype=np.single)
    
    fname2 = 'examples/data/rbc0.f00002'
    pwritenek(fname2, data_pynek, comm)
    data_pymech = readnek(fname2, dtype=np.single)
    
    for e in range(data_pymech.nel):

        t1 = np.allclose(data_pynek.elem[e].pos[0], data_pymech.elem[e].pos[0])
        t2 = np.allclose(data_pynek.elem[e].pos[1], data_pymech.elem[e].pos[1])
        t3 = np.allclose(data_pynek.elem[e].pos[2], data_pymech.elem[e].pos[2])
        
        t4 = np.allclose(data_pynek.elem[e].vel[0], data_pymech.elem[e].vel[0])
        t5 = np.allclose(data_pynek.elem[e].vel[1], data_pymech.elem[e].vel[1])
        t6 = np.allclose(data_pynek.elem[e].vel[2], data_pymech.elem[e].vel[2])
        
        t7 = np.allclose(data_pynek.elem[e].pres[0], data_pymech.elem[e].pres[0])
        t8 = np.allclose(data_pynek.elem[e].temp[0], data_pymech.elem[e].temp[0])

        passed = np.all([t1, t2, t3, t4, t5, t6, t7, t8])

        if not passed:
            break
      
    assert passed

