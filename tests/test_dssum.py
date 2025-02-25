from mpi4py import MPI
import numpy as np
import os

comm = MPI.COMM_WORLD

from pysemtools.io.ppymech.neksuite import pynekread, pynekwrite
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.datatypes.msh_connectivity import MeshConnectivity


# First do one section in rank 0
split_comm = comm.Split(comm.Get_rank(), comm.Get_rank())

def test_dssum():
        
    fname_single_rank = "examples/data/rbc0.f00002"

    # First test that the results are fine in one rank, using old methods
    if comm.Get_rank() == 0:

        # Read the data        
        fname = "examples/data/rbc0.f00001"
        msh = Mesh(split_comm, create_connectivity=True)
        fld = FieldRegistry(split_comm)
        pynekread(fname, split_comm, data_dtype=np.single, msh=msh, fld=fld)

        # Init coef and connectivity
        coef = Coef(msh, split_comm)
        conn = MeshConnectivity(split_comm, msh)

        # Create a random field
        np.random.seed(0)
        tst = np.random.random(msh.x.shape)
        
        # Do the sum with new
        xx = conn.dssum(field = tst, msh = msh, average="multiplicity")

        # Do the sum with old
        xx_old = coef.dssum(np.copy(tst), msh)


        print(np.max(np.abs(xx)-np.abs(xx_old)))

        passed_single_rank = np.allclose(xx, xx_old, atol=1e-7)

        print(passed_single_rank)

        fld.add_field(split_comm, field_name="w", field=xx, dtype=xx.dtype)

        pynekwrite(fname_single_rank, split_comm, msh=msh, fld=fld)
        
    else:
        passed_single_rank = True

    comm.Barrier()

    # Now check if things work with all ranks
    # Read the data        
    fname = "examples/data/rbc0.f00001"
    msh = Mesh(comm, create_connectivity=True)
    fld = FieldRegistry(comm)
    pynekread(fname, comm, data_dtype=np.single, msh=msh, fld=fld)

    # Init connectivity
    conn = MeshConnectivity(comm, msh, use_hashtable=True)

    # Create a random field. It needs to be the full size of the field. 
    # Otherwise all ranks will get the same and it will not match the result form one rank
    np.random.seed(0)
    if comm.Get_size() == 1:
        tst2 = np.random.random(msh.x.shape)
    elif comm.Get_size() == 2:
        tst_ = np.random.random((600, msh.lz, msh.ly, msh.lx))
        if comm.Get_rank() == 0:
            tst2 = tst_[:300]
        else:
            tst2 = tst_[300:]
    elif comm.Get_size() == 4:
        tst_ = np.random.random((600, msh.lz, msh.ly, msh.lx))
        if comm.Get_rank() == 0:
            tst2 = tst_[:150]
        elif comm.Get_rank() == 1:
            tst2 = tst_[150:300]
        elif comm.Get_rank() == 2:
            tst2 = tst_[300:450]
        else:
            tst2 = tst_[450:] 
    else:
        raise ValueError("This test only works for 1, 2 or 4 ranks")

    # Do the sum
    xx2 = conn.dssum(field = tst2, msh = msh, average="multiplicity")

    # Read the written file from one rank
    fld_1_rank = FieldRegistry(comm)
    pynekread(fname_single_rank, comm, data_dtype=np.single, fld=fld_1_rank)
    
    print(np.max(np.abs(xx2)-np.abs(fld_1_rank.registry["w"])))

    # Check if the results are the same
    passed = np.allclose(xx2, fld_1_rank.registry["w"], atol=1e-7)

    print(passed) 
    
    # Delete any extra files
    if comm.Get_rank()==0: 
        os.system(f"rm {fname_single_rank}")

    assert np.all([passed_single_rank, passed])


test_dssum()
