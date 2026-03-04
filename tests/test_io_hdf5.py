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
from pysemtools.io.wrappers import read_data, write_data

NoneType = type(None)

#==============================================================================

def test_io_hdf5():

    fname = "examples/data/points.hdf5"

    data = read_data(comm, fname, keys=["mass"], parallel_io=False, dtype=np.single, distributed_axis=2)

    volume = comm.allreduce(np.sum(data["mass"]), op=MPI.SUM)
    t1 = np.allclose(np.float32(volume), 0.007853981)

    fname_out = "examples/data/points_out.hdf5"
    write_data(comm, fname_out, data, parallel_io=False, dtype=np.single, distributed_axis=2)

    t2 = True
    if comm.Get_rank() == 0:
        data2 = read_data(comm, fname_out, keys=["mass"], parallel_io=False, dtype=np.single, distributed_axis=0)

        if data2["mass"].shape != (30,30, 80):
            t2 = False

    passed = np.all([t1, t2])

    assert passed

test_io_hdf5()