# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

import os

# Import general modules
import numpy as np
# Import relevant modules
from pysemtools.io.ppymech.neksuite import pynekread, pynekwrite
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.datatypes.msh import Mesh
from pysemtools.postprocessing.statistics.space_averaging import *
from pysemtools.comm.router import Router
from pysemtools.monitoring.logger import Logger
from pysemtools.postprocessing.statistics.uq import NOBM

rt = Router(comm)
logger = Logger(comm=comm, module_name="uncertainty quantification")
rel_tol = 0.01

def test_NOBM():

    dtype = np.single

    # Read the mesh data
    fname = 'examples/data/rbc0.f00001'    
    # Read the original mesh data
    msh = Mesh(comm, create_connectivity=False)
    pynekread(fname, comm, data_dtype = dtype, msh = msh)

    # Create fiction mean fields
    fld = FieldRegistry(comm)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 10
    u2 = np.ones_like(msh.x) * 20
    u3 = np.ones_like(msh.x) * 30
    s1 = np.ones_like(msh.x) * 40

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 20
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(0).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)
    
    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 13
    u2 = np.ones_like(msh.x) * 23
    u3 = np.ones_like(msh.x) * 33
    s1 = np.ones_like(msh.x) * 43

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 20
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(1).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 17
    u2 = np.ones_like(msh.x) * 27
    u3 = np.ones_like(msh.x) * 37
    s1 = np.ones_like(msh.x) * 47

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 20
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(2).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)

    # Now test it
    file_sequence = [ "mean_rbc0.f00000", "mean_rbc0.f00001", "mean_rbc0.f00002" ]
    field_names = ["vel_0", "vel_1", "vel_2", "scal_0"]
    output_field_names = ["u", "v", "w", "s0"]
    mean, var = NOBM(comm, file_sequence, field_names, output_field_names=output_field_names)

    t1 = np.allclose(mean["u"], np.ones_like(msh.x) * (10 + 13 + 17) / 3)
    t2 = np.allclose(mean["v"], np.ones_like(msh.x) * (20 + 23 + 27) / 3)
    t3 = np.allclose(mean["w"], np.ones_like(msh.x) * (30 + 33 + 37) / 3)
    t4 = np.allclose(mean["s0"], np.ones_like(msh.x) * (40 + 43 + 47) / 3)

    passed1 = np.all([t1, t2, t3, t4])

    t1 = np.allclose(var["u"], np.ones_like(msh.x) * (((10**2 + 13**2 + 17**2) / 3) - ((10 + 13 + 17) / 3)**2)/3)
    t2 = np.allclose(var["v"], np.ones_like(msh.x) * (((20**2 + 23**2 + 27**2) / 3) - ((20 + 23 + 27) / 3)**2)/3)
    t3 = np.allclose(var["w"], np.ones_like(msh.x) * (((30**2 + 33**2 + 37**2) / 3) - ((30 + 33 + 37) / 3)**2)/3)
    t4 = np.allclose(var["s0"], np.ones_like(msh.x) * (((40**2 + 43**2 + 47**2) / 3) - ((40 + 43 + 47) / 3)**2)/3)

    passed2 = np.all([t1, t2, t3, t4])

    passed = np.all([passed1, passed2])

    if comm.Get_rank() == 0:    
        # delete the wirtten files
        os.system("rm ./mean_rbc0.f00000")
        os.system("rm ./mean_rbc0.f00001")
        os.system("rm ./mean_rbc0.f00002")
        os.system("rm ./mean_rbc_index.json")
        os.system("rm ./batches.json")
        os.system("rm ./batch_mean_rbc0.f00000")
    comm.Barrier()

    print(passed)

    assert passed

test_NOBM()