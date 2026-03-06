'''
Be sure to set a proper output folder in the pipeline script:

Either manually:
options.ExtractsOutputDirectory = "/home/adperez/software/pySEMTools/examples/data/"
options.GlobalTrigger = "TimeStep"
options.CatalystLiveTrigger = "TimeStep"

Or through the paraview options when saving the catalyst state file.

Run this with:
mpirun -np 4 pvbatch --sym catalyst_insitu.py

'''

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from pysemtools.datatypes import Mesh, FieldRegistry
from pysemtools.io.ppymech import pynekread
from pysemtools.io.catalyst import CatalystSession

if __name__ == "__main__":
    # Read SEM data
    msh = Mesh(comm)
    fld = FieldRegistry(comm)
    pynekread("../../examples/data/rbc0.f00001", comm, data_dtype=np.single, msh=msh, fld=fld)

    # Initialize catalyst
    cs = CatalystSession(comm, "pipeline.py", "rbc00001.vtkhdf", 
                         implementation_name="paraview", 
                         implementation_path="/home/adperez/software/experimental/paraview_build/lib/catalyst")

    # Set the mesh
    cs.set_mesh(msh.x, msh.y, msh.z, cell_type="hex")

    # Set the fields for this step
    cs.set_field(fld.registry)

    # Execute the catalyst pipeline for 3 steps
    cs.execute(timestep=0, time_value=1.0)

    # Finalize
    cs.finalize()


