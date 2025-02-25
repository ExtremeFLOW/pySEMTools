# Import general modules
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import numpy as np
import matplotlib.pyplot as plt

# Import MPI
from mpi4py import MPI #equivalent to the use of MPI_init() in C

# Split communicator for MPI - MPMD
worldcomm = MPI.COMM_WORLD
worldrank = worldcomm.Get_rank()
worldsize = worldcomm.Get_size()
col = 1
comm = worldcomm.Split(col,worldrank)
rank = comm.Get_rank()
size = comm.Get_size()

# Open input file to see path
f = open ("inputs.json", "r")
params_file = json.loads(f.read())
f.close()

# Read the POD inputs
pod_number_of_snapshots = params_file["number_of_snapshots"]
pod_fields = params_file["fields"]
number_of_pod_fields = len(pod_fields)
pod_batch_size = params_file["batch_size"]
pod_keep_modes = params_file["keep_modes"]
pod_write_modes = params_file["write_modes"]

# Import IO helper functions
from pysemtools.io.utils import get_fld_from_ndarray, IoPathData
# Import modules for reading and writing
from pysemtools.io.ppymech.neksuite import preadnek, pwritenek
# Import the data types
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field
from pysemtools.datatypes.utils import create_hexadata_from_msh_fld
# Import types asociated with POD
from pysemtools.rom.pod import POD
from pysemtools.rom.io_help import IoHelp

# Start time
start_time = MPI.Wtime()

# Read the data paths from the input file
mesh_data = IoPathData(params_file["IO"]["mesh_data"])
field_data = IoPathData(params_file["IO"]["field_data"])

# Instance the POD object
pod = POD(comm, number_of_modes_to_update = pod_keep_modes, global_updates = True, auto_expand = False)

# Initialize the mesh file
path     = mesh_data.dataPath
casename = mesh_data.casename
index    = mesh_data.index
fname    = path+casename+'0.f'+str(index).zfill(5)
data     = preadnek(fname, comm)
msh      = Mesh(comm, data = data)
del data

# Initialize coef to get the mass matrix
coef = Coef(msh, comm)
bm = coef.B

# Instance io helper that will serve as buffer for the snapshots
ioh = IoHelp(comm, number_of_fields = number_of_pod_fields, batch_size = pod_batch_size, field_size = bm.size)

# Put the mass matrix in the appropiate format (long 1d array)
mass_list = []
for i in range(0, number_of_pod_fields):
    mass_list.append(np.copy(np.sqrt(bm)))
ioh.copy_fieldlist_to_xi(mass_list)
ioh.bm1sqrt[:,:] = np.copy(ioh.xi[:,:])

j = 0
while j < pod_number_of_snapshots:

    # Recieve the data from fortran
    path     = field_data.dataPath
    casename = field_data.casename
    index    = field_data.index
    fname=path+casename+'0.f'+str(index + j).zfill(5)
    fld_data = preadnek(fname, comm)

    # Get the data in field format
    fld = Field(comm, data = fld_data)

    # Get the required fields
    u = fld.fields["vel"][0]
    v = fld.fields["vel"][1]
    w = fld.fields["vel"][2]

    # Put the snapshot data into a column array
    ioh.copy_fieldlist_to_xi([u, v, w])

    # Load the column array into the buffer
    ioh.load_buffer(scale_snapshot = True)

    # Update POD modes
    if ioh.update_from_buffer:
        pod.update(comm, buff = ioh.buff[:,:(ioh.buffer_index)])

    j += 1


# Check if there is information in the buffer that should be taken in case the loop exit without flushing
if ioh.buffer_index > ioh.buffer_max_index:
    ioh.log.write("info","All snapshots where properly included in the updates")
else: 
    ioh.log.write("warning","Last loaded snapshot to buffer was: "+repr(ioh.buffer_index-1))
    ioh.log.write("warning","The buffer updates when it is full to position: "+repr(ioh.buffer_max_index))
    ioh.log.write("warning","Data must be updated now to not lose anything,  Performing an update with data in buffer ")
    pod.update(comm, buff = ioh.buff[:,:(ioh.buffer_index)])

# Scale back the modes
pod.scale_modes(comm, bm1sqrt = ioh.bm1sqrt, op = "div")

# Rotate local modes back to global, This only enters in effect if global_update = false
pod.rotate_local_modes_to_global(comm)

# Write the data out
for j in range(0, pod_write_modes):

    if (j+1) < pod.u_1t.shape[1]:

        ## Split the snapshots into the proper fields
        field_list1d = ioh.split_narray_to_1dfields(pod.u_1t[:,j])
        u_mode = get_fld_from_ndarray(field_list1d[0], msh.lx, msh.ly, msh.lz, msh.nelv)
        v_mode = get_fld_from_ndarray(field_list1d[1], msh.lx, msh.ly, msh.lz, msh.nelv)
        w_mode = get_fld_from_ndarray(field_list1d[2], msh.lx, msh.ly, msh.lz, msh.nelv)

        ## Create an empty field and update its metadata
        out_fld = Field(comm)
        out_fld.fields["scal"].append(u_mode)
        out_fld.fields["scal"].append(v_mode)
        out_fld.fields["scal"].append(w_mode)
        out_fld.update_vars()

        ## Create the hexadata to write out
        out_data = create_hexadata_from_msh_fld(msh = msh, fld = out_fld)

        ## Write out a file
        fname = "modes0.f"+str(j).zfill(5)
        pwritenek("./"+fname,out_data, comm)
        if comm.Get_rank() == 0: print("Wrote file: " + fname)

# Write the singular values and vectors
if comm.Get_rank() == 0:
    np.save("singular_values", pod.d_1t)
    print("Wrote signular values")
    np.save("right_singular_vectors", pod.vt_1t)
    print("Wrote right signular values")

# End time
end_time = MPI.Wtime()
# Print the time
if comm.Get_rank() == 0:
    print("Time to complete: ", end_time - start_time)
