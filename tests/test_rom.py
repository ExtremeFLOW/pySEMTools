# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

import os
import sys

# Import general modules
import numpy as np
# Import relevant modules
from pysemtools.rom.pod import POD
from pysemtools.rom.io_help import IoHelp

def test_POD():

    if comm.Get_size() > 1:
        sys.exit("This test is not parallelized")

    # Generate a toy random matrix
    np.random.seed(0)
    field_size = 100
    snapshots = 50
    S = np.random.rand(field_size, snapshots)
    bm = np.ones((field_size, 1))*2

    # Perform the SVD
    u_ref, d_ref, vt_ref = np.linalg.svd(S*np.sqrt(bm), full_matrices=False)
    u_ref = u_ref/np.sqrt(bm)

    # Do it with our methods
    ioh = IoHelp(comm, number_of_fields=1, batch_size=1, field_size=field_size, module_name='test')
    pod = POD(comm, number_of_modes_to_update=snapshots)

    # Initialize the mass matrix
    mass_list = []
    for i in range(0, 1):
        mass_list.append(np.copy(np.sqrt(bm)))
    ioh.copy_fieldlist_to_xi(mass_list)
    ioh.bm1sqrt[:, :] = np.copy(ioh.xi[:, :])

    # Perform reading and updates
    j = 0
    while j < S.shape[1]:

        # Load the snapshot data
        fld_data = [S[:, j]]

        # Put the snapshot data into a column array
        ioh.copy_fieldlist_to_xi(fld_data)

        # Load the column array into the buffer
        ioh.load_buffer(scale_snapshot=True)

        # Update POD modes
        if ioh.update_from_buffer:
            pod.update(comm, buff=ioh.buff[:, : (ioh.buffer_index)])

        j += 1

    # Check if there is information in the buffer that should be taken in case the loop exit without flushing
    if ioh.buffer_index > ioh.buffer_max_index:
        ioh.log.write("info", "All snapshots where properly included in the updates")
    else:
        ioh.log.write(
            "warning",
            "Last loaded snapshot to buffer was: " + repr(ioh.buffer_index - 1),
        )
        ioh.log.write(
            "warning",
            "The buffer updates when it is full to position: "
            + repr(ioh.buffer_max_index),
        )
        ioh.log.write(
            "warning",
            "Data must be updated now to not lose anything,  Performing an update with data in buffer ",
        )
        pod.update(comm, buff=ioh.buff[:, : (ioh.buffer_index)])

    # Scale back the modes
    pod.scale_modes(comm, bm1sqrt=ioh.bm1sqrt, op="div")

    # Rotate local modes back to global, This only enters in effect if global_update = false
    pod.rotate_local_modes_to_global(comm)

    # Check if the results are okay
    t1 = np.allclose(pod.d_1t, d_ref)
    t2 = np.allclose(pod.u_1t**2, u_ref**2)
    t3 = np.allclose(pod.vt_1t**2, vt_ref**2)
    passed = np.all([t1, t2, t3])

    print(passed)

    assert passed

test_POD()