from mpi4py import MPI #equivalent to the use of MPI_init() in C
comm = MPI.COMM_WORLD

import numpy as np
# Import modules for reading and writing
from pysemtools.io.ppymech.neksuite import pynekread, pynekwrite
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import Field
# Import types asociated with interpolation
from pysemtools.interpolation.probes import Probes
import pysemtools.interpolation.utils as interp_utils
import pysemtools.interpolation.pointclouds as pcs
from pysemtools.monitoring.logger import Logger
from pysemtools.monitoring.memory_monitor import MemoryMonitor
from pysemtools.interpolation.mesh_to_mesh import PRefiner

log = Logger(comm=comm, module_name="main")
log.write("info", "Starting execution")

def test_probes_msh_single():

    mm = MemoryMonitor()

    ddtype = np.single
    # Prepare the fields to keep the data
    msh_og = Mesh(comm=comm, create_connectivity=False)

    # Read the data
    fname = "examples/data/rbc0.f00001"
    pynekread(fname, comm, data_dtype=ddtype, msh=msh_og)

    # Create a Coarser mesh to make it easier
    n_new = 3
    pref = PRefiner(n_old = msh_og.lx, n_new = n_new, dtype = ddtype)
    msh = pref.create_refined_mesh(comm, msh = msh_og)

    log.write("info", "Creating mesh to interpolate")
    log.tic()
    # Create a polar mesh
    nn = msh.x.size
    nx = int(nn**(1/3))
    ny = int(nn**(1/3))
    nz = int(nn**(1/3))

    # Choose the boundaries of the interpolation mesh
    # boundaries
    x_bbox = [0, 0.05]
    y_bbox = [0, 2*np.pi]
    z_bbox = [0 , 1]

    # Generate the points in 1D
    start_time = MPI.Wtime()
    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")

    # Create 3D arrays
    r, th, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x = r*np.cos(th)
    y = r*np.sin(th)

    # Create a list with the points
    if comm.Get_rank() == 0:    
        xyz = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])
    else:
        xyz = 1
    log.write("info", "Interpolating mesh created")
    log.toc()

    # Create the probes object
    tlist = []
    point_int_l = ["single_point_legendre", "multiple_point_legendre_numpy", "multiple_point_legendre_torch"]
    global_tree_type_l = ["rank_bbox", "domain_binning"]
    find_points_iterative = [[False], [True, 1]]

    for point_int in point_int_l:
        for global_tree_type in global_tree_type_l:
            for find_points_it in find_points_iterative:
                # Create the probes object
                log.write("warning", f"Creating probes with point_int = {point_int} and global_tree_type = {global_tree_type} and find_points_it = {find_points_it}")
                log.tic()
                probes = Probes(comm, probes=xyz, msh=msh, point_interpolator_type=point_int, find_points_comm_pattern="point_to_point", global_tree_type=global_tree_type, find_points_iterative=find_points_it)    

                # Interpolate the data
                probes.interpolate_from_field_list(0, [msh.x, msh.y, msh.z], comm, write_data=False)

                passed = False
                if comm.Get_rank() == 0: 

                    passed = np.allclose(probes.interpolated_fields[:,1:], xyz, atol=1e-7)

                log.write("warning", f"Test passed = {passed}")
                log.toc()
                tlist.append(passed)
    

    passed = np.all(tlist)

    #mm.object_memory_usage(comm, probes, "Probes")
    #mm.object_memory_usage_per_attribute(comm, probes, "Probes")
    
    #mm.object_memory_usage(comm, probes.itp, "interpolator")
    #mm.object_memory_usage_per_attribute(comm, probes.itp, "interpolator")
    
    #if comm.Get_rank() == 0:
    #    for key in mm.object_report.keys():
    #        mm.report_object_information(comm, key)

    #log.write("info", "Verify that the interpolator just references the mesh")
    #value_int = probes.itp.x[100,0,0,0]
    #value_mesh = msh.x[100,0,0,0]
    #log.write("info", f" 1 value in the interpolator = {value_int}, and in mesh ={value_mesh}")
    #probes.itp.x[100,0,0,0] = 1
    #value_mesh = msh.x[100,0,0,0]
    #log.write("info", f" Assing it to be 1 in interpolator -> New value in the mesh = {value_mesh}")

    assert passed

# =============================================================================

def test_probes_msh_double():

    mm = MemoryMonitor()

    ddtype = np.double
    # Prepare the fields to keep the data
    msh_og = Mesh(comm=comm, create_connectivity=False)

    # Read the data
    fname = "examples/data/rbc0.f00001"
    pynekread(fname, comm, data_dtype=ddtype, msh=msh_og)

    # Create a Coarser mesh to make it easier
    n_new = 3
    pref = PRefiner(n_old = msh_og.lx, n_new = n_new, dtype = ddtype)
    msh = pref.create_refined_mesh(comm, msh = msh_og)

    log.write("info", "Creating mesh to interpolate")
    log.tic()
    # Create a polar mesh
    nn = msh.x.size
    nx = int(nn**(1/3))
    ny = int(nn**(1/3))
    nz = int(nn**(1/3))

    # Choose the boundaries of the interpolation mesh
    # boundaries
    x_bbox = [0, 0.05]
    y_bbox = [0, 2*np.pi]
    z_bbox = [0 , 1]

    # Generate the points in 1D
    start_time = MPI.Wtime()
    x_1d = pcs.generate_1d_arrays(x_bbox, nx, mode="equal")
    y_1d = pcs.generate_1d_arrays(y_bbox, ny, mode="equal")
    z_1d = pcs.generate_1d_arrays(z_bbox, nz, mode="equal")

    # Create 3D arrays
    r, th, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x = r*np.cos(th)
    y = r*np.sin(th)

    # Create a list with the points
    if comm.Get_rank() == 0:    
        xyz = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])
    else:
        xyz = 1
    log.write("info", "Interpolating mesh created")
    log.toc()

    # Create the probes object
    tlist = []
    point_int_l = ["single_point_legendre", "multiple_point_legendre_numpy", "multiple_point_legendre_torch"]
    point_int_l = ["multiple_point_legendre_torch"]
    global_tree_type_l = ["rank_bbox", "domain_binning"]

    for point_int in point_int_l:
        for global_tree_type in global_tree_type_l:
            # Create the probes object
            log.write("warning", f"Creating probes with point_int = {point_int} and global_tree_type = {global_tree_type}")
            log.tic()
            probes = Probes(comm, probes=xyz, msh=msh, point_interpolator_type=point_int, find_points_comm_pattern="point_to_point", global_tree_type=global_tree_type)    

            # Interpolate the data
            probes.interpolate_from_field_list(0, [msh.x, msh.y, msh.z], comm, write_data=False)

            passed = False
            if comm.Get_rank() == 0: 

                passed = np.allclose(probes.interpolated_fields[:,1:], xyz, atol=1e-7)

            log.write("warning", f"Test passed = {passed}")
            log.toc()
            tlist.append(passed)
 

    passed = np.all(tlist)

    #mm.object_memory_usage(comm, probes, "Probes")
    #mm.object_memory_usage_per_attribute(comm, probes, "Probes")
    
    #mm.object_memory_usage(comm, probes.itp, "interpolator")
    #mm.object_memory_usage_per_attribute(comm, probes.itp, "interpolator")
    
    #if comm.Get_rank() == 0:
    #    for key in mm.object_report.keys():
    #        mm.report_object_information(comm, key)

    #log.write("info", "Verify that the interpolator just references the mesh")
    #value_int = probes.itp.x[100,0,0,0]
    #value_mesh = msh.x[100,0,0,0]
    #log.write("info", f" 1 value in the interpolator = {value_int}, and in mesh ={value_mesh}")
    #probes.itp.x[100,0,0,0] = 1
    #value_mesh = msh.x[100,0,0,0]
    #log.write("info", f" Assing it to be 1 in interpolator -> New value in the mesh = {value_mesh}")

    assert passed

# =============================================================================

test_probes_msh_single()
test_probes_msh_double()