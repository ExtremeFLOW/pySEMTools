try: 
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    have_torch = True
except ImportError:
    print('could not import torch')
    have_torch = False

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
    if have_torch:
        point_int_l = ["single_point_legendre", "multiple_point_legendre_numpy", "multiple_point_legendre_torch"]
    else:
        point_int_l = ["single_point_legendre", "multiple_point_legendre_numpy"]
    global_tree_type_l = ["rank_bbox", "domain_binning"]
    local_data_structure_l = ["kdtree", "rtree", "hashtable"]
    find_points_iterative = [[False], [True, 1]]

    for point_int in point_int_l:
        for global_tree_type in global_tree_type_l:
            for find_points_it in find_points_iterative:
                for local_data_structucture in local_data_structure_l:
                    # Create the probes object
                    log.write("warning", f"Creating probes with point_int = {point_int} and global_tree_type = {global_tree_type} and find_points_it = {find_points_it}")
                    log.tic()
                    probes = Probes(comm, probes=xyz, msh=msh, point_interpolator_type=point_int, find_points_comm_pattern="point_to_point", global_tree_type=global_tree_type,
                                    global_tree_nbins=9, find_points_iterative=find_points_it, local_data_structure=local_data_structucture)    

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
    #point_int_l = ["single_point_legendre", "multiple_point_legendre_numpy", "multiple_point_legendre_torch"]
    if have_torch:
        point_int_l = ["multiple_point_legendre_torch"]
    else:
        point_int_l = ["multiple_point_legendre_numpy"]

    global_tree_type_l = ["rank_bbox", "domain_binning"]

    for point_int in point_int_l:
        for global_tree_type in global_tree_type_l:
            # Create the probes object
            log.write("warning", f"Creating probes with point_int = {point_int} and global_tree_type = {global_tree_type}")
            log.tic()
            probes = Probes(comm, probes=xyz, msh=msh, point_interpolator_type=point_int, find_points_comm_pattern="point_to_point", global_tree_type=global_tree_type,
                            global_tree_nbins=9)    

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

def test_probes_gap_between_elements():
    """
    Regression test: a probe that falls in the micro gap between two elements.

    Meshes stored in single precision can have shared-face nodes that differ by one
    float32 ulp between the two neighbouring elements (each element carries its own
    copy of the face nodes). A probe located in such a gap is strictly outside every
    element and can only be accepted through the test pattern fallback. That fallback
    must be evaluated in double precision and be independent of the magnitude of the
    coordinates: at |x| ~ 3e3 the float32 spacing of x**2 + y**2 + z**2 is 1.0 and at
    |x| ~ 3e6 the float64 spacing is 2e-3, both far above the default tolerance.
    The same points must be found normally (error code 1) when the slack allowed
    outside the reference element (find_points_rst_tol) covers the gap.

    Note: with PYSEMTOOLS_INTERPOLATION_DTYPE=single the probe coordinates are rounded
    to float32, the probe in the gap collapses onto the element face and is found
    normally, so in that configuration this test only checks that nothing breaks.
    """
    from pysemtools.interpolation.point_interpolator.single_point_helper_functions import GLL_pwts

    lx = 8
    xi = np.sort(GLL_pwts(lx)[0])
    h = 5.0

    # (mesh dtype, position of the two elements, gap between them)
    # float32 spacing at x ~ 3000 is 2.44e-4: use exactly one ulp.
    # In double precision use a gap of 1e-6 relative to the element size.
    cases = [(np.single, 3000.0, None), (np.double, 3.0e6, 1e-6 * h)]

    point_interpolators = ["single_point_legendre", "multiple_point_legendre_numpy"]
    if have_torch:
        point_interpolators.append("multiple_point_legendre_torch")

    passed = True
    for ddtype, x0, gap in cases:

        # Two hexahedral elements side by side in x, shape (nelv, lz, ly, lx)
        xa = x0 + h * (1 + xi) / 2
        xb = x0 + h + h * (1 + xi) / 2
        yz = h * (1 + xi) / 2
        X = np.zeros((2, lx, lx, lx))
        Y = np.zeros((2, lx, lx, lx))
        Z = np.zeros((2, lx, lx, lx))
        X[0, :, :, :] = xa.reshape(1, 1, lx)
        X[1, :, :, :] = xb.reshape(1, 1, lx)
        Y[:, :, :, :] = yz.reshape(1, 1, lx, 1)
        Z[:, :, :, :] = yz.reshape(1, lx, 1, 1)
        X = X.astype(ddtype)
        Y = Y.astype(ddtype)
        Z = Z.astype(ddtype)

        # Element 1 carries its own copy of the shared face, shifted to the right
        if gap is None:
            X[1, :, :, 0] = np.nextafter(X[1, :, :, 0], ddtype(np.inf))
        else:
            X[1, :, :, 0] = X[1, :, :, 0] + ddtype(gap)
        gap = float(X[1, 0, 0, 0]) - float(X[0, 0, 0, -1])
        assert gap > 0

        msh = Mesh(comm, create_connectivity=False, x=X, y=Y, z=Z)

        # One probe in the gap, one well inside each element
        if comm.Get_rank() == 0:
            xyz = np.array(
                [
                    [float(X[0, 0, 0, -1]) + gap / 2, h / 2, h / 2],
                    [x0 + h / 4, h / 3, h / 5],
                    [x0 + 7 * h / 4, h / 3, h / 5],
                ]
            )
        else:
            xyz = None

        # A linear field is reproduced exactly by the interpolant,
        # also when extrapolating by a fraction of a ulp outside the element
        fld = (X.astype(np.double) + 2.0 * Y + 3.0 * Z).astype(np.double)

        for point_int in point_interpolators:
            for rst_tol in [np.finfo(np.single).eps, 1e-4]:
                probes = Probes(
                    comm,
                    probes=xyz,
                    msh=msh,
                    point_interpolator_type=point_int,
                    write_coords=False,
                    find_points_rst_tol=rst_tol,
                )
                probes.interpolate_from_field_list(0.0, [fld], comm, write_data=False)

                if comm.Get_rank() == 0:
                    err_code = probes.itp.err_code
                    test_pattern = probes.itp.test_pattern
                    reference = xyz[:, 0] + 2.0 * xyz[:, 1] + 3.0 * xyz[:, 2]
                    interpolated = probes.interpolated_fields[:, 1]
                    error = np.max(np.abs(interpolated - reference) / np.abs(reference))
                    log.write(
                        "info",
                        f"{np.dtype(ddtype).name}, x0={x0:.0e}, {point_int}, rst_tol={rst_tol:.1e}: "
                        f"err_code = {err_code}, test_pattern[gap point] = {test_pattern[0]:.3e}, "
                        f"max relative error = {error:.3e}",
                    )
                    # The point in the gap must never be marked as not found (error code 0)
                    # and must be interpolated correctly
                    passed = passed and err_code[0] != 0
                    passed = passed and np.all(err_code[1:] == 1)
                    passed = passed and (err_code[0] == 1 or test_pattern[0] < 1e-4)
                    passed = passed and error < 1e-6
                    # With a slack that covers the gap the point is found normally
                    if rst_tol > 1e-5:
                        passed = passed and err_code[0] == 1

    passed = comm.bcast(passed, root=0)
    assert passed

# =============================================================================

test_probes_msh_single()
test_probes_msh_double()
test_probes_gap_between_elements()
