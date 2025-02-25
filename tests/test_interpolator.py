import os
import sys
import pytest
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try: 
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except ImportError:
    print('could not import torch')
device = 'cpu'
# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Import general modules
import numpy as np

# Import relevant modules
from pysemtools.interpolation.mesh_to_mesh import PRefiner
from pysemtools.interpolation.interpolator import Interpolator
from pysemtools.interpolation.point_interpolator.single_point_legendre_interpolator import LegendreInterpolator as element_interpolator_c
from pysemtools.io.ppymech.neksuite import preadnek
from pysemtools.datatypes.msh import Mesh as msh_c
from pysemtools.comm.router import Router

NoneType = type(None)
rt = Router(comm)

#==============================================================================

def test_single_point_interpolator():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data     = preadnek(fname, comm)
    msh      = msh_c(comm, data = data)
    del data

    # Create a refined mesh
    n_new = 3
    pref = PRefiner(n_old = msh.lx, n_new = n_new)
    msh_ref = pref.create_refined_mesh(comm, msh = msh)

    # Instance an interpolator for the refined mesh to know the exact rst coordinates
    ei_ref = element_interpolator_c(n_new)

    # For an element this is true
    exact_r = ei_ref.x_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_s = ei_ref.y_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_t = ei_ref.z_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 

    interpolate_sem_mesh = False

    if interpolate_sem_mesh:

        # Get the points to find
        probes = np.zeros((msh_ref.x.size, 3))
        probes_rst_exact = np.zeros((msh_ref.x.size, 3))
        point = 0
        for e in range(msh_ref.nelv):
            for k in range(0, msh_ref.lz):
                for j in range(0, msh_ref.ly):
                    for i in range(0, msh_ref.lx):
                        probes[point, 0] = msh_ref.x[e, k, j, i]
                        probes[point, 1] = msh_ref.y[e, k, j, i]
                        probes[point, 2] = msh_ref.z[e, k, j, i]
                        probes_rst_exact[point, 0] = exact_r[k, j, i]
                        probes_rst_exact[point, 1] = exact_s[k, j, i]
                        probes_rst_exact[point, 2] = exact_t[k, j, i]
                        point = point + 1

    else:
        import pysemtools.interpolation.utils as interp_utils
        import pysemtools.interpolation.pointclouds as pcs

        # Create a polar mesh
        nn = msh_ref.x.size
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
            probes = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])
        else:
            probes = 1

    # Instance the interpolator
    itp = Interpolator(msh.x, msh.y, msh.z, probes, comm, progress_bar = False, point_interpolator_type='single_point_legendre')

    # Set up the global tree
    itp.set_up_global_tree(comm, find_points_comm_pattern="point_to_point", global_tree_type="rank_bbox")

    # Scatter the probes to all ranks
    itp.scatter_probes_from_io_rank(0, comm)

    # Find the points
    itp.find_points(comm, find_points_comm_pattern='point_to_point', elem_percent_expansion = 0.01)

    # Peform needed redistributions
    itp.gather_probes_to_io_rank(0, comm)
    itp.redistribute_probes_to_owners_from_io_rank(0, comm)

    # Now interpolate xyz to check if the process is correct
    my_interpolated_fields = np.zeros((itp.my_probes.shape[0], 3), dtype = np.double)
    if comm.Get_rank() == 0:
        interpolated_fields = np.zeros((probes.shape[0], 3), dtype = np.double)
    else:
        interpolated_fields = None
    # Set the time
    my_interpolated_fields[:, 0] =  itp.interpolate_field_from_rst(msh.x)[:]
    my_interpolated_fields[:, 1] =  itp.interpolate_field_from_rst(msh.y)[:]
    my_interpolated_fields[:, 2] =  itp.interpolate_field_from_rst(msh.z)[:]

    # Gather in rank zero for processing
    root = 0
    sendbuf = my_interpolated_fields.reshape((my_interpolated_fields.size))
    recvbuf, _ = rt.gather_in_root(sendbuf, root, np.double)
    
    if type(recvbuf) != NoneType:
        tmp = recvbuf.reshape((int(recvbuf.size/(3)), 3))
        interpolated_fields[itp.sort_by_rank] = tmp
            
    t1 = np.allclose(interpolated_fields, probes)

    passed = np.all([t1])

    assert passed

#==============================================================================

def test_multiple_point_interpolator_numpy():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data     = preadnek(fname, comm)
    msh      = msh_c(comm, data = data)
    del data

    # Create a refined mesh
    n_new = 3
    pref = PRefiner(n_old = msh.lx, n_new = n_new)
    msh_ref = pref.create_refined_mesh(comm, msh = msh)

    # Instance an interpolator for the refined mesh to know the exact rst coordinates
    ei_ref = element_interpolator_c(n_new)

    # For an element this is true
    exact_r = ei_ref.x_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_s = ei_ref.y_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_t = ei_ref.z_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 

    interpolate_sem_mesh = False

    if interpolate_sem_mesh:

        # Get the points to find
        probes = np.zeros((msh_ref.x.size, 3))
        probes_rst_exact = np.zeros((msh_ref.x.size, 3))
        point = 0
        for e in range(msh_ref.nelv):
            for k in range(0, msh_ref.lz):
                for j in range(0, msh_ref.ly):
                    for i in range(0, msh_ref.lx):
                        probes[point, 0] = msh_ref.x[e, k, j, i]
                        probes[point, 1] = msh_ref.y[e, k, j, i]
                        probes[point, 2] = msh_ref.z[e, k, j, i]
                        probes_rst_exact[point, 0] = exact_r[k, j, i]
                        probes_rst_exact[point, 1] = exact_s[k, j, i]
                        probes_rst_exact[point, 2] = exact_t[k, j, i]
                        point = point + 1

    else:
        import pysemtools.interpolation.utils as interp_utils
        import pysemtools.interpolation.pointclouds as pcs

        # Create a polar mesh
        nn = msh_ref.x.size
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
            probes = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])
        else:
            probes = 1

    use_torch = False
    if not use_torch:
        max_pts = 128
        interpolator_type = 'multiple_point_legendre_numpy' 
    else:
        max_pts = probes.shape[0]
        interpolator_type = 'multiple_point_legendre_torch' 

    t_itp = Interpolator(msh.x, msh.y, msh.z, probes, comm, progress_bar = False, point_interpolator_type=interpolator_type,  max_pts = max_pts, max_elems = 1)  

    # Set up the global tree
    t_itp.set_up_global_tree(comm, find_points_comm_pattern="point_to_point", global_tree_type="rank_bbox")

    # Scatter the probes to all ranks
    t_itp.scatter_probes_from_io_rank(0, comm)

    # Find the points
    t_itp.find_points(comm, find_points_comm_pattern='point_to_point', elem_percent_expansion = 0.01)

    # Peform needed redistributions
    t_itp.gather_probes_to_io_rank(0, comm)
    t_itp.redistribute_probes_to_owners_from_io_rank(0, comm)

    # Now interpolate xyz to check if the process is correct
    my_interpolated_fields = np.zeros((t_itp.my_probes.shape[0], 3), dtype = np.double)
    if comm.Get_rank() == 0:
        interpolated_fields = np.zeros((probes.shape[0], 3), dtype = np.double)
    else:
        interpolated_fields = None
    # Set the time
    start_time = MPI.Wtime()
    my_interpolated_fields[:, 0] =  t_itp.interpolate_field_from_rst(msh.x)[:]
    my_interpolated_fields[:, 1] =  t_itp.interpolate_field_from_rst(msh.y)[:]
    my_interpolated_fields[:, 2] =  t_itp.interpolate_field_from_rst(msh.z)[:]
    print('tensor_interpolator: Time to interpolate: {}'.format(MPI.Wtime() - start_time))

    # Gather in rank zero for processing
    root = 0
    sendbuf = my_interpolated_fields.reshape((my_interpolated_fields.size))
    recvbuf, _ = rt.gather_in_root(sendbuf, root, np.double)
    
    if type(recvbuf) != NoneType:
        tmp = recvbuf.reshape((int(recvbuf.size/(3)), 3))
        interpolated_fields[t_itp.sort_by_rank] = tmp


    t1 = np.allclose(interpolated_fields, probes)

    passed = np.all([t1])

    assert passed

#==============================================================================

def test_multiple_point_interpolator_torch():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data     = preadnek(fname, comm)
    msh      = msh_c(comm, data = data)
    del data

    # Create a refined mesh
    n_new = 3
    pref = PRefiner(n_old = msh.lx, n_new = n_new)
    msh_ref = pref.create_refined_mesh(comm, msh = msh)

    # Instance an interpolator for the refined mesh to know the exact rst coordinates
    ei_ref = element_interpolator_c(n_new)

    # For an element this is true
    exact_r = ei_ref.x_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_s = ei_ref.y_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_t = ei_ref.z_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 

    interpolate_sem_mesh = False

    if interpolate_sem_mesh:

        # Get the points to find
        probes = np.zeros((msh_ref.x.size, 3))
        probes_rst_exact = np.zeros((msh_ref.x.size, 3))
        point = 0
        for e in range(msh_ref.nelv):
            for k in range(0, msh_ref.lz):
                for j in range(0, msh_ref.ly):
                    for i in range(0, msh_ref.lx):
                        probes[point, 0] = msh_ref.x[e, k, j, i]
                        probes[point, 1] = msh_ref.y[e, k, j, i]
                        probes[point, 2] = msh_ref.z[e, k, j, i]
                        probes_rst_exact[point, 0] = exact_r[k, j, i]
                        probes_rst_exact[point, 1] = exact_s[k, j, i]
                        probes_rst_exact[point, 2] = exact_t[k, j, i]
                        point = point + 1

    else:
        import pysemtools.interpolation.utils as interp_utils
        import pysemtools.interpolation.pointclouds as pcs

        # Create a polar mesh
        nn = msh_ref.x.size
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
            probes = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])
        else:
            probes = 1

    use_torch = True
    if not use_torch:
        max_pts = 128
        interpolator_type = 'multiple_point_legendre_numpy' 
    else:
        max_pts = probes.shape[0]
        interpolator_type = 'multiple_point_legendre_torch' 

    t_itp = Interpolator(msh.x, msh.y, msh.z, probes, comm, progress_bar = False, point_interpolator_type=interpolator_type,  max_pts = max_pts, max_elems = 1)  

    # Set up the global tree
    t_itp.set_up_global_tree(comm, find_points_comm_pattern="point_to_point", global_tree_type="rank_bbox")

    # Scatter the probes to all ranks
    t_itp.scatter_probes_from_io_rank(0, comm)

    # Find the points
    t_itp.find_points(comm, find_points_comm_pattern='point_to_point', elem_percent_expansion = 0.01)

    # Peform needed redistributions
    t_itp.gather_probes_to_io_rank(0, comm)
    t_itp.redistribute_probes_to_owners_from_io_rank(0, comm)

    # Now interpolate xyz to check if the process is correct
    my_interpolated_fields = np.zeros((t_itp.my_probes.shape[0], 3), dtype = np.double)
    if comm.Get_rank() == 0:
        interpolated_fields = np.zeros((probes.shape[0], 3), dtype = np.double)
    else:
        interpolated_fields = None
    # Set the time
    start_time = MPI.Wtime()
    my_interpolated_fields[:, 0] =  t_itp.interpolate_field_from_rst(msh.x)[:]
    my_interpolated_fields[:, 1] =  t_itp.interpolate_field_from_rst(msh.y)[:]
    my_interpolated_fields[:, 2] =  t_itp.interpolate_field_from_rst(msh.z)[:]
    print('tensor_interpolator: Time to interpolate: {}'.format(MPI.Wtime() - start_time))

    # Gather in rank zero for processing
    root = 0
    sendbuf = my_interpolated_fields.reshape((my_interpolated_fields.size))
    recvbuf, _ = rt.gather_in_root(sendbuf, root, np.double)
    
    if type(recvbuf) != NoneType:
        tmp = recvbuf.reshape((int(recvbuf.size/(3)), 3))
        interpolated_fields[t_itp.sort_by_rank] = tmp


    t1 = np.allclose(interpolated_fields, probes)

    passed = np.all([t1])

    assert passed

#==============================================================================

@pytest.mark.skipif(device == "cpu", reason="Interpolation with autograd has only been properly tested on GPUs")
def test_multiple_point_interpolator_torch_autograd():

    # Read the original mesh data
    fname = 'examples/data/rbc0.f00001'
    data     = preadnek(fname, comm)
    msh      = msh_c(comm, data = data)
    del data

    # Create a refined mesh
    n_new = 3
    pref = PRefiner(n_old = msh.lx, n_new = n_new)
    msh_ref = pref.create_refined_mesh(comm, msh = msh)

    # Instance an interpolator for the refined mesh to know the exact rst coordinates
    ei_ref = element_interpolator_c(n_new)

    # For an element this is true
    exact_r = ei_ref.x_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_s = ei_ref.y_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
    exact_t = ei_ref.z_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 

    interpolate_sem_mesh = False

    if interpolate_sem_mesh:

        # Get the points to find
        probes = np.zeros((msh_ref.x.size, 3))
        probes_rst_exact = np.zeros((msh_ref.x.size, 3))
        point = 0
        for e in range(msh_ref.nelv):
            for k in range(0, msh_ref.lz):
                for j in range(0, msh_ref.ly):
                    for i in range(0, msh_ref.lx):
                        probes[point, 0] = msh_ref.x[e, k, j, i]
                        probes[point, 1] = msh_ref.y[e, k, j, i]
                        probes[point, 2] = msh_ref.z[e, k, j, i]
                        probes_rst_exact[point, 0] = exact_r[k, j, i]
                        probes_rst_exact[point, 1] = exact_s[k, j, i]
                        probes_rst_exact[point, 2] = exact_t[k, j, i]
                        point = point + 1

    else:
        import pysemtools.interpolation.utils as interp_utils
        import pysemtools.interpolation.pointclouds as pcs

        # Create a polar mesh
        nn = msh_ref.x.size
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
            probes = interp_utils.transform_from_array_to_list(nx,ny,nz,[x, y, z])
        else:
            probes = 1

    use_torch = True
    use_autograd = True
    if not use_torch:
        max_pts = 128
        interpolator_type = 'multiple_point_legendre_numpy' 
    else:
        max_pts = probes.shape[0]
        interpolator_type = 'multiple_point_legendre_torch' 

    t_itp = Interpolator(msh.x, msh.y, msh.z, probes, comm, progress_bar = False, point_interpolator_type=interpolator_type,  max_pts = max_pts, max_elems = 1, use_autograd = use_autograd)  

    # Set up the global tree
    t_itp.set_up_global_tree(comm, find_points_comm_pattern="point_to_point", global_tree_type="rank_bbox")

    # Scatter the probes to all ranks
    t_itp.scatter_probes_from_io_rank(0, comm)

    # Find the points
    t_itp.find_points(comm, find_points_comm_pattern='point_to_point', elem_percent_expansion = 0.01)

    # Peform needed redistributions
    t_itp.gather_probes_to_io_rank(0, comm)
    t_itp.redistribute_probes_to_owners_from_io_rank(0, comm)

    # Now interpolate xyz to check if the process is correct
    my_interpolated_fields = np.zeros((t_itp.my_probes.shape[0], 3), dtype = np.double)
    if comm.Get_rank() == 0:
        interpolated_fields = np.zeros((probes.shape[0], 3), dtype = np.double)
    else:
        interpolated_fields = None
    # Set the time
    start_time = MPI.Wtime()
    my_interpolated_fields[:, 0] =  t_itp.interpolate_field_from_rst(msh.x)[:]
    my_interpolated_fields[:, 1] =  t_itp.interpolate_field_from_rst(msh.y)[:]
    my_interpolated_fields[:, 2] =  t_itp.interpolate_field_from_rst(msh.z)[:]
    print('tensor_interpolator: Time to interpolate: {}'.format(MPI.Wtime() - start_time))

    # Gather in rank zero for processing
    root = 0
    sendbuf = my_interpolated_fields.reshape((my_interpolated_fields.size))
    recvbuf, _ = rt.gather_in_root(sendbuf, root, np.double)
    
    if type(recvbuf) != NoneType:
        tmp = recvbuf.reshape((int(recvbuf.size/(3)), 3))
        interpolated_fields[t_itp.sort_by_rank] = tmp


    t1 = np.allclose(interpolated_fields, probes)

    passed = np.all([t1])

    assert passed

test_multiple_point_interpolator_numpy()
test_multiple_point_interpolator_torch()