import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Import general modules
import numpy as np

# Import relevant modules
from pysemtools.interpolation.mesh_to_mesh import PRefiner
from pysemtools.interpolation.point_interpolator.single_point_legendre_interpolator import LegendreInterpolator as element_interpolator_c
from pysemtools.io.ppymech.neksuite import preadnek
from pysemtools.datatypes.msh import Mesh as msh_c

def test_sem_and_tensor_sem(n_new = 8, elem = 1, max_pts = 1, use_torch = False, verbose = False):

    # Read the original mesh data
    fname = '../examples/data/rbc0.f00001'
    data     = preadnek(fname, comm)
    msh      = msh_c(comm, data = data)
    del data

    # Create a refined mesh
    n_new = n_new
    pref = PRefiner(n_old = msh.lx, n_new = n_new)
    msh_ref = pref.create_refined_mesh(comm, msh = msh)

    # Create the element interpolator for both meshes
    ei = element_interpolator_c(msh.lx)
    ei_ref = element_interpolator_c(msh_ref.lx)

    if comm.Get_rank() == 0:

        # ======================================================#
        # Test interpolating all points within the same element #
        # ======================================================#


        print('Now testing: sem.py: element_interpolator_c')
        print('using only one element to perform all operations')

        # Now choose 1 element to interpolate
        elem = elem

        # Get the extact rst coordinates of the refined element
        exact_r = ei_ref.x_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
        exact_s = ei_ref.y_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 
        exact_t = ei_ref.z_e.reshape(msh_ref.lz, msh_ref.ly, msh_ref.lx) 

        # Allocate found coordinates
        found_r = np.zeros_like(exact_r)
        found_s = np.zeros_like(exact_s)
        found_t = np.zeros_like(exact_s)
        
        found_x = np.zeros_like(exact_r)
        found_y = np.zeros_like(exact_s)
        found_z = np.zeros_like(exact_s)
        
        # Find the rst coordinates of the refined element
        start_time = MPI.Wtime()
        for k in range(0, msh_ref.lz):
            for j in range(0, msh_ref.ly):
                for i in range(0, msh_ref.lx):
                    ei.project_element_into_basis(msh.x[elem,:,:,:], msh.y[elem,:,:,:], msh.z[elem,:,:,:])    
                    found_r[k,j,i], found_s[k,j,i], found_t[k,j,i] = ei.find_rst_from_xyz(msh_ref.x[elem,k,j,i], msh_ref.y[elem,k,j,i], msh_ref.z[elem,k,j,i])
        print('sem.py: Time to find rst: {}'.format(MPI.Wtime() - start_time))
        print('sem.py: The last point took: {} iterations'.format(ei.iterations))

        t1 = np.allclose(exact_r, found_r)
        t2 = np.allclose(exact_s, found_s)
        t3 = np.allclose(exact_t, found_t)
            
        passed = np.all([t1, t2, t3])
        
        if verbose:        
            print('sem.py: The r coordinates were found correctly: {}'.format(t1))
            print('sem.py: The s coordinates were found correctly: {}'.format(t2))
            print('sem.py: The t coordinates were found correctly: {}'.format(t3))

        if not passed:
            sys.exit('sem.py: The r, s, and t coordinates were not found correctly')
        else:
            print('sem.py: find_rst_form_xyz: passed')
            
        # Interpolate back to the xyz coordinates
        start_time = MPI.Wtime()
        for k in range(0, msh_ref.lz):
            for j in range(0, msh_ref.ly):
                for i in range(0, msh_ref.lx):
                    found_x[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.x[elem,:,:,:])
                    found_y[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.y[elem,:,:,:])
                    found_z[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.z[elem,:,:,:]) 
        print('sem.py: Time to interpolate: {}'.format(MPI.Wtime() - start_time))
        
        t1 = np.allclose(msh_ref.x[elem,:,:,:], found_x)
        t2 = np.allclose(msh_ref.y[elem,:,:,:], found_y)
        t3 = np.allclose(msh_ref.z[elem,:,:,:], found_z)
            
        passed = np.all([t1, t2, t3])
        
        if verbose:        
            print('sem.py: The x coordinates were found correctly: {}'.format(t1))
            print('sem.py: The y coordinates were found correctly: {}'.format(t2))
            print('sem.py: The z coordinates were found correctly: {}'.format(t3))

        if not passed:
            sys.exit('sem.py: The x, y, and z coordinates were not found correctly')
        else:
            print('sem.py: interpolate_field_at_rst: passed')

         
        # ========================================================================================#
        # Test interpolating all points within the same element using the tensor supported module #
        # ========================================================================================#
        
        print('=====================')
        print('Now testing: tensor_sem.py: element_interpolator_c')

        from pysemtools.interpolation.point_interpolator.multiple_point_interpolator_legendre_numpy import LegendreInterpolator as numpy_tensor_element_interpolator_c
        from pysemtools.interpolation.point_interpolator.multiple_point_interpolator_legendre_torch import LegendreInterpolator as torch_tensor_element_interpolator_c

        # Instance the tensor interpolator
        max_pts   = max_pts
        max_elems = 1
        use_torch = use_torch
        if not use_torch:
            tei = numpy_tensor_element_interpolator_c(msh.lx, max_pts=max_pts)
        else:
            tei = torch_tensor_element_interpolator_c(msh.lx, max_pts=max_pts)

        print('using torch: {}'.format(use_torch))
        if use_torch: print('Using device: {}'.format(device))

        # Allocate the data with the candidate elements per point
        rshp_x = np.zeros((max_pts, 1, msh.lz, msh.ly, msh.lx))
        rshp_y = np.zeros((max_pts, 1, msh.lz, msh.ly, msh.lx))
        rshp_z = np.zeros((max_pts, 1, msh.lz, msh.ly, msh.lx))

        # Project all the involved elements at the same time
        rshp_x[:,:,:,:,:] = msh.x[elem,:,:,:].reshape(1, 1, msh.lz, msh.ly, msh.lx)
        rshp_y[:,:,:,:,:] = msh.y[elem,:,:,:].reshape(1, 1, msh.lz, msh.ly, msh.lx)
        rshp_z[:,:,:,:,:] = msh.z[elem,:,:,:].reshape(1, 1, msh.lz, msh.ly, msh.lx)
        tei.project_element_into_basis(rshp_x, rshp_y, rshp_z)

        if not use_torch: 
            t1 = np.allclose(ei.x_e_hat, tei.x_e_hat[0,0,:,:])
            t2 = np.allclose(ei.y_e_hat, tei.y_e_hat[0,0,:,:])
            t3 = np.allclose(ei.z_e_hat, tei.z_e_hat[0,0,:,:])
        else:
            t1 = np.allclose(ei.x_e_hat, tei.x_e_hat[0,0,:,:].cpu().numpy())
            t2 = np.allclose(ei.y_e_hat, tei.y_e_hat[0,0,:,:].cpu().numpy())
            t3 = np.allclose(ei.z_e_hat, tei.z_e_hat[0,0,:,:].cpu().numpy())

        passed = np.all([t1, t2, t3])
        
        if verbose:        
            print('tensor_sem.py: The xhat projected correclty: {}'.format(t1))
            print('tensor_sem.py: The yhat projected correctly: {}'.format(t2))
            print('tensor_sem.py: The zhat prohectec correctly: {}'.format(t3))

        if not passed:
            sys.exit('tensor_sem.py: project_element_to_basis: failed')
        else:
            print('tensor_sem.py: project_element_to_basis: passed')


        # Reshape the points to the new format
        new_r = found_r.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
        new_s = found_s.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
        new_t = found_t.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
        new_x = found_x.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
        new_y = found_y.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)
        new_z = found_z.reshape(msh_ref.lz*msh_ref.ly*msh_ref.lx, 1, 1, 1)

        # Get the xyz coordinates from the rst coordinates with the tensor
        if not use_torch:
            xx = np.zeros_like(new_x)
            yy = np.zeros_like(new_y)
            zz = np.zeros_like(new_z)
        else:
            xx = torch.zeros(new_x.shape, dtype=torch.float64, device=device)
            yy = torch.zeros(new_y.shape, dtype=torch.float64, device=device)
            zz = torch.zeros(new_z.shape, dtype=torch.float64, device=device)

        total_pts = msh_ref.lz*msh_ref.ly*msh_ref.lx
        total_iterations = np.ceil(total_pts / max_pts)

        start_time = MPI.Wtime()
        for i in range(0, int(total_iterations)):
            start = i * max_pts
            end   = (i+1) * max_pts
            if end > total_pts:
                end = total_pts
            npoints = end - start
            tei.project_element_into_basis(rshp_x[:npoints,:,:,:,:], rshp_y[:npoints,:,:,:,:], rshp_z[:npoints,:,:,:,:])
            xx[start:end], yy[start:end], zz[start:end] = tei.get_xyz_from_rst(new_r[start:end], new_s[start:end], new_t[start:end], apply_1d_ops = True)
        print('tensor_sem.py: Time to get all xyz: {}'.format(MPI.Wtime() - start_time))
        
        if not use_torch: 
            t1 = np.allclose(new_x, xx)
            t2 = np.allclose(new_y, yy)
            t3 = np.allclose(new_z, zz)
        else:
            t1 = np.allclose(new_x, xx.cpu().numpy())
            t2 = np.allclose(new_y, yy.cpu().numpy())
            t3 = np.allclose(new_z, zz.cpu().numpy())

        passed = np.all([t1, t2, t3])
        
        if verbose:        
            print('tensor_sem.py: x calculated correctly from rst: {}'.format(t1))
            print('tensor_sem.py: y calculated correctly from rst: {}'.format(t2))
            print('tensor_sem.py: z calculated correctly from rst: {}'.format(t3))

        if not passed:
            sys.exit('tensor_sem.py: get_xyz_from_rst: failed')
        else:
            print('tensor_sem.py: get_xyz_from_rst: passed')

        # Check the gradients
        if max_pts == 1 and not use_torch:
            x1, y1, z1 = ei.get_xyz_from_rst(-0.3, 0.3, 0.4, apply_1d_ops = True)
            test_r = (np.ones((1))*-0.3).reshape((1, 1, 1, 1))
            test_s = (np.ones((1))*0.3).reshape((1, 1, 1, 1))
            test_t = (np.ones((1))*0.4).reshape((1, 1, 1, 1))
            x2, y2, z2 = tei.get_xyz_from_rst(test_r, test_s, test_t)
            print('tensor_sem.py: Jacobian calculated correctly: {}'.format(np.allclose(ei.jac, tei.jac[0,0,:,:])))

        # Get the xyz coordinates from the rst coordinates with the tensor
        if not use_torch:
            rr = np.zeros_like(new_r)
            ss = np.zeros_like(new_s)
            tt = np.zeros_like(new_t)
        else:
            rr = torch.zeros(new_r.shape, dtype=torch.float64, device=device)
            ss = torch.zeros(new_s.shape, dtype=torch.float64, device=device)
            tt = torch.zeros(new_t.shape, dtype=torch.float64, device=device)

        total_pts = msh_ref.lz*msh_ref.ly*msh_ref.lx
        total_iterations = np.ceil(total_pts / max_pts)

        #Check the search function
        start_time = MPI.Wtime()
        for i in range(0, int(total_iterations)):
            start = i * max_pts
            end   = (i+1) * max_pts
            if end > total_pts:
                end = total_pts
            npoints = end - start
            tei.project_element_into_basis(rshp_x[:npoints,:,:,:,:], rshp_y[:npoints,:,:,:,:], rshp_z[:npoints,:,:,:,:])
            rr[start:end], ss[start:end], tt[start:end] = tei.find_rst_from_xyz(new_x[start:end], new_y[start:end], new_z[start:end])
        print('tensor_sem.py: Time to find rst: {}'.format(MPI.Wtime() - start_time))
        print('tensor_sem.py: Last run took: {} iterations'.format(tei.iterations))
        
        if not use_torch: 
            t1 = np.allclose(new_r, rr)
            t2 = np.allclose(new_s, ss)
            t3 = np.allclose(new_t, tt)
        else:
            t1 = np.allclose(new_r, rr.cpu().numpy())
            t2 = np.allclose(new_s, ss.cpu().numpy())
            t3 = np.allclose(new_t, tt.cpu().numpy())

        passed = np.all([t1, t2, t3])
        
        if verbose:        
            print('tensor_sem.py: r found correctly from xyz: {}'.format(t1))
            print('tensor_sem.py: s found correctly from xyz: {}'.format(t2))
            print('tensor_sem.py: t found correctly from xyz: {}'.format(t3))

        if not passed:
            sys.exit('tensor_sem.py: find_rst_from_xyz: failed')
        else:
            print('tensor_sem.py: find_rst_from_xyz: passed')

        # Test interpolation 
        if not use_torch:
            xx_int = np.zeros_like(new_x)
            yy_int = np.zeros_like(new_y)
            zz_int = np.zeros_like(new_z)
        else:
            xx_int = torch.zeros(new_x.shape, dtype=torch.float64, device=device)
            yy_int = torch.zeros(new_y.shape, dtype=torch.float64, device=device)
            zz_int = torch.zeros(new_z.shape, dtype=torch.float64, device=device)

        start_time = MPI.Wtime()
        for i in range(0, int(total_iterations)):
            start = i * max_pts
            end   = (i+1) * max_pts
            if end > total_pts:
                end = total_pts
            npoints = end - start
            xx_int[start:end] = tei.interpolate_field_at_rst(rr[start:end], ss[start:end], tt[start:end], rshp_x[:npoints,:,:,:,:])
            yy_int[start:end] = tei.interpolate_field_at_rst(rr[start:end], ss[start:end], tt[start:end], rshp_y[:npoints,:,:,:,:])
            zz_int[start:end] = tei.interpolate_field_at_rst(rr[start:end], ss[start:end], tt[start:end], rshp_z[:npoints,:,:,:,:])
        print('tensor_sem.py: Time to interpolate: {}'.format(MPI.Wtime() - start_time))
 
        if not use_torch: 
            t1 = np.allclose(new_x, xx_int)
            t2 = np.allclose(new_y, yy_int)
            t3 = np.allclose(new_z, zz_int)
        else:
            t1 = np.allclose(new_x, xx_int.cpu().numpy())
            t2 = np.allclose(new_y, yy_int.cpu().numpy())
            t3 = np.allclose(new_z, zz_int.cpu().numpy())

        passed = np.all([t1, t2, t3])
        
        if verbose:        
            print('tensor_sem.py: x interpolated correctly: {}'.format(t1))
            print('tensor_sem.py: y interpolated correctly: {}'.format(t2))
            print('tensor_sem.py: z interpolated correctly: {}'.format(t3))

        if not passed:
            sys.exit('tensor_sem.py: interpolate_field_at_rst: failed')
        else:
            print('tensor_sem.py: interpolate_field_at_rst: passed')


if comm.Get_rank() == 0: print('================')
if comm.Get_rank() == 0: print('================')

test_sem_and_tensor_sem(n_new = 8, elem = 1, max_pts = 128, use_torch = False, verbose = False)

if comm.Get_rank() == 0: print('================')
if comm.Get_rank() == 0: print('================')

test_sem_and_tensor_sem(n_new = 8, elem = 1, max_pts = 1000, use_torch = True, verbose = False)
