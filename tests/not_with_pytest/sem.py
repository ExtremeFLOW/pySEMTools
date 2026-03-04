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
from pysemtools.datatypes.coef import Coef as coef_c
from pysemtools.datatypes.field import Field as field_c

# Read the original mesh data
fname = '../examples/data/rbc0.f00001'
data     = preadnek(fname, comm)
msh      = msh_c(comm, data = data)
del data

# Create a refined mesh
pref = PRefiner(n_old = msh.lx, n_new = 10)
msh_ref = pref.create_refined_mesh(comm, msh = msh)

# Create the element interpolator for both meshes
ei = element_interpolator_c(msh.lx)
ei_ref = element_interpolator_c(msh_ref.lx)

if comm.Get_rank() == 0:

    # ======================================================#
    # Test interpolating all points within the same element #
    # ======================================================#

    # Now choose 1 element to interpolate
    elem = 1

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
    ei.project_element_into_basis(msh.x[elem,:,:,:], msh.y[elem,:,:,:], msh.z[elem,:,:,:])    
    for k in range(0, msh_ref.lz):
        for j in range(0, msh_ref.ly):
            for i in range(0, msh_ref.lx):
                found_r[k,j,i], found_s[k,j,i], found_t[k,j,i] = ei.find_rst_from_xyz(msh_ref.x[elem,k,j,i], msh_ref.y[elem,k,j,i], msh_ref.z[elem,k,j,i])

    print('The r coordinates were found correctly: {}'.format(np.allclose(exact_r, found_r)))
    print('The s coordinates were found correctly: {}'.format(np.allclose(exact_s, found_s)))
    print('The t coordinates were found correctly: {}'.format(np.allclose(exact_t, found_t)))
    print('sem: The last point took: {} iterations'.format(ei.iterations))
    
    # Interpolate back to the xyz coordinates
    for k in range(0, msh_ref.lz):
        for j in range(0, msh_ref.ly):
            for i in range(0, msh_ref.lx):
                found_x[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.x[elem,:,:,:])
                found_y[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.y[elem,:,:,:])
                found_z[k,j,i] = ei.interpolate_field_at_rst(found_r[k,j,i], found_s[k,j,i], found_t[k,j,i], msh.z[elem,:,:,:])

    print('The x coordinates were found correctly: {}'.format(np.allclose(msh_ref.x[elem,:,:,:], found_x)))
    print('The y coordinates were found correctly: {}'.format(np.allclose(msh_ref.y[elem,:,:,:], found_y)))
    print('The z coordinates were found correctly: {}'.format(np.allclose(msh_ref.z[elem,:,:,:], found_z)))

