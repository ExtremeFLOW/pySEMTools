from mpi4py import MPI
import torch
import math
import sys

from pysemtools.io.ppymech.neksuite import preadnek
from pysemtools.datatypes.msh import Mesh
from pysemtools.interpolation.point_interpolator.multiple_point_interpolator_legendre_torch import LegendreInterpolator
from pysemtools.interpolation.point_interpolator.multiple_point_helper_functions_torch import legendre_basis_at_xtest, legendre_basis_derivative_at_xtest
from pysemtools.interpolation.point_interpolator.multiple_point_helper_functions_torch import apply_operators_3d
import numpy as np

comm = MPI.COMM_WORLD
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == 'cuda:0': 
    torch.cuda.set_device(device)
def test_autograd():

    fname = "examples/data/rbc0.f00001"
    data = preadnek(fname, comm)
    msh = Mesh(comm, data = data, create_connectivity=False)

    # Define the initialization parameters
    npoints = msh.lx
    nelems = 1
    ei = LegendreInterpolator(msh.lx, max_pts=npoints, max_elems=nelems)

    # Allocate buffers
    r = torch.zeros((npoints, 1, 1, 1), dtype=dtype, device=device)
    s = torch.zeros((npoints, 1, 1, 1), dtype=dtype, device=device)
    t = torch.zeros((npoints, 1, 1, 1), dtype=dtype, device=device)

    # Update the points to test scenario where points are less that those in buffer
    npoints = msh.lx - 2
    elem = [100 for i in range(npoints)]

    # Assign the r,s,t coordinates.
    for i in range(0, npoints):
        r[i, 0, 0, 0] = ei.x_gll[-1 -i, 0, 0, 0]
        s[i, 0, 0, 0] = ei.x_gll[-1 -i, 0, 0, 0]
        t[i, 0, 0, 0] = ei.x_gll[-1 -i, 0, 0, 0]

    # Specify that gradients are required
    ## After computations are done, we will be able to see the gradient of a function
    ## wrt these parameters by checking the .grad attribute of the tensor
    r.requires_grad_(True)
    s.requires_grad_(True)
    t.requires_grad_(True)

    # Project the needed elements into the basis of choice
    elem_new_shape = (npoints, nelems, msh.x.shape[1], msh.x.shape[2], msh.x.shape[3])

    ei.project_element_into_basis(msh.x[elem].reshape(elem_new_shape),
                        msh.y[elem].reshape(elem_new_shape),
                        msh.z[elem].reshape(elem_new_shape),
                    )
    n = msh.lx
    
    # ====================
    # Autograd calculation
    # ====================

    # Find the basis and do the fordward pass
    ortho_basis_rj = legendre_basis_at_xtest(n, r[:npoints, : ,: ,:])
    ortho_basis_sj = legendre_basis_at_xtest(n, s[:npoints, : ,: ,:])
    ortho_basis_tj = legendre_basis_at_xtest(n, t[:npoints, : ,: ,:])

    # Apply the operators
    x = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.x_e_hat)
    y = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.y_e_hat)
    z = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.z_e_hat)

    # Calculate the jacobian
    j_autograd = torch.zeros((npoints, nelems, 3, 3), dtype=dtype, device=device)
    if r.grad is not None:
        r.grad.zero_()
    if s.grad is not None:
        s.grad.zero_()
    if t.grad is not None:
        t.grad.zero_()

    x.backward(torch.ones_like(x), retain_graph=True)
    j_autograd[:, :, 0, 0] = r.grad[:npoints, :, 0, 0]
    j_autograd[:, :, 0, 1] = s.grad[:npoints, :, 0, 0]
    j_autograd[:, :, 0, 2] = t.grad[:npoints, :, 0, 0]

    r.grad.zero_(), s.grad.zero_(), t.grad.zero_()
    y.backward(torch.ones_like(y), retain_graph=True)
    j_autograd[:, :, 1, 0] = r.grad[:npoints, :, 0, 0]
    j_autograd[:, :, 1, 1] = s.grad[:npoints, :, 0, 0]
    j_autograd[:, :, 1, 2] = t.grad[:npoints, :, 0, 0]

    r.grad.zero_(), s.grad.zero_(), t.grad.zero_()
    z.backward(torch.ones_like(z), retain_graph=True)
    j_autograd[:, :, 2, 0] = r.grad[:npoints, :, 0, 0]
    j_autograd[:, :, 2, 1] = s.grad[:npoints, :, 0, 0]
    j_autograd[:, :, 2, 2] = t.grad[:npoints, :, 0, 0]

    # ======================
    # Polynomial calculation
    # ======================

    # Stop gradient requirements 
    r.requires_grad_(False)
    s.requires_grad_(False)
    t.requires_grad_(False)
    
    # Find the basis and do the fordward pass
    ortho_basis_rj = legendre_basis_at_xtest(n, r[:npoints, : ,: ,:])
    ortho_basis_sj = legendre_basis_at_xtest(n, s[:npoints, : ,: ,:])
    ortho_basis_tj = legendre_basis_at_xtest(n, t[:npoints, : ,: ,:])

    ortho_basis_prm_rj = legendre_basis_derivative_at_xtest(ortho_basis_rj, r)
    ortho_basis_prm_sj = legendre_basis_derivative_at_xtest(ortho_basis_sj, s)
    ortho_basis_prm_tj = legendre_basis_derivative_at_xtest(ortho_basis_tj, t)

    j_polynomial = torch.zeros((npoints, nelems, 3, 3), dtype=dtype, device=device, requires_grad=False)
    j_polynomial[:, :, 0, 0] = apply_operators_3d(ortho_basis_prm_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.x_e_hat)[:, :, 0, 0]
    j_polynomial[:, :, 0, 1] = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_prm_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.x_e_hat)[:, :, 0, 0]
    j_polynomial[:, :, 0, 2] = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_prm_tj.permute(0, 1, 3, 2), ei.x_e_hat)[:, :, 0, 0]

    j_polynomial[:, :, 1, 0] = apply_operators_3d(ortho_basis_prm_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.y_e_hat)[:, :, 0, 0]
    j_polynomial[:, :, 1, 1] = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_prm_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.y_e_hat)[:, :, 0, 0]
    j_polynomial[:, :, 1, 2] = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_prm_tj.permute(0, 1, 3, 2), ei.y_e_hat)[:, :, 0, 0]

    j_polynomial[:, :, 2, 0] = apply_operators_3d(ortho_basis_prm_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.z_e_hat)[:, :, 0, 0]
    j_polynomial[:, :, 2, 1] = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_prm_sj.permute(0, 1, 3, 2), ortho_basis_tj.permute(0, 1, 3, 2), ei.z_e_hat)[:, :, 0, 0]
    j_polynomial[:, :, 2, 2] = apply_operators_3d(ortho_basis_rj.permute(0, 1, 3, 2), ortho_basis_sj.permute(0, 1, 3, 2), ortho_basis_prm_tj.permute(0, 1, 3, 2), ei.z_e_hat)[:, :, 0, 0]
    print(j_autograd)
    print(j_polynomial)
    passed = torch.allclose(j_autograd, j_polynomial)

    assert passed