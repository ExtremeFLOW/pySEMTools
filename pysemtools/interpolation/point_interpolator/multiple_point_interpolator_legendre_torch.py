""" Contains class to interpolate multiple points using numpy"""

import numpy as np
import torch

from tqdm import tqdm
from .multiple_point_interpolator import MultiplePointInterpolator
from .multiple_point_helper_functions_numpy import get_basis_transformation_matrices
from .multiple_point_helper_functions_torch import (
    apply_operators_3d,
    legendre_basis_at_xtest,
    legendre_basis_derivative_at_xtest,
    lag_interp_matrix_at_xtest,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NoneType = type(None)


class LegendreInterpolator(MultiplePointInterpolator):
    """Class to interpolate multiple points using Legendre polynomials in 3D."""

    def __init__(self, n, max_pts=1, max_elems=1, use_autograd=False):

        # Initialize parent class
        super().__init__(n, max_pts=max_pts, max_elems=max_elems)

        # Initialize the class attributes
        self.use_autograd = use_autograd
        self.optimizer = "newton"  # "gd" or "newton"

        # Get reference element
        self.v1d = None
        self.v1d_inv = None
        self.v = None
        self.v_inv = None
        get_basis_transformation_matrices(self)
        self.v1d = torch.as_tensor(self.v1d, dtype=torch.float64, device=device)
        self.v1d_inv = torch.as_tensor(self.v1d_inv, dtype=torch.float64, device=device)
        self.v = torch.as_tensor(self.v, dtype=torch.float64, device=device)
        self.v_inv = torch.as_tensor(self.v_inv, dtype=torch.float64, device=device)

        # Convert all attributes to torch tensors
        self.x_gll = torch.as_tensor(self.x_gll, dtype=torch.float64, device=device)
        self.w_gll = torch.as_tensor(self.w_gll, dtype=torch.float64, device=device)
        self.x_e = torch.as_tensor(self.x_e, dtype=torch.float64, device=device)
        self.y_e = torch.as_tensor(self.y_e, dtype=torch.float64, device=device)
        self.z_e = torch.as_tensor(self.z_e, dtype=torch.float64, device=device)

        self.xj = torch.as_tensor(self.xj, dtype=torch.float64, device=device)
        self.yj = torch.as_tensor(self.yj, dtype=torch.float64, device=device)
        self.zj = torch.as_tensor(self.zj, dtype=torch.float64, device=device)
        self.rj = torch.as_tensor(
            self.rj, dtype=torch.float64, device=device
        ).requires_grad_(use_autograd)
        self.sj = torch.as_tensor(
            self.sj, dtype=torch.float64, device=device
        ).requires_grad_(use_autograd)
        self.tj = torch.as_tensor(
            self.tj, dtype=torch.float64, device=device
        ).requires_grad_(use_autograd)
        self.rstj = torch.as_tensor(self.rstj, dtype=torch.float64, device=device)
        self.eps_rst = torch.as_tensor(self.eps_rst, dtype=torch.float64, device=device)
        self.jac = torch.as_tensor(self.jac, dtype=torch.float64, device=device)
        self.field_e = torch.as_tensor(self.field_e, dtype=torch.float64, device=device)
        self.point_inside_element = torch.as_tensor(
            self.point_inside_element, dtype=torch.bool, device=device
        )

        # dummy variables
        self.x_e_hat = None
        self.y_e_hat = None
        self.z_e_hat = None

        return

    def project_element_into_basis(self, x_e, y_e, z_e, apply_1d_ops=True):

        npoints = x_e.shape[0]
        nelems = x_e.shape[1]
        n = x_e.shape[2] * x_e.shape[3] * x_e.shape[4]

        # Assing the inputs to proper formats
        self.x_e[:npoints, :nelems, :, :] = torch.as_tensor(
            x_e.reshape(npoints, nelems, n, 1)[:, :, :, :],
            dtype=torch.float64,
            device=device,
        )
        self.y_e[:npoints, :nelems, :, :] = torch.as_tensor(
            y_e.reshape(npoints, nelems, n, 1)[:, :, :, :],
            dtype=torch.float64,
            device=device,
        )
        self.z_e[:npoints, :nelems, :, :] = torch.as_tensor(
            z_e.reshape(npoints, nelems, n, 1)[:, :, :, :],
            dtype=torch.float64,
            device=device,
        )

        # Get the modal representation
        if not apply_1d_ops:

            self.x_e_hat = torch.einsum(
                "ijkl,ijlm->ijkm", self.v_inv, self.x_e[:npoints, :nelems, :, :]
            )
            self.y_e_hat = torch.einsum(
                "ijkl,ijlm->ijkm", self.v_inv, self.y_e[:npoints, :nelems, :, :]
            )
            self.z_e_hat = torch.einsum(
                "ijkl,ijlm->ijkm", self.v_inv, self.z_e[:npoints, :nelems, :, :]
            )

        else:

            # Keep in mind, the operators are already transposed here.
            self.x_e_hat = apply_operators_3d(
                self.v1d_inv,
                self.v1d_inv,
                self.v1d_inv,
                self.x_e[:npoints, :nelems, :, :],
            )
            self.y_e_hat = apply_operators_3d(
                self.v1d_inv,
                self.v1d_inv,
                self.v1d_inv,
                self.y_e[:npoints, :nelems, :, :],
            )
            self.z_e_hat = apply_operators_3d(
                self.v1d_inv,
                self.v1d_inv,
                self.v1d_inv,
                self.z_e[:npoints, :nelems, :, :],
            )

        return

    def get_xyz_from_rst(self, rj, sj, tj, apply_1d_ops=True):

        if not self.use_autograd:
            return self.get_xyz_from_rst_polynomial_(rj, sj, tj, apply_1d_ops)
        else:
            return self.get_xyz_from_rst_autograd_(rj, sj, tj, apply_1d_ops)

    def get_xyz_from_rst_polynomial_(self, rj, sj, tj, apply_1d_ops=True):
        """
        This function calculates the xyz coordinates from the given rst coordinates for points
        in the elements that were projected into xhat, yhat, zhat.
        """

        npoints = rj.shape[0]
        nelems = self.x_e_hat.shape[1]
        n = self.n

        self.rj[:npoints, :nelems, :, :] = torch.as_tensor(
            rj[:, :, :, :], dtype=torch.float64, device=device
        )
        self.sj[:npoints, :nelems, :, :] = torch.as_tensor(
            sj[:, :, :, :], dtype=torch.float64, device=device
        )
        self.tj[:npoints, :nelems, :, :] = torch.as_tensor(
            tj[:, :, :, :], dtype=torch.float64, device=device
        )

        # If modal search, the basis is legendre
        ortho_basis_rj = legendre_basis_at_xtest(n, self.rj[:npoints, :nelems, :, :])
        ortho_basis_sj = legendre_basis_at_xtest(n, self.sj[:npoints, :nelems, :, :])
        ortho_basis_tj = legendre_basis_at_xtest(n, self.tj[:npoints, :nelems, :, :])

        ortho_basis_prm_rj = legendre_basis_derivative_at_xtest(
            ortho_basis_rj, self.rj[:npoints, :nelems, :, :]
        )
        ortho_basis_prm_sj = legendre_basis_derivative_at_xtest(
            ortho_basis_sj, self.sj[:npoints, :nelems, :, :]
        )
        ortho_basis_prm_tj = legendre_basis_derivative_at_xtest(
            ortho_basis_tj, self.tj[:npoints, :nelems, :, :]
        )

        if not apply_1d_ops:
            raise RuntimeError("Only worrking by applying 1d operators")

        elif apply_1d_ops:

            x = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.x_e_hat,
            )
            y = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.y_e_hat,
            )
            z = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.z_e_hat,
            )

            self.jac[:npoints, :nelems, 0, 0] = apply_operators_3d(
                ortho_basis_prm_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.x_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 0, 1] = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_prm_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.x_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 0, 2] = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_prm_tj.permute(0, 1, 3, 2),
                self.x_e_hat,
            )[:, :, 0, 0]

            self.jac[:npoints, :nelems, 1, 0] = apply_operators_3d(
                ortho_basis_prm_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.y_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 1, 1] = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_prm_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.y_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 1, 2] = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_prm_tj.permute(0, 1, 3, 2),
                self.y_e_hat,
            )[:, :, 0, 0]

            self.jac[:npoints, :nelems, 2, 0] = apply_operators_3d(
                ortho_basis_prm_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.z_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 2, 1] = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_prm_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.z_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 2, 2] = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_prm_tj.permute(0, 1, 3, 2),
                self.z_e_hat,
            )[:, :, 0, 0]

        return x, y, z

    def get_xyz_from_rst_autograd_(self, rj, sj, tj, apply_1d_ops=True):
        """
        This function calculates the xyz coordinates from the given rst coordinates for points
        in the elements that were projected into xhat, yhat, zhat.
        """

        npoints = rj.shape[0]
        nelems = self.x_e_hat.shape[1]
        n = self.n

        with torch.no_grad():
            self.rj[:npoints, :nelems, :, :] = torch.as_tensor(
                rj[:, :, :, :], dtype=torch.float64, device=device
            )
            self.sj[:npoints, :nelems, :, :] = torch.as_tensor(
                sj[:, :, :, :], dtype=torch.float64, device=device
            )
            self.tj[:npoints, :nelems, :, :] = torch.as_tensor(
                tj[:, :, :, :], dtype=torch.float64, device=device
            )

        # Evaluate the basis functions
        ortho_basis_rj = legendre_basis_at_xtest(n, self.rj[:npoints, :nelems, :, :])
        ortho_basis_sj = legendre_basis_at_xtest(n, self.sj[:npoints, :nelems, :, :])
        ortho_basis_tj = legendre_basis_at_xtest(n, self.tj[:npoints, :nelems, :, :])

        if not apply_1d_ops:
            raise RuntimeError("Only worrking by applying 1d operators")

        elif apply_1d_ops:

            # Evaluate the forward pass
            x = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.x_e_hat,
            )
            y = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.y_e_hat,
            )
            z = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.z_e_hat,
            )

            # Make sure the gradients are zero
            if self.rj.grad is not None:
                self.rj.grad.zero_()
            if self.sj.grad is not None:
                self.sj.grad.zero_()
            if self.tj.grad is not None:
                self.tj.grad.zero_()

            # Do a backward pass on x to calculate dx/drdsdt
            x.backward(torch.ones_like(x), retain_graph=True)
            self.jac[:npoints, :nelems, 0, 0] = self.rj.grad[:npoints, :nelems, 0, 0]
            self.jac[:npoints, :nelems, 0, 1] = self.sj.grad[:npoints, :nelems, 0, 0]
            self.jac[:npoints, :nelems, 0, 2] = self.tj.grad[:npoints, :nelems, 0, 0]

            # Zero out the gradients and repeat with y
            self.rj.grad.zero_(), self.sj.grad.zero_(), self.tj.grad.zero_()
            y.backward(torch.ones_like(y), retain_graph=True)
            self.jac[:npoints, :nelems, 1, 0] = self.rj.grad[:npoints, :nelems, 0, 0]
            self.jac[:npoints, :nelems, 1, 1] = self.sj.grad[:npoints, :nelems, 0, 0]
            self.jac[:npoints, :nelems, 1, 2] = self.tj.grad[:npoints, :nelems, 0, 0]

            # Zero out the gradients and repeat with z
            self.rj.grad.zero_(), self.sj.grad.zero_(), self.tj.grad.zero_()
            z.backward(torch.ones_like(z), retain_graph=True)
            self.jac[:npoints, :nelems, 2, 0] = self.rj.grad[:npoints, :nelems, 0, 0]
            self.jac[:npoints, :nelems, 2, 1] = self.sj.grad[:npoints, :nelems, 0, 0]
            self.jac[:npoints, :nelems, 2, 2] = self.tj.grad[:npoints, :nelems, 0, 0]

        return x, y, z

    def get_xyz_from_rst_no_jac_(self, rj, sj, tj, apply_1d_ops=True):
        """
        This function calculates the xyz coordinates from the given rst coordinates for points
        in the elements that were projected into xhat, yhat, zhat.
        """

        npoints = rj.shape[0]
        nelems = self.x_e_hat.shape[1]
        n = self.n

        with torch.no_grad():
            self.rj[:npoints, :nelems, :, :] = torch.as_tensor(
                rj[:, :, :, :], dtype=torch.float64, device=device
            )
            self.sj[:npoints, :nelems, :, :] = torch.as_tensor(
                sj[:, :, :, :], dtype=torch.float64, device=device
            )
            self.tj[:npoints, :nelems, :, :] = torch.as_tensor(
                tj[:, :, :, :], dtype=torch.float64, device=device
            )

        # Evaluate the basis functions
        ortho_basis_rj = legendre_basis_at_xtest(n, self.rj[:npoints, :nelems, :, :])
        ortho_basis_sj = legendre_basis_at_xtest(n, self.sj[:npoints, :nelems, :, :])
        ortho_basis_tj = legendre_basis_at_xtest(n, self.tj[:npoints, :nelems, :, :])

        if not apply_1d_ops:
            raise RuntimeError("Only worrking by applying 1d operators")

        elif apply_1d_ops:

            # Evaluate the forward pass
            x = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.x_e_hat,
            )
            y = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.y_e_hat,
            )
            z = apply_operators_3d(
                ortho_basis_rj.permute(0, 1, 3, 2),
                ortho_basis_sj.permute(0, 1, 3, 2),
                ortho_basis_tj.permute(0, 1, 3, 2),
                self.z_e_hat,
            )

        return (x, y, z, [ortho_basis_rj, ortho_basis_sj, ortho_basis_tj])

    def find_rst_from_xyz(
        self, rj, sj, tj, tol=np.finfo(np.double).eps * 10, max_iterations=50
    ):

        if self.optimizer == "newton":
            return self.find_rst_from_xyz_newton(
                rj, sj, tj, tol=tol, max_iterations=max_iterations
            )
        if self.optimizer == "gd":
            return self.find_rst_from_xyz_gd(
                rj, sj, tj, tol=tol, max_iterations=max_iterations
            )

    def find_rst_from_xyz_newton(
        self, xj, yj, zj, tol=np.finfo(np.double).eps * 10, max_iterations=50
    ):
        """

        Find rst coordinates from a given xyz group of points.
        Note that this function needs to be called after the
        element has been projected into the basis.

        """

        self.point_inside_element[:, :, :, :] = False

        npoints = xj.shape[0]
        nelems = self.x_e_hat.shape[1]
        # n = self.n

        self.xj[:npoints, :, :, :] = torch.as_tensor(
            xj[:, :, :, :], dtype=torch.float64, device=device
        )
        self.yj[:npoints, :, :, :] = torch.as_tensor(
            yj[:, :, :, :], dtype=torch.float64, device=device
        )
        self.zj[:npoints, :, :, :] = torch.as_tensor(
            zj[:, :, :, :], dtype=torch.float64, device=device
        )

        # Determine the initial conditions
        determine_initial_guess(
            self, npoints=npoints, nelems=nelems
        )  # This populates self.rj, self.sj, self.tj for 1st iteration

        # Use the newton method to identify the coordinates
        self.iterations = 0
        self.eps_rst[:npoints, :nelems, :, :] = 1

        # create an integer array to store the number of iterations that it took for each point
        iterations_per_point = torch.zeros(
            npoints, nelems, 1, 1, dtype=torch.int32, device=device
        )
        iterations_per_point[:, :, :, :] = max_iterations
        points_already_found = torch.any(iterations_per_point[:npoints, :nelems] < max_iterations, dim=(2, 3))

        while (
            torch.any(torch.norm(self.eps_rst[:npoints, :nelems], dim=(2, 3)) > tol)
            and self.iterations < max_iterations
        ):

            with torch.no_grad():

                # Update the guess
                self.rstj[:npoints, :nelems, 0, 0] = self.rj[:npoints, :nelems, 0, 0]
                self.rstj[:npoints, :nelems, 1, 0] = self.sj[:npoints, :nelems, 0, 0]
                self.rstj[:npoints, :nelems, 2, 0] = self.tj[:npoints, :nelems, 0, 0]

            # Estimate the xyz values from rst and also the jacobian
            # (it is updated inside self.jac)
            # The elements are determined by the number of x_hats,
            # this is given in the projection function
            # Check that one out if you forget.
            xj_found, yj_found, zj_found = self.get_xyz_from_rst(
                self.rj[:npoints, :nelems, :, :],
                self.sj[:npoints, :nelems, :, :],
                self.tj[:npoints, :nelems, :, :],
            )

            with torch.no_grad():
                # Find the residuals and the jacobian inverse.
                self.eps_rst[:npoints, :nelems, 0, 0] = (
                    self.xj[:npoints, :nelems, :, :] - xj_found
                )[:, :, 0, 0]
                self.eps_rst[:npoints, :nelems, 1, 0] = (
                    self.yj[:npoints, :nelems, :, :] - yj_found
                )[:, :, 0, 0]
                self.eps_rst[:npoints, :nelems, 2, 0] = (
                    self.zj[:npoints, :nelems, :, :] - zj_found
                )[:, :, 0, 0]
                # zero out differences of points that have already been found, so they do not keep being updated
                self.eps_rst[torch.where(points_already_found)] = 0

                jac_inv = invert_jac(self.jac[:npoints, :nelems])

                # Find the new guess
                self.rstj[:npoints, :nelems] = self.rstj[:npoints, :nelems] - (
                    0
                    - torch.einsum(
                        "ijkl,ijlm->ijkm", jac_inv, self.eps_rst[:npoints, :nelems]
                    )
                )

                # Update the guess
                self.rj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 0, 0]
                self.sj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 1, 0]
                self.tj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 2, 0]
                self.iterations += 1
                
                # Determine which points have already been found so they are not updated anymore
                points_found_this_it = (
                    torch.norm(self.eps_rst[:npoints, :nelems], dim=(2, 3)) <= tol
                )
                points_already_found = torch.any(iterations_per_point[:npoints, :nelems] < max_iterations, dim=(2, 3))
                # Update the number of iterations only if the point has newly been found
                iterations_per_point[(points_found_this_it & ~points_already_found)] = self.iterations

        with torch.no_grad():
            # Check if points are inside the element
            limit = 1 + np.finfo(np.single).eps
            t1 = (abs(self.rj[:npoints, :nelems, 0, 0]) <= limit).reshape(
                npoints, nelems, 1, 1
            )
            t2 = (abs(self.sj[:npoints, :nelems, 0, 0]) <= limit).reshape(
                npoints, nelems, 1, 1
            )
            t3 = (abs(self.tj[:npoints, :nelems, 0, 0]) <= limit).reshape(
                npoints, nelems, 1, 1
            )

            t4 = iterations_per_point < max_iterations

            # Pointwise comparison
            self.point_inside_element[:npoints, :nelems, :, :] = t1 & t2 & t3 & t4

        return (
            self.rj[:npoints, :nelems].detach(),
            self.sj[:npoints, :nelems].detach(),
            self.tj[:npoints, :nelems].detach(),
        )

    def find_rst_from_xyz_gd(
        self, xj, yj, zj, tol=np.finfo(np.double).eps * 10, max_iterations=50
    ):
        """

        Find rst coordinates from a given xyz group of points.
        Note that this function needs to be called after the
        element has been projected into the basis.

        """

        self.point_inside_element[:, :, :, :] = False

        npoints = xj.shape[0]
        nelems = self.x_e_hat.shape[1]
        # n = self.n

        self.xj[:npoints, :, :, :] = torch.as_tensor(
            xj[:, :, :, :], dtype=torch.float64, device=device
        )
        self.yj[:npoints, :, :, :] = torch.as_tensor(
            yj[:, :, :, :], dtype=torch.float64, device=device
        )
        self.zj[:npoints, :, :, :] = torch.as_tensor(
            zj[:, :, :, :], dtype=torch.float64, device=device
        )

        # Determine the initial conditions
        determine_initial_guess(
            self, npoints=npoints, nelems=nelems
        )  # This populates self.rj, self.sj, self.tj for 1st iteration

        # Use the newton method to identify the coordinates
        self.iterations = 0
        self.eps_rst[:npoints, :nelems, :, :] = 1

        loss = torch.norm(self.eps_rst[:npoints, :nelems], dim=(2, 3))

        # Coordinates that will be found
        xj_tofind = self.xj[:npoints, :nelems, :, :]
        yj_tofind = self.yj[:npoints, :nelems, :, :]
        zj_tofind = self.zj[:npoints, :nelems, :, :]

        while torch.any(loss > tol) and self.iterations < max_iterations:

            # Find current xyz
            xj_found, yj_found, zj_found, _ = self.get_xyz_from_rst_no_jac_(
                self.rj[:npoints, :nelems, :, :],
                self.sj[:npoints, :nelems, :, :],
                self.tj[:npoints, :nelems, :, :],
            )

            # Calculate the differences
            x_diff = xj_tofind - xj_found
            y_diff = yj_tofind - yj_found
            z_diff = zj_tofind - zj_found

            # Stack the differences into one tensor
            diff_tensor = torch.cat((x_diff, y_diff, z_diff), dim=2)

            # Mean squared error per point
            loss = (
                torch.norm(diff_tensor, dim=(2, 3)).reshape(npoints, nelems, 1, 1) ** 2
                / 3
            )

            # Get the gradients
            loss.backward(torch.ones_like(loss))

            # Choose the learning rate (step size)
            if self.iterations == 0:
                # First iteration is the same for every point
                lr = 1 / torch.mean(
                    torch.mean(torch.abs(self.rj.grad[:npoints, :nelems, 0, 0]))
                    + torch.mean(torch.abs(self.sj.grad[:npoints, :nelems, 0, 0]))
                    + torch.mean(torch.abs(self.tj.grad[:npoints, :nelems, 0, 0]))
                )
            else:
                # Second iteration forward is adapted per point
                with torch.no_grad():
                    rn = self.rj[:npoints, :nelems, :, :].clone()
                    sn = self.sj[:npoints, :nelems, :, :].clone()
                    tn = self.tj[:npoints, :nelems, :, :].clone()
                    dldrn = self.rj.grad[:npoints, :nelems, :, :].clone()
                    dldsn = self.sj.grad[:npoints, :nelems, :, :].clone()
                    dldtn = self.tj.grad[:npoints, :nelems, :, :].clone()

                    # Substract current and previous values
                    rst = torch.cat((rn - rn_1, sn - sn_1, tn - tn_1), dim=2)
                    dfdrst = torch.cat(
                        (dldrn - dldrn_1, dldsn - dldsn_1, dldtn - dldtn_1), dim=2
                    )

                    # Perform inner products and set the step size
                    num = torch.abs(
                        torch.einsum(
                            "ijkl, ijlm->ijkm", rst.permute(0, 1, 3, 2), dfdrst
                        )
                    )
                    den = torch.einsum(
                        "ijkl, ijlm->ijkm", dfdrst.permute(0, 1, 3, 2), dfdrst
                    )
                    lr = num / den

                    # Do some checks
                    lr[torch.isnan(lr)] = 1000
                    lr[torch.where(loss < tol)] = 0

            # Store the values for next iteration
            with torch.no_grad():
                rn_1 = self.rj[:npoints, :nelems, :, :].clone()
                sn_1 = self.sj[:npoints, :nelems, :, :].clone()
                tn_1 = self.tj[:npoints, :nelems, :, :].clone()
                dldrn_1 = self.rj.grad[:npoints, :nelems, :, :].clone()
                dldsn_1 = self.sj.grad[:npoints, :nelems, :, :].clone()
                dldtn_1 = self.tj.grad[:npoints, :nelems, :, :].clone()

            # Update the guess
            with torch.no_grad():
                self.rj[:npoints, :nelems, :, :] = (
                    self.rj[:npoints, :nelems, :, :]
                    - lr * self.rj.grad[:npoints, :nelems, :, :]
                )
                self.sj[:npoints, :nelems, :, :] = (
                    self.sj[:npoints, :nelems, :, :]
                    - lr * self.sj.grad[:npoints, :nelems, :, :]
                )
                self.tj[:npoints, :nelems, :, :] = (
                    self.tj[:npoints, :nelems, :, :]
                    - lr * self.tj.grad[:npoints, :nelems, :, :]
                )
            # Zero out the gradients for next it
            self.rj.grad.zero_(), self.sj.grad.zero_(), self.tj.grad.zero_()
            self.iterations += 1

        # Check if points are inside the element
        with torch.no_grad():
            limit = 1 + np.finfo(np.single).eps
            t1 = (abs(self.rj[:npoints, :nelems, 0, 0]) <= limit).reshape(
                npoints, nelems, 1, 1
            )
            t2 = (abs(self.sj[:npoints, :nelems, 0, 0]) <= limit).reshape(
                npoints, nelems, 1, 1
            )
            t3 = (abs(self.tj[:npoints, :nelems, 0, 0]) <= limit).reshape(
                npoints, nelems, 1, 1
            )

            # Pointwise comparison
            self.point_inside_element[:npoints, :nelems, :, :] = t1 & t2 & t3

        return (
            self.rj[:npoints, :nelems].detach(),
            self.sj[:npoints, :nelems].detach(),
            self.tj[:npoints, :nelems].detach(),
        )

    def interpolate_field_at_rst(self, rj, sj, tj, field_e, apply_1d_ops=True):
        """
        Interpolate each point in a given field.
        EACH POINT RECIEVES ONE FIELD! SO FIELDS MIGHT BE DUPLICATED

        """

        with torch.no_grad():
            npoints = rj.shape[0]
            nelems = rj.shape[1]
            n = field_e.shape[2] * field_e.shape[3] * field_e.shape[4]

            self.rj[:npoints, :nelems, :, :] = torch.as_tensor(
                rj[:, :, :, :], dtype=torch.float64, device=device
            )
            self.sj[:npoints, :nelems, :, :] = torch.as_tensor(
                sj[:, :, :, :], dtype=torch.float64, device=device
            )
            self.tj[:npoints, :nelems, :, :] = torch.as_tensor(
                tj[:, :, :, :], dtype=torch.float64, device=device
            )

            # Assing the inputs to proper formats
            self.field_e[:npoints, :nelems, :, :] = torch.as_tensor(
                field_e.reshape(npoints, nelems, n, 1)[:, :, :, :],
                dtype=torch.float64,
                device=device,
            )

            lk_r = lag_interp_matrix_at_xtest(
                self.x_gll, self.rj[:npoints, :nelems, :, :]
            )
            lk_s = lag_interp_matrix_at_xtest(
                self.x_gll, self.sj[:npoints, :nelems, :, :]
            )
            lk_t = lag_interp_matrix_at_xtest(
                self.x_gll, self.tj[:npoints, :nelems, :, :]
            )

            if not apply_1d_ops:
                raise RuntimeError("Only worrking by applying 1d operators")
            elif apply_1d_ops:
                field_at_rst = apply_operators_3d(
                    lk_r.permute(0, 1, 3, 2),
                    lk_s.permute(0, 1, 3, 2),
                    lk_t.permute(0, 1, 3, 2),
                    self.field_e[:npoints, :nelems, :, :],
                )

        return field_at_rst

    def alloc_result_buffer(self, **kwargs):
        dtype = kwargs.get("dtype", "double")

        if dtype == "double":
            return torch.zeros(
                (self.max_pts, self.max_elems, 1, 1), dtype=torch.float64, device=device
            )

    def find_rst(self, probes_info, mesh_info, settings, buffers=None):

        # Parse the inputs
        ## Probes information
        probes = probes_info.get("probes", None)
        probes_rst = probes_info.get("probes_rst", None)
        el_owner = probes_info.get("el_owner", None)
        glb_el_owner = probes_info.get("glb_el_owner", None)
        rank_owner = probes_info.get("rank_owner", None)
        err_code = probes_info.get("err_code", None)
        test_pattern = probes_info.get("test_pattern", None)
        rank = probes_info.get("rank", None)
        offset_el = probes_info.get("offset_el", None)
        # Mesh information
        x = mesh_info.get("x", None)
        y = mesh_info.get("y", None)
        z = mesh_info.get("z", None)
        kd_tree = mesh_info.get("kd_tree", None)
        bbox = mesh_info.get("bbox", None)
        bbox_max_dist = mesh_info.get("bbox_max_dist", None)
        # Settings
        not_found_code = settings.get("not_found_code", -10)
        use_test_pattern = settings.get("use_test_pattern", True)
        elem_percent_expansion = settings.get("elem_percent_expansion", 0.01)
        progress_bar = settings.get("progress_bar", False)
        find_pts_tol = settings.get("find_pts_tol", np.finfo(np.double).eps * 10)
        find_pts_max_iterations = settings.get("find_pts_max_iterations", 50)
        # Buffers
        r = buffers.get("r", None)
        s = buffers.get("s", None)
        t = buffers.get("t", None)
        test_interp = buffers.get("test_interp", None)
        # Test patter for found
        test_pattern_for_found = 1e-9

        # Reset the element owner and the error code so this rank checks again
        err_code[:] = not_found_code

        if isinstance(kd_tree, NoneType):
            element_candidates = []
            i = 0
            if progress_bar:
                pbar = tqdm(total=probes.shape[0])
            for pt in probes:
                element_candidates.append([])
                for e in range(0, bbox.shape[0]):
                    if pt_in_bbox(pt, bbox[e], rel_tol=elem_percent_expansion):
                        element_candidates[i].append(e)
                i = i + 1
                if progress_bar:
                    pbar.update(1)
            if progress_bar:
                pbar.close()
        else:

            # Query the tree with the probes to reduce the bbox search
            candidate_elements = kd_tree.query_ball_point(
                x=probes,
                r=bbox_max_dist * (1 + 1e-6),
                p=2.0,
                eps=elem_percent_expansion,
                workers=1,
                return_sorted=False,
                return_length=False,
            )

            # New way of checking as of april 4 2025
            if 0==0:
                element_candidates = refine_candidates(probes, candidate_elements, bbox, rel_tol=elem_percent_expansion)
            else:
                element_candidates = []
                i = 0
                if progress_bar:
                    pbar = tqdm(total=probes.shape[0])
                for pt in probes:
                    element_candidates.append([])
                    for e in candidate_elements[i]:
                        if pt_in_bbox(pt, bbox[e], rel_tol=elem_percent_expansion):
                            element_candidates[i].append(e)
                    i = i + 1
                    if progress_bar:
                        pbar.update(1)
                if progress_bar:
                    pbar.close()

        # Identify variables
        max_pts = self.max_pts
        pts_n = probes.shape[0]
        max_candidate_elements = np.max([len(elist) for elist in element_candidates])
        iterations = np.ceil((pts_n / max_pts))
        checked_elements = [[] for i in range(0, pts_n)]

        exit_flag = False
        # The following logic only works for nelems = 1
        npoints = 10000
        nelems = 1
        for e in range(0, max_candidate_elements):
            if exit_flag:
                break
            for j in range(0, int(iterations)):
                if npoints == 0:
                    exit_flag = True
                    break

                # Get the index of points that have not been found
                pt_not_found_indices = get_points_not_found_index(
                    err_code, checked_elements, element_candidates, max_pts
                )

                # See which element should be checked in this iteration
                elem_to_check_per_point = get_element_to_check(
                    pt_not_found_indices, element_candidates, checked_elements
                )

                # Update the checked elements
                checked_elements = update_checked_elements(
                    checked_elements, pt_not_found_indices, elem_to_check_per_point
                )

                npoints = len(pt_not_found_indices)

                if npoints == 0:
                    exit_flag = True
                    break

                probe_new_shape = (npoints, 1, 1, 1)
                elem_new_shape = (npoints, nelems, x.shape[1], x.shape[2], x.shape[3])

                self.project_element_into_basis(
                    x[elem_to_check_per_point].reshape(elem_new_shape),
                    y[elem_to_check_per_point].reshape(elem_new_shape),
                    z[elem_to_check_per_point].reshape(elem_new_shape),
                )
                r[:npoints, :nelems], s[:npoints, :nelems], t[:npoints, :nelems] = (
                    self.find_rst_from_xyz(
                        probes[pt_not_found_indices, 0].reshape(probe_new_shape),
                        probes[pt_not_found_indices, 1].reshape(probe_new_shape),
                        probes[pt_not_found_indices, 2].reshape(probe_new_shape),
                        tol=find_pts_tol,
                        max_iterations=find_pts_max_iterations,
                    )
                )

                # Reshape results
                result_r = r[:npoints, :nelems, :, :].reshape(
                    (len(pt_not_found_indices))
                )
                result_s = s[:npoints, :nelems, :, :].reshape(
                    (len(pt_not_found_indices))
                )
                result_t = t[:npoints, :nelems, :, :].reshape(
                    (len(pt_not_found_indices))
                )
                result_code_bool = self.point_inside_element[
                    :npoints, :nelems, :, :
                ].reshape((len(pt_not_found_indices)))
                # Assign the error codes

                result_code_bool = result_code_bool.cpu().numpy()
                result_r = result_r.cpu().detach().numpy()
                result_s = result_s.cpu().detach().numpy()
                result_t = result_t.cpu().detach().numpy()

                # Update indices of points that were found and those that were not
                pt_found_this_it = np.where(result_code_bool)[0]
                pt_not_found_this_it = np.where(~result_code_bool)[0]

                # Create a list with the original indices for each of this
                real_index_pt_found_this_it = [
                    pt_not_found_indices[pt_found_this_it[i]]
                    for i in range(0, len(pt_found_this_it))
                ]
                real_index_pt_not_found_this_it = [
                    pt_not_found_indices[pt_not_found_this_it[i]]
                    for i in range(0, len(pt_not_found_this_it))
                ]

                # Update codes for points found in this iteration
                probes_rst[real_index_pt_found_this_it, 0] = result_r[pt_found_this_it]
                probes_rst[real_index_pt_found_this_it, 1] = result_s[pt_found_this_it]
                probes_rst[real_index_pt_found_this_it, 2] = result_t[pt_found_this_it]
                el_owner[real_index_pt_found_this_it] = np.array(
                    elem_to_check_per_point
                )[pt_found_this_it]
                glb_el_owner[real_index_pt_found_this_it] = (
                    el_owner[real_index_pt_found_this_it] + offset_el
                )
                rank_owner[real_index_pt_found_this_it] = rank
                err_code[real_index_pt_found_this_it] = 1

                # If user has selected to check a test pattern:
                if use_test_pattern:

                    # Get shapes
                    ntest = len(pt_not_found_this_it)
                    test_probe_new_shape = (ntest, nelems, 1, 1)
                    test_elem_new_shape = (
                        ntest,
                        nelems,
                        x.shape[1],
                        x.shape[2],
                        x.shape[3],
                    )

                    # Define new arrays (On the cpu)
                    test_elems = np.array(elem_to_check_per_point)[pt_not_found_this_it]
                    test_fields = (
                        x[test_elems, :, :, :] ** 2
                        + y[test_elems, :, :, :] ** 2
                        + z[test_elems, :, :, :] ** 2
                    )
                    test_probes = (
                        probes[real_index_pt_not_found_this_it, 0] ** 2
                        + probes[real_index_pt_not_found_this_it, 1] ** 2
                        + probes[real_index_pt_not_found_this_it, 2] ** 2
                    )

                    # Perform the test interpolation
                    test_interp[:ntest, :nelems] = self.interpolate_field_at_rst(
                        result_r[pt_not_found_this_it].reshape(test_probe_new_shape),
                        result_s[pt_not_found_this_it].reshape(test_probe_new_shape),
                        result_t[pt_not_found_this_it].reshape(test_probe_new_shape),
                        test_fields.reshape(test_elem_new_shape),
                    )
                    test_result = test_interp[:ntest, :nelems].reshape(ntest)

                    # Check if the test pattern is satisfied
                    test_error = abs(test_probes - test_result.cpu().numpy())

                    # Now assign
                    real_list = np.array(real_index_pt_not_found_this_it)
                    relative_list = np.array(pt_not_found_this_it)
                    better_test = np.where(
                        test_error < test_pattern[real_index_pt_not_found_this_it]
                    )[0]
                    #set_as_found = np.where(test_error < test_pattern_for_found)[0]

                    if len(better_test) > 0:
                        probes_rst[real_list[better_test], 0] = result_r[
                            relative_list[better_test]
                        ]
                        probes_rst[real_list[better_test], 1] = result_s[
                            relative_list[better_test]
                        ]
                        probes_rst[real_list[better_test], 2] = result_t[
                            relative_list[better_test]
                        ]
                        el_owner[real_list[better_test]] = np.array(
                            elem_to_check_per_point
                        )[relative_list[better_test]]
                        glb_el_owner[real_list[better_test]] = (
                            el_owner[real_list[better_test]] + offset_el
                        )
                        rank_owner[real_list[better_test]] = rank
                        err_code[real_list[better_test]] = not_found_code
                        test_pattern[real_list[better_test]] = test_error[better_test]

                    # if len(set_as_found) > 0:
                    #    err_code[real_list[set_as_found]] = 1

                else:

                    probes_rst[real_index_pt_not_found_this_it, 0] = result_r[
                        pt_not_found_this_it
                    ]
                    probes_rst[real_index_pt_not_found_this_it, 1] = result_s[
                        pt_not_found_this_it
                    ]
                    probes_rst[real_index_pt_not_found_this_it, 2] = result_t[
                        pt_not_found_this_it
                    ]
                    el_owner[real_index_pt_not_found_this_it] = np.array(
                        elem_to_check_per_point
                    )[pt_not_found_this_it]
                    glb_el_owner[real_index_pt_not_found_this_it] = (
                        el_owner[real_index_pt_not_found_this_it] + offset_el
                    )
                    rank_owner[real_index_pt_not_found_this_it] = rank
                    err_code[real_index_pt_not_found_this_it] = not_found_code

        return (
            probes,
            probes_rst,
            el_owner,
            glb_el_owner,
            rank_owner,
            err_code,
            test_pattern,
        )

    def interpolate_field_from_rst(
        self, probes_info, interpolation_buffer=None, sampled_field=None, settings=None
    ):

        # Parse the inputs
        ## Probes information
        probes = probes_info.get("probes", None)
        probes_rst = probes_info.get("probes_rst", None)
        el_owner = probes_info.get("el_owner", None)
        err_code = probes_info.get("err_code", None)
        # Settings
        if not isinstance(settings, NoneType):
            progress_bar = settings.get("progress_bar", False)
        else:
            progress_bar = False

        max_pts = self.max_pts
        pts_n = probes.shape[0]
        iterations = np.ceil((pts_n / max_pts))

        sampled_field_at_probe = np.empty((probes.shape[0]))
        probe_interpolated = np.zeros((probes.shape[0]))

        for i in range(0, int(iterations)):

            # Check the probes to interpolate this iteration
            probes_to_interpolate = np.where(
                (err_code != 0) & (probe_interpolated == 0)
            )[0]
            probes_to_interpolate = probes_to_interpolate[:max_pts]

            # Check the number of probes
            npoints = len(probes_to_interpolate)
            nelems = 1

            if npoints == 0:
                break

            # Inmediately update the points that will be interpolated
            probe_interpolated[probes_to_interpolate] = 1

            rst_new_shape = (npoints, nelems, 1, 1)
            field_new_shape = (
                npoints,
                nelems,
                sampled_field.shape[1],
                sampled_field.shape[2],
                sampled_field.shape[3],
            )

            interpolation_buffer[:npoints, :nelems] = self.interpolate_field_at_rst(
                probes_rst[probes_to_interpolate, 0].reshape(rst_new_shape),
                probes_rst[probes_to_interpolate, 1].reshape(rst_new_shape),
                probes_rst[probes_to_interpolate, 2].reshape(rst_new_shape),
                sampled_field[el_owner[probes_to_interpolate]].reshape(field_new_shape),
            )

            # Populate the sampled field
            sampled_field_at_probe[probes_to_interpolate] = (
                interpolation_buffer[:npoints, :nelems]
                .reshape(npoints)
                .to("cpu")
                .numpy()
            )

        return sampled_field_at_probe


def determine_initial_guess(self, npoints=1, nelems=1):
    """
    Note: Find a way to evaluate if this routine does help.
    It might be that this is not such a good way of making the guess.
    """
    with torch.no_grad():
        self.rj[:npoints, :nelems, :, :] = 0 + 1e-6
        self.sj[:npoints, :nelems, :, :] = 0 + 1e-6
        self.tj[:npoints, :nelems, :, :] = 0 + 1e-6

    return


def pt_in_bbox(pt, bbox, rel_tol=0.01):
    """Determine if point is inside bounding boxes"""
    # rel_tol=1% enlargement of the bounding box by default

    state = False
    found_x = False
    found_y = False
    found_z = False

    d = bbox[1] - bbox[0]
    tol = d * rel_tol / 2
    if pt[0] >= bbox[0] - tol and pt[0] <= bbox[1] + tol:
        found_x = True

    d = bbox[3] - bbox[2]
    tol = d * rel_tol / 2
    if pt[1] >= bbox[2] - tol and pt[1] <= bbox[3] + tol:
        found_y = True

    d = bbox[5] - bbox[4]
    tol = d * rel_tol / 2
    if pt[2] >= bbox[4] - tol and pt[2] <= bbox[5] + tol:
        found_z = True

    if found_x is True and found_y is True and found_z is True:
        state = True
    else:
        state = False

    return state

def pt_in_bbox_vectorized(pt, bboxes, rel_tol):
    """
    Check if a point (pt) is inside multiple bounding boxes.
    
    Parameters:
        pt : array-like of shape (3,)
            The (x, y, z) coordinates of the point.
        bboxes : ndarray of shape (N, 6)
            Each row is [xmin, xmax, ymin, ymax, zmin, zmax].
        rel_tol : float
            Relative tolerance used to expand the bounding box.
    
    Returns:
        mask : ndarray of shape (N,)
            Boolean array where True indicates the point is within the expanded bbox.
    """
    # For the x dimension:
    dx = bboxes[:, 1] - bboxes[:, 0]
    tol_x = dx * rel_tol / 2.0
    lower_x = bboxes[:, 0] - tol_x
    upper_x = bboxes[:, 1] + tol_x

    # For the y dimension:
    dy = bboxes[:, 3] - bboxes[:, 2]
    tol_y = dy * rel_tol / 2.0
    lower_y = bboxes[:, 2] - tol_y
    upper_y = bboxes[:, 3] + tol_y

    # For the z dimension:
    dz = bboxes[:, 5] - bboxes[:, 4]
    tol_z = dz * rel_tol / 2.0
    lower_z = bboxes[:, 4] - tol_z
    upper_z = bboxes[:, 5] + tol_z

    return ((pt[0] >= lower_x) & (pt[0] <= upper_x) &
            (pt[1] >= lower_y) & (pt[1] <= upper_y) &
            (pt[2] >= lower_z) & (pt[2] <= upper_z))

def refine_candidates(probes, candidate_elements, bboxes, rel_tol):
    """
    Refine candidate elements for each probe by keeping only those where the probe 
    lies within the corresponding expanded bounding box.
    
    Parameters:
        probes : ndarray of shape (N, 3)
            The (x, y, z) coordinates of each probe.
        candidate_elements : list of lists
            Each inner list contains candidate bbox indices (from a kd-tree query) for a probe.
        bboxes : ndarray of shape (M, 6)
            All bounding boxes, each row is [xmin, xmax, ymin, ymax, zmin, zmax].
        rel_tol : float
            Relative tolerance (expansion factor) for the bbox check.
    
    Returns:
        refined_candidates : list of lists
            For each probe, a list of candidate indices for which the point lies inside the bbox.
    """
    refined_candidates = []
    for i, pt in enumerate(probes):
        cands = candidate_elements[i]
        if cands:  # if non-empty
            # Convert candidate indices to a numpy array
            cands = np.array(cands, dtype=int)
            # Get the corresponding bounding boxes
            candidate_bboxes = bboxes[cands]
            # Vectorized check: get a boolean mask for candidates that pass the bbox test
            mask = pt_in_bbox_vectorized(pt, candidate_bboxes, rel_tol)
            refined_candidates.append(cands[mask].tolist())
        else:
            refined_candidates.append([])
    return refined_candidates

def get_points_not_found_index_slow_obsolete(err_code, checked_elements, element_candidates, max_pts):
    # Get the index of points that have not been found
    pt_not_found_indices = np.where(err_code != 1)[0]
    # Get the indices of these points that still have elements remaining to check
    pt_not_found_indices = pt_not_found_indices[
        np.where(
            [
                len(checked_elements[i]) < len(element_candidates[i])
                for i in pt_not_found_indices
            ]
        )[0]
    ]
    # Select only the maximum number of points
    pt_not_found_indices = pt_not_found_indices[:max_pts]
    return pt_not_found_indices

def get_points_not_found_index(err_code, checked_elements, element_candidates, max_pts):
    # Find candidate indices where err_code != 1
    candidate_idx = np.flatnonzero(err_code != 1)
    result = []
    for i in candidate_idx:
        if len(checked_elements[i]) < len(element_candidates[i]):
            result.append(i)
            if len(result) >= max_pts:
                break
    return np.array(result)


def get_element_to_check(pt_not_found_indices, element_candidates, checked_elements):

    # See which element should be checked in this iteration
    temp_candidates = [element_candidates[i] for i in pt_not_found_indices]
    temp_checked = [checked_elements[i] for i in pt_not_found_indices]
    temp_to_check_ = [
        list(set(temp_candidates[i]) - set(temp_checked[i]))
        for i in range(len(temp_candidates))
    ]
    # Sort them by order of closeness
    temp_to_check = [
        sorted(temp_to_check_[i], key=temp_candidates[i].index)
        for i in range(len(temp_candidates))
    ]

    elem_to_check_per_point = [elist[0] for elist in temp_to_check]

    return elem_to_check_per_point


def update_checked_elements(
    checked_elements, pt_not_found_indices, elem_to_check_per_point
):
    for i in range(0, len(pt_not_found_indices)):
        checked_elements[pt_not_found_indices[i]].append(elem_to_check_per_point[i])
    return checked_elements


def invert_jac(jac):
    """
    Invert the jacobian matrix
    """

    jac_inv = torch.zeros_like(jac)

    a = jac[:, :, 0, 0]
    b = jac[:, :, 0, 1]
    c = jac[:, :, 0, 2]
    d = jac[:, :, 1, 0]
    e = jac[:, :, 1, 1]
    f = jac[:, :, 1, 2]
    g = jac[:, :, 2, 0]
    h = jac[:, :, 2, 1]
    i = jac[:, :, 2, 2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    jac_inv[:, :, 0, 0] = (e * i - f * h) / det
    jac_inv[:, :, 0, 1] = (c * h - b * i) / det
    jac_inv[:, :, 0, 2] = (b * f - c * e) / det
    jac_inv[:, :, 1, 0] = (f * g - d * i) / det
    jac_inv[:, :, 1, 1] = (a * i - c * g) / det
    jac_inv[:, :, 1, 2] = (c * d - a * f) / det
    jac_inv[:, :, 2, 0] = (d * h - e * g) / det
    jac_inv[:, :, 2, 1] = (b * g - a * h) / det
    jac_inv[:, :, 2, 2] = (a * e - b * d) / det

    return jac_inv
