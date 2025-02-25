""" Module for the LegendreInterpolator class with a single point"""

import numpy as np
from .single_point_interpolator import SinglePointInterpolator
from .single_point_helper_functions import (
    legendre_basis_at_xtest,
    legendre_basis_derivative_at_xtest,
    apply_operators_3d,
    get_legendre_transformation_matrices,
)


class LegendreInterpolator(SinglePointInterpolator):
    """Class to interpolate a point using legendre basis"""

    def __init__(self, n):
        # Initialize parent class
        super().__init__(n)

        # Get reference element
        self.v_inv = None
        self.v1d_inv = None
        self.v1d = None
        self.v1d_inv = None
        get_legendre_transformation_matrices(self)

    def project_element_into_basis(self, x_e, y_e, z_e, apply_1d_ops=True):

        # Assing the inputs to proper formats
        self.x_e[:, 0] = x_e[:, :, :].reshape(-1, 1)[:, 0]
        self.y_e[:, 0] = y_e[:, :, :].reshape(-1, 1)[:, 0]
        self.z_e[:, 0] = z_e[:, :, :].reshape(-1, 1)[:, 0]

        if not apply_1d_ops:
            self.x_e_hat = self.v_inv @ self.x_e
            self.y_e_hat = self.v_inv @ self.y_e
            self.z_e_hat = self.v_inv @ self.z_e
        elif apply_1d_ops:
            self.x_e_hat = apply_operators_3d(
                self.v1d_inv, self.v1d_inv, self.v1d_inv, self.x_e
            )
            self.y_e_hat = apply_operators_3d(
                self.v1d_inv, self.v1d_inv, self.v1d_inv, self.y_e
            )
            self.z_e_hat = apply_operators_3d(
                self.v1d_inv, self.v1d_inv, self.v1d_inv, self.z_e
            )

        return

    def get_xyz_from_rst(self, rj, sj, tj, apply_1d_ops=True):

        n = self.n
        self.rj[0] = rj
        self.sj[0] = sj
        self.tj[0] = tj

        # Find the basis for each coordinate separately
        ortho_basis_rj = legendre_basis_at_xtest(n, self.rj)
        ortho_basis_sj = legendre_basis_at_xtest(n, self.sj)
        ortho_basis_tj = legendre_basis_at_xtest(n, self.tj)

        ortho_basis_prm_rj = legendre_basis_derivative_at_xtest(ortho_basis_rj, self.rj)
        ortho_basis_prm_sj = legendre_basis_derivative_at_xtest(ortho_basis_sj, self.sj)
        ortho_basis_prm_tj = legendre_basis_derivative_at_xtest(ortho_basis_tj, self.tj)

        if not apply_1d_ops:
            # Construct the 3d basis
            ortho_basis_rstj = np.kron(
                ortho_basis_tj.T, np.kron(ortho_basis_sj.T, ortho_basis_rj.T)
            )

            ortho_basis_drj = np.kron(
                ortho_basis_tj.T, np.kron(ortho_basis_sj.T, ortho_basis_prm_rj.T)
            )
            ortho_basis_dsj = np.kron(
                ortho_basis_tj.T, np.kron(ortho_basis_prm_sj.T, ortho_basis_rj.T)
            )
            ortho_basis_dtj = np.kron(
                ortho_basis_prm_tj.T, np.kron(ortho_basis_sj.T, ortho_basis_rj.T)
            )

            x = (ortho_basis_rstj @ self.x_e_hat)[0, 0]
            y = (ortho_basis_rstj @ self.y_e_hat)[0, 0]
            z = (ortho_basis_rstj @ self.z_e_hat)[0, 0]

            self.jac[0, 0] = (ortho_basis_drj @ self.x_e_hat)[0, 0]
            self.jac[0, 1] = (ortho_basis_dsj @ self.x_e_hat)[0, 0]
            self.jac[0, 2] = (ortho_basis_dtj @ self.x_e_hat)[0, 0]

            self.jac[1, 0] = (ortho_basis_drj @ self.y_e_hat)[0, 0]
            self.jac[1, 1] = (ortho_basis_dsj @ self.y_e_hat)[0, 0]
            self.jac[1, 2] = (ortho_basis_dtj @ self.y_e_hat)[0, 0]

            self.jac[2, 0] = (ortho_basis_drj @ self.z_e_hat)[0, 0]
            self.jac[2, 1] = (ortho_basis_dsj @ self.z_e_hat)[0, 0]
            self.jac[2, 2] = (ortho_basis_dtj @ self.z_e_hat)[0, 0]

        elif apply_1d_ops:
            # Apply the 1d operators to the 3d field
            x = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.x_e_hat
            )[0, 0]
            y = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.y_e_hat
            )[0, 0]
            z = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.z_e_hat
            )[0, 0]

            self.jac[0, 0] = apply_operators_3d(
                ortho_basis_prm_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.x_e_hat
            )[0, 0]
            self.jac[0, 1] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_prm_sj.T, ortho_basis_tj.T, self.x_e_hat
            )[0, 0]
            self.jac[0, 2] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_prm_tj.T, self.x_e_hat
            )[0, 0]

            self.jac[1, 0] = apply_operators_3d(
                ortho_basis_prm_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.y_e_hat
            )[0, 0]
            self.jac[1, 1] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_prm_sj.T, ortho_basis_tj.T, self.y_e_hat
            )[0, 0]
            self.jac[1, 2] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_prm_tj.T, self.y_e_hat
            )[0, 0]

            self.jac[2, 0] = apply_operators_3d(
                ortho_basis_prm_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.z_e_hat
            )[0, 0]
            self.jac[2, 1] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_prm_sj.T, ortho_basis_tj.T, self.z_e_hat
            )[0, 0]
            self.jac[2, 2] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_prm_tj.T, self.z_e_hat
            )[0, 0]

        return x, y, z

    def determine_initial_guess(self):

        self.rj[0] = 0 + 1e-6
        self.sj[0] = 0 + 1e-6
        self.tj[0] = 0 + 1e-6

        return

    def alloc_result_buffer(self, *args, **kwargs):
        return 1
