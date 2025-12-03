""" Contains class to interpolate multiple points using numpy"""

import os
import numpy as np
from tqdm import tqdm
from .multiple_point_interpolator import MultiplePointInterpolator
from .multiple_point_helper_functions_numpy import (
    get_basis_transformation_matrices,
    apply_operators_3d,
    legendre_basis_at_xtest,
    legendre_basis_derivative_at_xtest,
    lag_interp_matrix_at_xtest,
    bar_interp_matrix_at_xtest,
)


NoneType = type(None)
INTERPOLATION_BATCHED_SEARCH = os.getenv("PYSEMTOOLS_INTERPOLATION_BATCHED_SEARCH", "False").lower() in ("true", "1", "t")

class LegendreInterpolator(MultiplePointInterpolator):
    """Class to interpolate multiple points using numpy and Legendre polynomials."""

    def __init__(self, n, max_pts=1, max_elems=1):
        # Initialize parent class
        super().__init__(n, max_pts=max_pts, max_elems=max_elems)

        # Get reference element
        self.v1d = None
        self.v1d_inv = None
        self.v = None
        self.v_inv = None
        get_basis_transformation_matrices(self)

        # Dummy parameters
        self.x_e_hat = None
        self.y_e_hat = None
        self.z_e_hat = None

    def project_element_into_basis(self, x_e, y_e, z_e, apply_1d_ops=True):
        """Project the element data into the appropiate basis"""
        npoints = x_e.shape[0]
        nelems = x_e.shape[1]
        n = x_e.shape[2] * x_e.shape[3] * x_e.shape[4]

        # Assing the inputs to proper formats
        self.x_e[:npoints, :nelems, :, :] = x_e.reshape(npoints, nelems, n, 1)[
            :, :, :, :
        ]
        self.y_e[:npoints, :nelems, :, :] = y_e.reshape(npoints, nelems, n, 1)[
            :, :, :, :
        ]
        self.z_e[:npoints, :nelems, :, :] = z_e.reshape(npoints, nelems, n, 1)[
            :, :, :, :
        ]

        # Get the modal representation
        if not apply_1d_ops:

            self.x_e_hat = np.einsum(
                "ijkl,ijlm->ijkm", self.v_inv, self.x_e[:npoints, :nelems, :, :]
            )
            self.y_e_hat = np.einsum(
                "ijkl,ijlm->ijkm", self.v_inv, self.y_e[:npoints, :nelems, :, :]
            )
            self.z_e_hat = np.einsum(
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
        """
        This function calculates the xyz coordinates from the given rst coordinates for points
        in the elements that were projected into xhat, yhat, zhat.
        """

        npoints = rj.shape[0]
        nelems = self.x_e_hat.shape[1]
        n = self.n

        self.rj[:npoints, :nelems, :, :] = rj[:, :, :, :]
        self.sj[:npoints, :nelems, :, :] = sj[:, :, :, :]
        self.tj[:npoints, :nelems, :, :] = tj[:, :, :, :]

        # Find the basis for each coordinate separately
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
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.x_e_hat,
            )
            y = apply_operators_3d(
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.y_e_hat,
            )
            z = apply_operators_3d(
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.z_e_hat,
            )

            self.jac[:npoints, :nelems, 0, 0] = apply_operators_3d(
                ortho_basis_prm_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.x_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 0, 1] = apply_operators_3d(
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_prm_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.x_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 0, 2] = apply_operators_3d(
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_prm_tj.transpose(0, 1, 3, 2),
                self.x_e_hat,
            )[:, :, 0, 0]

            self.jac[:npoints, :nelems, 1, 0] = apply_operators_3d(
                ortho_basis_prm_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.y_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 1, 1] = apply_operators_3d(
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_prm_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.y_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 1, 2] = apply_operators_3d(
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_prm_tj.transpose(0, 1, 3, 2),
                self.y_e_hat,
            )[:, :, 0, 0]

            self.jac[:npoints, :nelems, 2, 0] = apply_operators_3d(
                ortho_basis_prm_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.z_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 2, 1] = apply_operators_3d(
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_prm_sj.transpose(0, 1, 3, 2),
                ortho_basis_tj.transpose(0, 1, 3, 2),
                self.z_e_hat,
            )[:, :, 0, 0]
            self.jac[:npoints, :nelems, 2, 2] = apply_operators_3d(
                ortho_basis_rj.transpose(0, 1, 3, 2),
                ortho_basis_sj.transpose(0, 1, 3, 2),
                ortho_basis_prm_tj.transpose(0, 1, 3, 2),
                self.z_e_hat,
            )[:, :, 0, 0]

        return x, y, z

    def find_rst_from_xyz(
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

        self.xj[:npoints, :, :, :] = xj[:, :, :, :]
        self.yj[:npoints, :, :, :] = yj[:, :, :, :]
        self.zj[:npoints, :, :, :] = zj[:, :, :, :]

        # Determine the initial conditions
        determine_initial_guess(
            self, npoints=npoints, nelems=nelems
        )  # This populates self.rj, self.sj, self.tj for 1st iteration

        # Use the newton method to identify the coordinates
        self.iterations = 0
        self.eps_rst[:npoints, :nelems, :, :] = 1

        # create an integer array to store the number of iterations that it took for each point
        iterations_per_point = np.zeros_like(xj, dtype=np.int32)
        iterations_per_point[:, :, :, :] = max_iterations
        points_already_found = np.any(iterations_per_point[:npoints, :nelems] < max_iterations, axis=(2,3))  

        while (
            np.any(np.linalg.norm(self.eps_rst[:npoints, :nelems], axis=(2, 3)) > tol)
            and self.iterations < max_iterations
        ):

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
            self.eps_rst[np.where(points_already_found)] = 0

            #jac_inv = np.linalg.inv(self.jac[:npoints, :nelems])
            jac_inv = invert_jac(self.jac[:npoints, :nelems])

            # Find the new guess
            self.rstj[:npoints, :nelems] = self.rstj[:npoints, :nelems] - (
                0
                - np.einsum("ijkl,ijlm->ijkm", jac_inv, self.eps_rst[:npoints, :nelems])
            )

            # Update the values
            self.rj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 0, 0]
            self.sj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 1, 0]
            self.tj[:npoints, :nelems, 0, 0] = self.rstj[:npoints, :nelems, 2, 0]
            self.iterations += 1

            # Determine which points have already been found so they are not updated anymore
            points_found_this_it = (
                np.linalg.norm(self.eps_rst[:npoints, :nelems], axis=(2, 3)) <= tol
            )
            points_already_found = np.any(iterations_per_point[:npoints, :nelems] < max_iterations, axis=(2,3))
            # Update the number of iterations only if the point has newly been found
            iterations_per_point[(points_found_this_it & ~points_already_found)] = self.iterations

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
            self.rj[:npoints, :nelems],
            self.sj[:npoints, :nelems],
            self.tj[:npoints, :nelems],
        )

    def interpolate_field_at_rst(self, rj, sj, tj, field_e, apply_1d_ops=True, formula = 'barycentric'):
        """
        """
        if formula == 'barycentric':
            return self.interpolate_field_at_rst_barycentric(rj, sj, tj, field_e, apply_1d_ops)
        elif formula == 'lagrange':
            return self.interpolate_field_at_rst_lagrange(rj, sj, tj, field_e, apply_1d_ops)

    def interpolate_field_at_rst_lagrange(self, rj, sj, tj, field_e, apply_1d_ops=True):
        """
        Interpolate each point in a given field.
        EACH POINT RECIEVES ONE FIELD! SO FIELDS MIGHT BE DUPLICATED

        """
        npoints = rj.shape[0]
        nelems = rj.shape[1]
        n = field_e.shape[2] * field_e.shape[3] * field_e.shape[4]

        self.rj[:npoints, :nelems, :, :] = rj[:, :, :, :]
        self.sj[:npoints, :nelems, :, :] = sj[:, :, :, :]
        self.tj[:npoints, :nelems, :, :] = tj[:, :, :, :]

        # Assing the inputs to proper formats
        self.field_e[:npoints, :nelems, :, :] = field_e.reshape(npoints, nelems, n, 1)[
            :, :, :, :
        ]

        lk_r = lag_interp_matrix_at_xtest(self.x_gll, self.rj[:npoints, :nelems, :, :])
        lk_s = lag_interp_matrix_at_xtest(self.x_gll, self.sj[:npoints, :nelems, :, :])
        lk_t = lag_interp_matrix_at_xtest(self.x_gll, self.tj[:npoints, :nelems, :, :])

        if not apply_1d_ops:
            raise RuntimeError("Only worrking by applying 1d operators")
        elif apply_1d_ops:
            field_at_rst = apply_operators_3d(
                lk_r.transpose(0, 1, 3, 2),
                lk_s.transpose(0, 1, 3, 2),
                lk_t.transpose(0, 1, 3, 2),
                self.field_e[:npoints, :nelems, :, :],
            )

        return field_at_rst
    
    def interpolate_field_at_rst_barycentric(self, rj, sj, tj, field_e, apply_1d_ops=True):
        """
        Interpolate each point in a given field.
        EACH POINT RECIEVES ONE FIELD! SO FIELDS MIGHT BE DUPLICATED

        """
        npoints = rj.shape[0]
        nelems = rj.shape[1]
        n = field_e.shape[2] * field_e.shape[3] * field_e.shape[4]

        self.rj[:npoints, :nelems, :, :] = rj[:, :, :, :]
        self.sj[:npoints, :nelems, :, :] = sj[:, :, :, :]
        self.tj[:npoints, :nelems, :, :] = tj[:, :, :, :]

        # Assing the inputs to proper formats
        self.field_e[:npoints, :nelems, :, :] = field_e.reshape(npoints, nelems, n, 1)[
            :, :, :, :
        ]

        lk_r = bar_interp_matrix_at_xtest(self.x_gll, self.rj[:npoints, :nelems, :, :], w=self.barycentric_w)
        lk_s = bar_interp_matrix_at_xtest(self.x_gll, self.sj[:npoints, :nelems, :, :], w=self.barycentric_w)
        lk_t = bar_interp_matrix_at_xtest(self.x_gll, self.tj[:npoints, :nelems, :, :], w=self.barycentric_w)

        if not apply_1d_ops:
            raise RuntimeError("Only worrking by applying 1d operators")
        elif apply_1d_ops:
            field_at_rst = apply_operators_3d(
                lk_r.transpose(0, 1, 3, 2),
                lk_s.transpose(0, 1, 3, 2),
                lk_t.transpose(0, 1, 3, 2),
                self.field_e[:npoints, :nelems, :, :],
            )

        return field_at_rst

    def alloc_result_buffer(self, **kwargs):
        dtype = kwargs.get("dtype", "double")

        if dtype == "double":
            return np.zeros((self.max_pts, self.max_elems, 1, 1), dtype=np.double)

    def find_rst(self, probes_info, mesh_info, settings, buffers=None):
        if INTERPOLATION_BATCHED_SEARCH:
            return self.find_rst_batched(probes_info, mesh_info, settings, buffers)
        else:
            return self.find_rst_(probes_info, mesh_info, settings, buffers)

    def find_rst_batched(self, probes_info, mesh_info, settings, buffers=None):

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

        # Reset the element owner and the error code so this rank checks again
        err_code[:] = not_found_code

        max_pts = self.max_pts
        pts_total = probes.shape[0]

        # ---- Batched candidate generation: process blocks of up to max_pts points ----
        for bstart in range(0, pts_total, max_pts):
            bend = min(bstart + max_pts, pts_total)
            batch_idx = np.arange(bstart, bend, dtype=np.intp)  # global indices for this batch

            # Build element_candidates only for this batch
            if isinstance(kd_tree, NoneType):
                element_candidates = []
                i = 0
                if progress_bar:
                    pbar = tqdm(total=batch_idx.shape[0])
                for pt in probes[batch_idx]:
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
                element_candidates = kd_tree.search(probes[batch_idx])

            # Local (per-batch) sizes and pointers
            pts_n = batch_idx.shape[0]
            if pts_n == 0:
                continue
            max_candidate_elements = np.max([len(elist) for elist in element_candidates]) if pts_n > 0 else 0
            iterations = int(np.ceil(pts_n / max_pts))  # == 1 for this per-batch scheme
            next_candidate = np.zeros(pts_n, dtype=int)
            candidate_lengths = np.array([len(elist) for elist in element_candidates], dtype=int)

            exit_flag = False
            # The following logic only works for nelems = 1
            npoints = 10000
            nelems = 1
            for e in range(0, max_candidate_elements):
                if exit_flag:
                    break
                for j in range(0, iterations):
                    if npoints == 0:
                        exit_flag = True
                        break

                    # Get the index (LOCAL) of points in this batch that have not been found
                    local_mask = (err_code[batch_idx] != 1) & (next_candidate < candidate_lengths)
                    pt_not_found_local = np.flatnonzero(local_mask)
                    pt_not_found_local = pt_not_found_local[:max_pts]

                    # Map to GLOBAL indices (used below) and pick candidate element per LOCAL index
                    pt_not_found_indices = batch_idx[pt_not_found_local]
                    elem_to_check_per_point = [element_candidates[i][next_candidate[i]] for i in pt_not_found_local]

                    # Update the checked elements (LOCAL)
                    next_candidate[pt_not_found_local] += 1

                    npoints = len(pt_not_found_local)

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
                    result_r = r[:npoints, :nelems, :, :].reshape((len(pt_not_found_indices)))
                    result_s = s[:npoints, :nelems, :, :].reshape((len(pt_not_found_indices)))
                    result_t = t[:npoints, :nelems, :, :].reshape((len(pt_not_found_indices)))
                    result_code_bool = self.point_inside_element[:npoints, :nelems, :, :].reshape((len(pt_not_found_indices)))

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
                    el_owner[real_index_pt_found_this_it] = np.array(elem_to_check_per_point)[pt_found_this_it]
                    glb_el_owner[real_index_pt_found_this_it] = (el_owner[real_index_pt_found_this_it] + offset_el)
                    rank_owner[real_index_pt_found_this_it] = rank
                    err_code[real_index_pt_found_this_it] = 1

                    # If user has selected to check a test pattern:
                    if use_test_pattern:

                        # Get shapes
                        ntest = len(pt_not_found_this_it)
                        test_probe_new_shape = (ntest, nelems, 1, 1)
                        test_elem_new_shape = (ntest, nelems, x.shape[1], x.shape[2], x.shape[3])

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
                        test_error = abs(test_probes - test_result)

                        # Now assign
                        real_list = np.array(real_index_pt_not_found_this_it)
                        relative_list = np.array(pt_not_found_this_it)
                        better_test = np.where(
                            test_error < test_pattern[real_index_pt_not_found_this_it]
                        )[0]

                        if len(better_test) > 0:
                            probes_rst[real_list[better_test], 0] = result_r[relative_list[better_test]]
                            probes_rst[real_list[better_test], 1] = result_s[relative_list[better_test]]
                            probes_rst[real_list[better_test], 2] = result_t[relative_list[better_test]]
                            el_owner[real_list[better_test]] = np.array(elem_to_check_per_point)[relative_list[better_test]]
                            glb_el_owner[real_list[better_test]] = (el_owner[real_list[better_test]] + offset_el)
                            rank_owner[real_list[better_test]] = rank
                            err_code[real_list[better_test]] = not_found_code
                            test_pattern[real_list[better_test]] = test_error[better_test]

                    else:

                        probes_rst[real_index_pt_not_found_this_it, 0] = result_r[pt_not_found_this_it]
                        probes_rst[real_index_pt_not_found_this_it, 1] = result_s[pt_not_found_this_it]
                        probes_rst[real_index_pt_not_found_this_it, 2] = result_t[pt_not_found_this_it]
                        el_owner[real_index_pt_not_found_this_it] = np.array(elem_to_check_per_point)[pt_not_found_this_it]
                        glb_el_owner[real_index_pt_not_found_this_it] = (el_owner[real_index_pt_not_found_this_it] + offset_el)
                        rank_owner[real_index_pt_not_found_this_it] = rank
                        err_code[real_index_pt_not_found_this_it] = not_found_code

            # end for-e / for-j for this batch

        # end batches

        return (
            probes,
            probes_rst,
            el_owner,
            glb_el_owner,
            rank_owner,
            err_code,
            test_pattern,
        )


    def find_rst_(self, probes_info, mesh_info, settings, buffers=None):

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

            element_candidates = kd_tree.search(probes)
 
        max_pts = self.max_pts
        pts_n = probes.shape[0]
        max_candidate_elements = np.max([len(elist) for elist in element_candidates])
        iterations = np.ceil((pts_n / max_pts))
        # Use pointers instead of checked_elements
        next_candidate = np.zeros(pts_n, dtype=int)
        candidate_lengths = np.array([len(elist) for elist in element_candidates])

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
                pt_not_found_indices = np.flatnonzero((err_code != 1) & (next_candidate < candidate_lengths))
                pt_not_found_indices = pt_not_found_indices[:max_pts]

                # See which element should be checked in this iteration
                elem_to_check_per_point = [element_candidates[i][next_candidate[i]] for i in pt_not_found_indices]

                # Update the checked elements
                next_candidate[pt_not_found_indices] += 1

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
                    test_error = abs(test_probes - test_result)

                    # Now assign
                    real_list = np.array(real_index_pt_not_found_this_it)
                    relative_list = np.array(pt_not_found_this_it)
                    better_test = np.where(
                        test_error < test_pattern[real_index_pt_not_found_this_it]
                    )[0]

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

    def interpolate_field_from_rst(self, probes_info, interpolation_buffer=None, sampled_field=None, settings=None):
        # --- Parse Inputs ---
        probes      = probes_info.get("probes", None)
        probes_rst  = probes_info.get("probes_rst", None)
        el_owner    = probes_info.get("el_owner", None)
        err_code    = probes_info.get("err_code", None)
        
        # Get settings; default progress_bar to False
        progress_bar = settings.get("progress_bar", False) if settings is not None else False

        max_pts = self.max_pts
        num_probes = probes_rst.shape[0]

        # --- Precompute valid indices once ---
        valid_idx = np.nonzero(err_code != 0)[0]  # valid indices only
        n_valid = valid_idx.size

        # Prepare output array
        sampled_field_at_probe = np.zeros(num_probes)

        # --- Process valid indices in batches ---
        for start in range(0, n_valid, max_pts):
            # Select the current batch from the valid indices
            current = valid_idx[start:start + max_pts]
            npoints = current.size

            # Compute new shapes based on npoints (assumes nelems == 1)
            rst_new_shape   = (npoints, 1, 1, 1)
            field_new_shape = (npoints, 1) + sampled_field.shape[1:]
            
            # Call the interpolation routine on the current batch
            interpolation_buffer[:npoints, :1] = self.interpolate_field_at_rst(
                probes_rst[current, 0].reshape(rst_new_shape),
                probes_rst[current, 1].reshape(rst_new_shape),
                probes_rst[current, 2].reshape(rst_new_shape),
                sampled_field[el_owner[current]].reshape(field_new_shape)
            )
                
            # Store the interpolated values back into the result array
            sampled_field_at_probe[current] = interpolation_buffer[:npoints, 0].reshape(npoints)
                
        return sampled_field_at_probe
    
    def get_obb(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, max_pts: int =256, ndummy: int = 1):
        """
        Obtain the oriented bounding boxed of the element as shown by Mittal et al.
        """

        nelv = x.shape[0]
        lz = x.shape[1]
        ly = x.shape[2]
        lx = x.shape[3]

        # Allocate arrays to keep info
        obb_j = np.zeros((nelv, 3, 3), dtype=np.double)
        obb_c = np.zeros((nelv, 3), dtype=np.double)

        # Allocate the rst zeros
        rst = np.zeros((max_pts, ndummy, 1, 1), dtype=np.double)
        elem_data = np.zeros((max_pts, ndummy, lx*ly*lz, 3, 1), dtype=np.double)

        # Loop over the elements based on the max points
        for i in range(0, nelv, max_pts):
            nelems = min(max_pts, nelv - i)
            # Get the points
            x_e = x[i : i + nelems, :, :, :].reshape(nelems, ndummy, lx, ly, lz)
            y_e = y[i : i + nelems, :, :, :].reshape(nelems, ndummy, lx, ly, lz)
            z_e = z[i : i + nelems, :, :, :].reshape(nelems, ndummy, lx, ly, lz)

            # Project them into the basis
            self.project_element_into_basis(x_e, y_e, z_e)

            # Get the centers and their jacobian
            xc, yc, zc = self.get_xyz_from_rst(
                rst[:nelems, :ndummy, :, :],
                rst[:nelems, :ndummy, :, :],
                rst[:nelems, :ndummy, :, :],
            )
            jac = self.jac[:nelems, :ndummy]
            jac_inv = invert_jac(jac)

            # rearange the elements to columns to apply the jacobian
            elem_data[:nelems, :ndummy, :, 0, 0] = x_e.reshape(nelems, ndummy, lx*ly*lz) - xc.reshape(nelems, ndummy, 1)
            elem_data[:nelems, :ndummy, :, 1, 0] = y_e.reshape(nelems, ndummy, lx*ly*lz) - yc.reshape(nelems, ndummy, 1)
            elem_data[:nelems, :ndummy, :, 2, 0] = z_e.reshape(nelems, ndummy, lx*ly*lz) - zc.reshape(nelems, ndummy, 1)

            # Apply the jacobian to each point to get the tranformation
            x_tilde = np.einsum("ijklm, ijkmt -> ijklt" , jac_inv[:, :, np.newaxis, :, :], elem_data[:nelems, :ndummy])

            # Get the center of this reference element xc_moon
            xc_moon = np.mean(x_tilde, axis=(2))

            # Now get the AABB of the reference element
            x_min = np.min(x_tilde, axis=(2))
            x_max = np.max(x_tilde, axis=(2))
            x_diff = x_max - x_min
            # Get the jacobian as indicated by Mittal et al.
            jac_moon = x_diff * np.eye(3) / 2 * (1 + 0.05) # Expand 5% # Hard-coded

            # Save the centers
            xc_t =  np.concatenate((xc, yc, zc), axis=2) + np.matmul(jac, xc_moon)
            obb_c[i : i + nelems, :] = xc_t.reshape(nelems, 3)
            
            # save the jacobian
            jac_t = invert_jac(np.matmul(jac, jac_moon))
            obb_j[i : i + nelems, :, :] = jac_t.reshape(nelems, 3, 3)

        return obb_c, obb_j

def determine_initial_guess(self, npoints=1, nelems=1):
    """
    Note: Find a way to evaluate if this routine does help.
    It might be that this is not such a good way of making the guess.
    """

    self.rj[:npoints, :nelems, :, :] = 0 + 1e-6
    self.sj[:npoints, :nelems, :, :] = 0 + 1e-6
    self.tj[:npoints, :nelems, :, :] = 0 + 1e-6

    return


def pt_in_bbox(pt, bbox, rel_tol=0.01):
    """Determine if point is inside bounding box"""
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

    jac_inv = np.zeros_like(jac)

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