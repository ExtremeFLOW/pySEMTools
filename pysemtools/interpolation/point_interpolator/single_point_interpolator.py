""" Interface for single point interpolation"""

from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from .single_point_helper_functions import (
    apply_operators_3d,
    lag_interp_matrix_at_xtest,
    get_reference_element,
)

NoneType = type(None)


class SinglePointInterpolator(ABC):
    """Interface for single point interpolation"""

    def __init__(self, n):

        # Order of the element
        self.n = n

        # Attibutes concerning the reference element
        self.x_gll = None
        self.w_gll = None
        self.x_e = None
        self.y_e = None
        self.z_e = None
        get_reference_element(self)

        # Allocate relevant arrays
        self.xj = np.zeros(1)
        self.yj = np.zeros(1)
        self.zj = np.zeros(1)
        self.rj = np.zeros_like(self.xj)
        self.sj = np.zeros_like(self.yj)
        self.tj = np.zeros_like(self.zj)
        self.rstj = np.zeros((3, 1))
        self.eps_rst = np.ones((3, 1))
        self.jac = np.zeros((3, 3))
        self.iterations = 0
        self.field_e = np.zeros_like(self.x_e)
        self.point_inside_element = False

        # dummy variabels
        self.x_e_hat = None
        self.y_e_hat = None
        self.z_e_hat = None
        
        # If interpolate_field_at_rst_vector() happened once
        self.if_interpolated_vector = False

    def find_rst_from_xyz(
        self, xj, yj, zj, tol=np.finfo(np.double).eps * 10, max_iterations=50
    ):
        """Find the rst coordinates from the xyz coordinates using the newton method"""
        self.point_inside_element = False

        self.xj[0] = xj
        self.yj[0] = yj
        self.zj[0] = zj

        # Determine the initial conditions
        self.determine_initial_guess()

        # Use the newton method to identify the coordinates
        self.iterations = 0
        self.eps_rst[:, :] = 1
        while np.linalg.norm(self.eps_rst) > tol and self.iterations < max_iterations:

            # Update the guess
            self.rstj[0, 0] = self.rj[0]
            self.rstj[1, 0] = self.sj[0]
            self.rstj[2, 0] = self.tj[0]

            # Estimate the xyz values from rst and also the jacobian (it is updated inside self.jac)
            xj_found, yj_found, zj_found = self.get_xyz_from_rst(
                self.rj[0], self.sj[0], self.tj[0]
            )

            # Find the residuals and the jacobian inverse.
            self.eps_rst[0, 0] = self.xj[0] - xj_found
            self.eps_rst[1, 0] = self.yj[0] - yj_found
            self.eps_rst[2, 0] = self.zj[0] - zj_found
            jac_inv = np.linalg.inv(self.jac)

            # Find the new guess
            self.rstj = self.rstj - (0 - jac_inv @ self.eps_rst)

            # Update the values
            self.rj[0] = self.rstj[0, 0]
            self.sj[0] = self.rstj[1, 0]
            self.tj[0] = self.rstj[2, 0]
            self.iterations += 1

        limit = 1 + np.finfo(np.single).eps
        if (
            abs(self.rj[0]) <= limit
            and abs(self.sj[0]) <= limit
            and abs(self.tj[0]) <= limit
            and self.iterations < max_iterations
        ):
            self.point_inside_element = True
        else:
            self.point_inside_element = False

        return self.rj[0], self.sj[0], self.tj[0]

    def interpolate_field_at_rst(self, rj, sj, tj, field_e, apply_1d_ops=True):
        """interpolate field at rst"""
        r_j = np.ones((1))
        s_j = np.ones((1))
        t_j = np.ones((1))

        r_j[0] = rj
        s_j[0] = sj
        t_j[0] = tj

        self.field_e[:, 0] = field_e[:, :, :].reshape(-1, 1)[:, 0]

        lk_r = lag_interp_matrix_at_xtest(self.x_gll, r_j)
        lk_s = lag_interp_matrix_at_xtest(self.x_gll, s_j)
        lk_t = lag_interp_matrix_at_xtest(self.x_gll, t_j)

        if not apply_1d_ops:
            lk_3d = np.kron(lk_t.T, np.kron(lk_s.T, lk_r.T))
            field_at_rst = (lk_3d @ self.field_e)[0, 0]
        elif apply_1d_ops:
            field_at_rst = apply_operators_3d(lk_r.T, lk_s.T, lk_t.T, self.field_e)[
                0, 0
            ]

        return field_at_rst

    def interpolate_field_at_rst_vector(self, rj, sj, tj, field_e, apply_1d_ops=True):
        """Interpolate field at all points of an element at once"""
        """Only intended to cases: """
        """where the interpolation weights are the same for all elements"""
        r_j = np.ones((rj.size))
        s_j = np.ones((sj.size))
        t_j = np.ones((tj.size))

        r_j[:] = rj[:]
        s_j[:] = sj[:]
        t_j[:] = tj[:]

        self.field_e[:, 0] = field_e[:, :, :].reshape(-1, 1)[:, 0]

        if self.if_interpolated_vector == False:
            self.lk_r = lag_interp_matrix_at_xtest(self.x_gll, r_j)
            self.lk_s = lag_interp_matrix_at_xtest(self.x_gll, s_j)
            self.lk_t = lag_interp_matrix_at_xtest(self.x_gll, t_j)

        if not apply_1d_ops:
            if self.if_interpolated_vector == False:
                self.lk_3d = np.kron(self.lk_t.T, np.kron(self.lk_s.T, self.lk_r.T))
            self.if_interpolated_vector = True
            field_at_rst = self.lk_3d @ self.field_e
        elif apply_1d_ops:
            self.if_interpolated_vector = True
            field_at_rst = apply_operators_3d(self.lk_r.T, self.lk_s.T, self.lk_t.T, self.field_e)

        field_at_rst = field_at_rst.reshape((t_j.size, s_j.size, r_j.size))

        return field_at_rst

    def find_rst(self, probes_info, mesh_info, settings, buffers=None):
        """Find rst from probes list. Include logic for iterations"""
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

        if progress_bar:
            pbar = tqdm(total=probes.shape[0])
        for pts in range(0, probes.shape[0]):
            if err_code[pts] != 1:
                for e in element_candidates[pts]:
                    self.project_element_into_basis(
                        x[e, :, :, :], y[e, :, :, :], z[e, :, :, :]
                    )
                    r, s, t = self.find_rst_from_xyz(
                        probes[pts, 0],
                        probes[pts, 1],
                        probes[pts, 2],
                        tol=find_pts_tol,
                        max_iterations=find_pts_max_iterations,
                    )
                    if self.point_inside_element:
                        probes_rst[pts, 0] = r
                        probes_rst[pts, 1] = s
                        probes_rst[pts, 2] = t
                        el_owner[pts] = e
                        glb_el_owner[pts] = e + offset_el
                        rank_owner[pts] = rank
                        err_code[pts] = 1
                        break
                    else:

                        # Perform test interpolation and update if the
                        # results are better than previously stored
                        if use_test_pattern:
                            test_field = (
                                x[e, :, :, :] ** 2
                                + y[e, :, :, :] ** 2
                                + z[e, :, :, :] ** 2
                            )
                            test_probe = (
                                probes[pts, 0] ** 2
                                + probes[pts, 1] ** 2
                                + probes[pts, 2] ** 2
                            )
                            test_interp = self.interpolate_field_at_rst(
                                r, s, t, test_field
                            )

                            test_error = abs(test_probe - test_interp)

                            if test_error < test_pattern[pts]:
                                probes_rst[pts, 0] = r
                                probes_rst[pts, 1] = s
                                probes_rst[pts, 2] = t
                                el_owner[pts] = e
                                glb_el_owner[pts] = e + offset_el
                                rank_owner[pts] = rank
                                err_code[pts] = not_found_code
                                test_pattern[pts] = test_error

                        # Otherwise progressively update
                        else:
                            probes_rst[pts, 0] = r
                            probes_rst[pts, 1] = s
                            probes_rst[pts, 2] = t
                            el_owner[pts] = e
                            glb_el_owner[pts] = e + offset_el
                            rank_owner[pts] = rank
                            err_code[pts] = not_found_code

            if progress_bar:
                pbar.update(1)
        if progress_bar:
            pbar.close()

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
        """Interpolate fields for probes locations. Include logic for iterations"""

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

        sampled_field_at_probe = np.empty((probes.shape[0]))

        i = 0  # Counter for the number of probes
        if progress_bar:
            pbar = tqdm(total=probes.shape[0])
        for e in el_owner:
            if err_code[i] != 0:
                # self.project_element_into_basis(x[e,:,:,:], y[e,:,:,:], z[e,:,:,:])
                tmp = self.interpolate_field_at_rst(
                    probes_rst[i, 0],
                    probes_rst[i, 1],
                    probes_rst[i, 2],
                    sampled_field[e, :, :, :],
                )
                sampled_field_at_probe[i] = tmp
            else:
                sampled_field_at_probe[i] = 0
            i = i + 1
            if progress_bar:
                pbar.update(1)
        if progress_bar:
            pbar.close()

        return sampled_field_at_probe

    @abstractmethod
    def project_element_into_basis(self):
        """Project element into appropiate basis"""

    @abstractmethod
    def get_xyz_from_rst(self):
        """Get xyz from rst coordinates"""

    @abstractmethod
    def determine_initial_guess(self):
        """Determine the initial guess for the newton method"""

    @abstractmethod
    def alloc_result_buffer(self):
        """Allocate any necesary buffer for the probe interpolation"""


def pt_in_bbox(pt, bbox, rel_tol=0.01):
    """Check if a point is inside a bounding box with a relative tolerance"""
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
