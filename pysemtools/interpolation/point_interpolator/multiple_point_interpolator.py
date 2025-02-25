"""Interface for multiple point interpolators"""

from abc import ABC, abstractmethod
import numpy as np
from .multiple_point_helper_functions_numpy import get_reference_element


class MultiplePointInterpolator(ABC):
    """Interface for multiple point interpolators"""

    def __init__(self, n, max_pts=1, max_elems=1):

        # Order of the element
        self.n = n
        # Maximun of points to interpolate at the same time
        self.max_pts = max_pts
        # Maximun number of elements to interpolate at the same time
        self.max_elems = max_elems

        # Attributes reference element
        self.x_gll = None
        self.w_gll = None
        self.x_e = None
        self.y_e = None
        self.z_e = None
        get_reference_element(self)

        # Allocate relevant arrays
        ## The xyz coordinates are invariant
        self.xj = np.zeros((max_pts, 1, 1, 1))
        self.yj = np.zeros((max_pts, 1, 1, 1))
        self.zj = np.zeros((max_pts, 1, 1, 1))
        ## The rst coordinates depend on the point and on the element
        self.rj = np.zeros((max_pts, max_elems, 1, 1))
        self.sj = np.zeros((max_pts, max_elems, 1, 1))
        self.tj = np.zeros((max_pts, max_elems, 1, 1))
        self.rstj = np.zeros((max_pts, max_elems, 3, 1))
        self.eps_rst = np.ones((max_pts, max_elems, 3, 1))
        self.jac = np.zeros((max_pts, max_elems, 3, 3))
        self.iterations = 0
        self.field_e = np.zeros((max_pts, 1, self.x_e.shape[2], 1))
        self.point_inside_element = np.zeros((max_pts, max_elems, 1, 1), dtype=bool)

    @abstractmethod
    def project_element_into_basis(self):
        """Project the element into the basis"""

    @abstractmethod
    def get_xyz_from_rst(self):
        """Get the xyz coordinates from the rst coordinates"""

    @abstractmethod
    def find_rst_from_xyz(self):
        """Find the rst coordinates from the xyz coordinates using the newton method"""

    @abstractmethod
    def interpolate_field_at_rst(self):
        """Interpolate field from found coordinates"""

    @abstractmethod
    def alloc_result_buffer(self):
        """Allocate buffers to keep results on each iteration"""

    @abstractmethod
    def find_rst(self):
        """Find coordinates for full list of probes
        multiple iteration logic is implemented here"""

    @abstractmethod
    def interpolate_field_from_rst(self):
        """Interpolate data for full list of probes
        multiple iteration logic is implemented here"""
