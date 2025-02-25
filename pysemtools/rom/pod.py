"""Module that contain classes that wrap the svd to facilitate the use of the POD"""

import logging
from .svd import SVD as spSVD_c
from .math_ops import MathOps as math_ops_c
from ..monitoring.logger import Logger as logger_c


NoneType = type(None)


class POD:
    """Class that wraps the SVD to facilitate the use of the POD"""

    def __init__(
        self,
        comm,
        number_of_modes_to_update=1,
        global_updates=True,
        auto_expand=False,
        auto_expand_from_these_modes=1,
    ):

        # Initialize parameters
        self.k = number_of_modes_to_update
        self.setk = number_of_modes_to_update
        self.mink = auto_expand_from_these_modes
        self.ifgbl_update = global_updates
        self.ifautoupdate = auto_expand
        self.minimun_orthogonality_ratio = 0.99
        self.ortho = None

        # Change k to a low value if autoupdate is required
        if self.ifautoupdate:
            self.k = self.mink
        self.running_ra = []

        # Intance the modes and time coeff
        self.u_1t = None
        self.d_1t = None
        self.vt_1t = None

        # Instant the math functions that will help
        self.log = logger_c(comm=comm, module_name="pod")
        self.svd = spSVD_c(self.log)
        self.math = math_ops_c()

        # Number of updates
        self.number_of_updates = 0

        self.log.write("info", "POD Object initialized")

    def check_snapshot_orthogonality(self, comm, xi=None):
        """Check the level of orthogonality of the new snapshot with the current basis"""

        # Calculate the residual and check if basis needs to be expanded
        if self.number_of_updates >= 1:
            if self.ifautoupdate:
                if not self.ifgbl_update:
                    ra = self.math.get_perp_ratio(self.u_1t, xi.reshape((-1, 1)))
                    self.running_ra.append(ra)
                else:
                    ra = self.math.mpi_get_perp_ratio(
                        self.u_1t, xi.reshape((-1, 1)), comm
                    )
                    self.running_ra.append(ra)
            else:
                ra = 0
                self.running_ra.append(ra)

            if (
                self.ifautoupdate is True
                and ra >= self.minimun_orthogonality_ratio
                and self.k < self.setk
            ):
                self.k += 1
                print("New k is = " + repr(self.k))

    def update(self, comm, buff=None):
        """Update POD modes from a batch of snapshots in buff"""

        # Perform the update
        self.log.write("info", "Updating the POD modes")
        if self.ifgbl_update:
            self.u_1t, self.d_1t, self.vt_1t = self.svd.gbl_update(
                self.u_1t, self.d_1t, self.vt_1t, buff[:, :], self.k, comm
            )
        else:
            self.u_1t, self.d_1t, self.vt_1t = self.svd.lcl_update(
                self.u_1t, self.d_1t, self.vt_1t, buff[:, :], self.k
            )

        self.number_of_updates += 1

        string = f"The shape of the modes after this update is U[{self.u_1t.shape[0]},{self.u_1t.shape[1]}]"
        self.log.write("info", string)

        self.log.write(
            "info",
            "The total number of updates performed up to now is: "
            + repr(self.number_of_updates),
        )

    def scale_modes(self, comm, bm1sqrt=None, op="div"):
        """Scale the current modes with the given mass matrix and the
        provided opeartion (div or mult)"""

        self.log.write("info", "Rescaling the obtained modes...")
        # Scale the modes back before gathering them
        self.math.scale_data(
            self.u_1t, bm1sqrt, self.u_1t.shape[0], self.u_1t.shape[1], op
        )
        self.log.write("info", "Rescaling the obtained modes... Done")

    def rotate_local_modes_to_global(self, comm):
        """Do a rotation of the current modes into a global basis"""

        # If local updates where made
        if not self.ifgbl_update:
            self.log.write("info", "Obtaining global modes from local ones")
            ## Obtain global modes
            self.u_1t, self.d_1t, self.vt_1t = self.svd.lcl_to_gbl_svd(
                self.u_1t, self.d_1t, self.vt_1t, self.setk, comm
            )
            ## Gather the orthogonality record
            self.ortho = comm.gather(self.running_ra, root=0)
        else:
            self.ortho = self.running_ra
