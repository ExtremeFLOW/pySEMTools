from mpi4py import MPI  # equivalent to the use of MPI_init() in C
import numpy as np
from .mpi_spSVD import spSVD_c
from .math_ops import math_ops_c
from .logger import logger_c
import logging
import os

NoneType = type(None)


class POD_c:

    def __init__(
        self,
        comm,
        number_of_modes_to_update=1,
        global_updates=True,
        auto_expand=False,
        auto_expand_from_these_modes=1,
        threads=1,
    ):

        rank = comm.Get_rank()
        size = comm.Get_size()

        ## Set number of threads to be used
        # os.environ["OMP_NUM_THREADS"] = str(threads)
        # os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        # os.environ["MKL_NUM_THREADS"] = str(threads)
        # os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
        # os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

        # Initialize parameters
        self.k = number_of_modes_to_update
        self.setk = number_of_modes_to_update
        self.mink = auto_expand_from_these_modes
        self.ifgbl_update = global_updates
        self.ifautoupdate = auto_expand

        # Change k to a low value if autoupdate is required
        if self.ifautoupdate == True:
            self.k = self.mink
        self.running_ra = []

        # Intance the modes and time coeff
        self.U_1t = None
        self.D_1t = None
        self.Vt_1t = None

        # Instant the math functions that will help
        self.log = logger_c(level=logging.DEBUG, comm=comm, module_name="pod")
        self.svd = spSVD_c(self.log)
        self.math = math_ops_c()

        # Number of updates
        self.number_of_updates = 0

        self.log.write("info", "POD Object initialized")

        return

    def check_snapshot_orthogonality(self, comm, Xi=None):

        # Calculate the residual and check if basis needs to be expanded
        if self.number_of_updates >= 1:
            if self.ifautoupdate == True:
                if self.ifgbl_update == False:
                    ra = self.math.get_perp_ratio(self.U_1t, Xi.reshape((-1, 1)))
                    self.running_ra.append(ra)
                else:
                    ra = self.math.mpi_get_perp_ratio(
                        self.U_1t, Xi.reshape((-1, 1)), comm
                    )
                    self.running_ra.append(ra)
            else:
                ra = 0
                self.running_ra.append(ra)

            if (
                self.ifautoupdate == True
                and ra >= self.minimun_orthogonality_ratio
                and self.k < self.maximun_number_of_modes
            ):
                self.k += 1
                print("New k is = " + repr(self.k))

        return

    def update(self, comm, buff=None):

        # Get rank info
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Perform the update
        if self.ifgbl_update == True:
            self.U_1t, self.D_1t, self.Vt_1t = self.svd.gblSVD_update_fromBatch(
                self.U_1t, self.D_1t, self.Vt_1t, buff[:, :], self.k, comm
            )
        else:
            self.U_1t, self.D_1t, self.Vt_1t = self.svd.lclSVD_update_fromBatch(
                self.U_1t, self.D_1t, self.Vt_1t, buff[:, :], self.k
            )

        self.number_of_updates += 1

        string = "The shape of the modes after this update is U[%d,%d]" % (
            self.U_1t.shape[0],
            self.U_1t.shape[1],
        )
        self.log.write("info", string)

        self.log.write(
            "info",
            "The total number of updates performed up to now is: "
            + repr(self.number_of_updates),
        )

        return

    def scale_modes(self, comm, bm1sqrt=None, op="div"):

        self.log.write("info", "Rescaling the obtained modes...")
        # Scale the modes back before gathering them
        self.math.scale_data(
            self.U_1t, bm1sqrt, self.U_1t.shape[0], self.U_1t.shape[1], op
        )
        self.log.write("info", "Rescaling the obtained modes... Done")

    def rotate_local_modes_to_global(self, comm):

        # If local updates where made
        if self.ifgbl_update == False:
            self.log.write("info", "Obtaining global modes from local ones")
            ## Obtain global modes
            self.U_1t, self.D_1t, self.Vt_1t = self.svd.lclSVD_to_gblSVD(
                self.U_1t, self.D_1t, self.Vt_1t, self.setk, comm
            )
            ## Gather the orthogonality record
            ortho = comm.gather(self.running_ra, root=0)
        else:
            ortho = self.running_ra

        return
