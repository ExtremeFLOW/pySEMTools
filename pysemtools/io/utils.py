"""Utilities for I/O operations"""


class IoPathData:
    """This class stores the path, casename and index of fld inputs"""

    def __init__(self, params_file):
        self.casename = params_file["casename"]
        self.dataPath = params_file["dataPath"]
        self.index = params_file["first_index"]


def get_fld_from_ndarray(array, lx, ly, lz, nelv):
    """Reshape a 1D array obtained from fortran into a 4D array
    compliant with pySEMTools"""

    fld = array.reshape((nelv, lz, ly, lx))

    return fld
