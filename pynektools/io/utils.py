import numpy as np


class io_path_data:
    def __init__(self, params_file):
        self.casename = params_file["casename"]
        self.dataPath = params_file["dataPath"]
        self.index = params_file["first_index"]


def get_fld_from_ndarray(array, lx, ly, lz, nelv):

    fld = array.reshape((nelv, lz, ly, lx))

    return fld
