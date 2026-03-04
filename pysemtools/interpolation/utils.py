"""Contains utility functions for interpolation"""

import numpy as np


def linear_index(i, j, k, lx, ly, lz):
    """Maps from mulit-dimensional index to a single one"""
    l = 0
    return i + lx * ((j - 0) + ly * ((k - 0) + lz * ((l - 0))))


def transform_from_array_to_list(nx, ny, nz, array, use_indices = False):
    """Transform a multi-dimensional array to a list-like array"""

    if use_indices:
        xyz_coords = np.zeros((nx * ny * nz, len(array)))
        for k in range(0, nz):
            for j in range(0, ny):
                for i in range(0, nx):
                    position = linear_index(j, i, k, ny, nx, nz)
                    for ind in range(0, len(array)):
                        xyz_coords[position, ind] = array[ind][i, j, k]
    else:
        xyz_coords = [arr.flatten() for arr in array]
        xyz_coords = np.array(xyz_coords).T

    return xyz_coords


# Inverse transformation to "linear index"
def nonlinear_index(linear_index_, lx, ly, lz):
    """Maps from a single index to a multi-dimensional one"""
    index = np.zeros(4, dtype=int)
    lin_idx = linear_index_
    index[3] = lin_idx / (lx * ly * lz)
    index[2] = (lin_idx - (lx * ly * lz) * index[3]) / (lx * ly)
    index[1] = (lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2]) / lx
    index[0] = (
        lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2] - lx * index[1]
    )

    return index


def transform_from_list_to_array(nx, ny, nz, xyz_coords, use_indices = False):
    """Transform a list-like array to a multi-dimensional array"""
    num_points = xyz_coords.shape[0]

    try:
        len_array = xyz_coords.shape[1]
    except IndexError:
        len_array = 1
        xyz_coords = xyz_coords.reshape(-1, 1)

    if use_indices:
        array = []
        for i in range(0, len_array):
            array.append(np.zeros((nx, ny, nz)))

        for linear_index_ in range(0, num_points):
            index = nonlinear_index(linear_index_, ny, nx, nz)
            j = index[0]
            i = index[1]
            k = index[2]
            for ind in range(0, len(array)):
                array[ind][i, j, k] = xyz_coords[linear_index_, ind]
    else:
        fields = xyz_coords.shape[1]
        array = [xyz_coords[:, field].reshape(nx, ny, nz) for field in range(fields)]

    return array
