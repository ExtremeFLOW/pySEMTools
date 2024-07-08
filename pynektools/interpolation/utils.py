import numpy as np


def linear_index(i, j, k, lx, ly, lz):
    l = 0
    return i + lx * ((j - 0) + ly * ((k - 0) + lz * ((l - 0))))


def transform_from_array_to_list(nx, ny, nz, array):

    xyz_coords = np.zeros((nx * ny * nz, len(array)))
    for k in range(0, nz):
        for j in range(0, ny):
            for i in range(0, nx):
                position = linear_index(j, i, k, ny, nx, nz)
                for ind in range(0, len(array)):
                    xyz_coords[position, ind] = array[ind][i, j, k]

    return xyz_coords


# Inverse transformation to "linear index"
def nonlinear_index(linear_index, lx, ly, lz):
    index = np.zeros(4, dtype=int)
    lin_idx = linear_index
    index[3] = lin_idx / (lx * ly * lz)
    index[2] = (lin_idx - (lx * ly * lz) * index[3]) / (lx * ly)
    index[1] = (lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2]) / lx
    index[0] = (
        lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2] - lx * index[1]
    )

    return index


def transform_from_list_to_array(nx, ny, nz, xyz_coords):

    num_points = xyz_coords.shape[0]

    try:
        len_array = xyz_coords.shape[1]
    except:
        len_array = 1
        xyz_coords = xyz_coords.reshape(-1, 1)

    array = []
    for i in range(0, len_array):
        array.append(np.zeros((nx, ny, nz)))

    for linear_index in range(0, num_points):
        index = nonlinear_index(linear_index, ny, nx, nz)
        j = index[0]
        i = index[1]
        k = index[2]
        for ind in range(0, len(array)):
            array[ind][i, j, k] = xyz_coords[linear_index, ind]

    return array
