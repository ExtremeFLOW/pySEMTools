"""Contains method to easily create pointclouds to interpolate data"""

import numpy as np


def generate_1d_arrays(bbox, n, mode="equal", gain=1, endpoint=True):
    """Generate 1d arrays of points distributed in multiple manners"""
    if mode == "equal":
        x_1d = np.linspace(bbox[0], bbox[1], n, endpoint=endpoint)
    elif mode == "cheb":
        x_1d = np.zeros(n)
        i = 0
        for k in range(1, n + 1):
            # These are the chebyshev of the first kind (-1,1)
            # x_1d[i]=np.cos(((2*k-1)/(2*n))*np.pi)
            # These are the chebyshev of the second kind [-1,1]
            x_1d[i] = 1 / 2 * (bbox[0] + bbox[1]) + 1 / 2 * (
                bbox[1] - bbox[0]
            ) * np.cos(((k - 1) / (n - 1)) * np.pi)
            i = i + 1
        x_1d = np.flip(x_1d)
    elif mode == "half_tanh":
        z_unif = np.linspace(0, 1, int(2 * n))
        z = np.copy(z_unif)
        for i in range(0, int(2 * n)):
            z_m1_p1 = 2 * z[i] - 1  # Make z go [-1,1] here only
            z_strech_1 = np.tanh(gain * z_m1_p1) / np.tanh(gain)
            z[i] = z_strech_1
        x_1d = np.zeros(n)
        j = 0
        for i in range(n, int(2 * n)):
            x_1d[j] = bbox[0] + (bbox[1] - bbox[0]) * z[i]
            j = j + 1
    elif mode == "tanh":
        z_unif = np.linspace(0, 1, int(n))
        z = np.copy(z_unif)
        for i in range(0, int(n)):
            z_m1_p1 = 2 * z[i] - 1  # Make z go [-1,1] here only
            z_strech_1 = np.tanh(gain * z_m1_p1) / np.tanh(gain)
            z[i] = z_strech_1
            z[i] = (z_strech_1 + 1) / 2  # Make z go [0,1]

        x_1d = np.zeros(n)
        j = 0
        for i in range(0, int(n)):
            x_1d[j] = bbox[0] + (bbox[1] - bbox[0]) * z[i]
            j = j + 1
    return x_1d

def generate_1d_diff(z_1d, periodic=False):
    """ Generate differences of a 1d array such that each point has a contribution of the difference of the two points next to it"""

    dz = np.ones_like(z_1d)

    if dz.size != 1:

        if periodic:

            period = z_1d[-1] - z_1d[0]

            dz[1:-1] = (z_1d[2:] - z_1d[:-2]) / 2
            dz[0] = (z_1d[1] - (z_1d[-2] - period)) / 2
            dz[-1] = ((z_1d[1] + period) - z_1d[-2]) / 2

            # This will have a wrong scale since the enpoints are accounting for more information. We must rescale
            dz = dz * (period / np.sum(dz))
        
        else:

            dz[1:-1] = (z_1d[2:] - z_1d[:-2]) / 2
            dz[0] = (z_1d[1] - z_1d[0])/2  
            dz[-1] = (z_1d[-1] - z_1d[-2])/2

    return dz
