import numpy as np


def generate_1d_arrays(bbox, n, mode="equal", gain=1):
    if mode == "equal":
        x_1d = np.linspace(bbox[0], bbox[1], n)
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
