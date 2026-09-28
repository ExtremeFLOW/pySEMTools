"""Helper functions shared by the point interpolators when searching for points"""

import numpy as np


def test_pattern_field(elem_xyz, probe_xyz):
    """
    Build the test pattern used to accept points that are not strictly inside an element.

    The test field is ``x**2 + y**2 + z**2``. Its nodal values are interpolated at the
    rst coordinates found for the probe and compared with its exact value at the probe.
    The error of that comparison is normalized by ``max(1, x**2 + y**2 + z**2)`` at the
    probe, so it is an absolute error for coordinates of order one or smaller and a
    relative error for large coordinates. The field is always evaluated in double
    precision, regardless of the dtype of the mesh and of the probes: in single
    precision the field is quantized to ~6e-8 of its own magnitude, which is O(1) in
    absolute terms for coordinates of a few thousand length units and made the
    comparison with the tolerance meaningless for single precision meshes.

    Parameters
    ----------
    elem_xyz : tuple of ndarray
        (x, y, z) coordinates of the nodes of the elements. The last three axes are the
        nodes of an element (lz, ly, lx); any leading axes are batch axes.
    probe_xyz : tuple of ndarray or float
        (x, y, z) coordinates of the probes, one per element in the batch.

    Returns
    -------
    test_field : ndarray
        Nodal values of the test field, same shape as the coordinates, dtype double.
    test_probe : ndarray
        Exact value of the test field at each probe, shape of the batch axes.
    test_scale : ndarray
        Normalization of the error, ``max(1, test_probe)``, shape of the batch axes.
    """

    x_e = np.asarray(elem_xyz[0], dtype=np.double)
    y_e = np.asarray(elem_xyz[1], dtype=np.double)
    z_e = np.asarray(elem_xyz[2], dtype=np.double)
    batch_shape = x_e.shape[:-3]

    test_field = x_e**2 + y_e**2 + z_e**2

    px = np.asarray(probe_xyz[0], dtype=np.double).reshape(batch_shape)
    py = np.asarray(probe_xyz[1], dtype=np.double).reshape(batch_shape)
    pz = np.asarray(probe_xyz[2], dtype=np.double).reshape(batch_shape)
    test_probe = px**2 + py**2 + pz**2
    test_scale = np.maximum(1.0, test_probe)

    return test_field, test_probe, test_scale
