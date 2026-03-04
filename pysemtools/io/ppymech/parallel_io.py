""" Module providing parallel IO routines for fld files """

import numpy as np

int32_limit = np.int64(2 ** 31 - 1)

def fld_file_read_vector_field(fh, byte_offset, ioh, x=None, y=None, z=None):
    """Function used to read a vector field from a fld file"""

    # Associate
    fld_data_size = ioh.fld_data_size
    lx = ioh.lx
    ly = ioh.ly
    lz = ioh.lz
    nelv = ioh.nelv
    lxyz = ioh.lxyz
    gdim = ioh.gdim
    tmp_sp_vector = ioh.tmp_sp_vector
    tmp_dp_vector = ioh.tmp_dp_vector

    # Allocate
    return_values = False
    if (
        isinstance(x, type(None))
        or isinstance(y, type(None))
        or isinstance(z, type(None))
    ):
        x = np.zeros(nelv * lxyz, dtype=ioh.pynek_dtype)
        y = np.zeros(nelv * lxyz, dtype=ioh.pynek_dtype)
        z = np.zeros(nelv * lxyz, dtype=ioh.pynek_dtype)
        return_values = True
    else:
        x.shape = nelv * lxyz
        y.shape = nelv * lxyz
        z.shape = nelv * lxyz

    # Read
    # Here we reshape to have the data per element
    # the data is reshaped to be of the form:
    # [x1, y1, z1;
    #  x2, y2, z2;
    #  ...]
    # Where the indices alude the element index
    if fld_data_size == 4:
        # Check if it is too much data
        check_data_count(tmp_sp_vector)
        fh.Read_at_all(byte_offset, tmp_sp_vector, status=None)
        tmp_original_shape = tmp_sp_vector.shape

        tmp_sp_vector.shape = (nelv, lxyz * gdim)

        # Divide the data by the chunks
        x[:] = tmp_sp_vector[:, 0:lxyz].flatten()
        y[:] = tmp_sp_vector[:, lxyz : 2 * lxyz].flatten()
        if gdim > 2:
            z[:] = tmp_sp_vector[:, 2 * lxyz : 3 * lxyz].flatten()

        # Reshape the original shape
        tmp_sp_vector.shape = tmp_original_shape
    else:
        # Check if it is too much data
        check_data_count(tmp_dp_vector)
        fh.Read_at_all(byte_offset, tmp_dp_vector, status=None)

        tmp_original_shape = tmp_dp_vector.shape

        tmp_dp_vector.shape = (nelv, lxyz * gdim)

        # Divide the data by the chunks
        x[:] = tmp_dp_vector[:, 0:lxyz].flatten()
        y[:] = tmp_dp_vector[:, lxyz : 2 * lxyz].flatten()
        if gdim > 2:
            z[:] = tmp_dp_vector[:, 2 * lxyz : 3 * lxyz].flatten()

        # Reshape the original shape
        tmp_dp_vector.shape = tmp_original_shape

    # Reshape to pymech compatible
    x.shape = (nelv, lz, ly, lx)
    y.shape = (nelv, lz, ly, lx)
    z.shape = (nelv, lz, ly, lx)

    if return_values:
        return x, y, z
    else:
        return


def fld_file_read_field(fh, byte_offset, ioh, x=None):
    """Function used to read a scalar field from a fld file"""

    # Associate
    fld_data_size = ioh.fld_data_size
    lx = ioh.lx
    ly = ioh.ly
    lz = ioh.lz
    nelv = ioh.nelv
    lxyz = ioh.lxyz
    tmp_sp_field = ioh.tmp_sp_field
    tmp_dp_field = ioh.tmp_dp_field

    # Allocate
    return_values = False
    if isinstance(x, type(None)):
        x = np.zeros(nelv * lxyz, dtype=ioh.pynek_dtype)
        return_values = True
    else:
        x.shape = nelv * lxyz

    # Read
    if fld_data_size == 4:

        # Check if it is too much data
        check_data_count(tmp_sp_field)

        fh.Read_at_all(byte_offset, tmp_sp_field, status=None)
        x[:] = tmp_sp_field.flatten()

    else:

        # Check if it is too much data
        check_data_count(tmp_dp_field)

        fh.Read_at_all(byte_offset, tmp_dp_field, status=None)
        x[:] = tmp_dp_field.flatten()

    # Reshape to pymech compatible
    x.shape = (nelv, lz, ly, lx)

    if return_values:
        return x
    else:
        return


def fld_file_write_vector_field(fh, byte_offset, x, y, z, ioh):
    """Function used to write a vector field to a fld file"""

    # Associate
    fld_data_size = ioh.fld_data_size
    nelv = ioh.nelv
    lxyz = ioh.lxyz
    gdim = ioh.gdim
    tmp_sp_vector = ioh.tmp_sp_vector
    tmp_dp_vector = ioh.tmp_dp_vector

    # Reshape to be a column
    x.shape = (nelv, lxyz)
    y.shape = (nelv, lxyz)
    z.shape = (nelv, lxyz)

    # Write
    if fld_data_size == 4:

        tmp_original_shape = tmp_sp_vector.shape

        tmp_sp_vector.shape = (nelv, lxyz * gdim)

        tmp_sp_vector[:, 0:lxyz] = x[:, :]
        tmp_sp_vector[:, lxyz : 2 * lxyz] = y[:, :]
        if gdim > 2:
            tmp_sp_vector[:, 2 * lxyz : 3 * lxyz] = z[:, :]

        tmp_sp_vector.shape = tmp_original_shape

        # Check if it is too much data
        check_data_count(tmp_sp_vector)

        fh.Write_at_all(byte_offset, tmp_sp_vector, status=None)

    else:

        tmp_original_shape = tmp_dp_vector.shape

        tmp_dp_vector.shape = (nelv, lxyz * gdim)

        tmp_dp_vector[:, 0:lxyz] = x[:, :]
        tmp_dp_vector[:, lxyz : 2 * lxyz] = y[:, :]
        if gdim > 2:
            tmp_dp_vector[:, 2 * lxyz : 3 * lxyz] = z[:, :]

        tmp_dp_vector.shape = tmp_original_shape

        # Check if it is too much data
        check_data_count(tmp_dp_vector)

        fh.Write_at_all(byte_offset, tmp_dp_vector, status=None)

    return


def fld_file_write_field(fh, byte_offset, x, ioh):
    """Function used to write a scalar field to a fld file"""

    # Associate
    fld_data_size = ioh.fld_data_size
    nelv = ioh.nelv
    lxyz = ioh.lxyz
    tmp_sp_field = ioh.tmp_sp_field
    tmp_dp_field = ioh.tmp_dp_field

    # Reshape to single column
    x.shape = nelv * lxyz

    # Write
    if fld_data_size == 4:

        tmp_sp_field[:] = x.flatten()
        # Check if it is too much data
        check_data_count(tmp_sp_field)
        fh.Write_at_all(byte_offset, tmp_sp_field, status=None)

    else:

        tmp_dp_field[:] = x.flatten()
        # Check if it is too much data
        check_data_count(tmp_dp_field)
        fh.Write_at_all(byte_offset, tmp_dp_field, status=None)

    return


def fld_file_write_vector_metadata(fh, byte_offset, x, y, z, ioh):
    """Function used to write metadata of a vector field to a fld file"""

    # Associate
    nelv = ioh.nelv
    gdim = ioh.gdim

    buff = np.zeros(2 * gdim * nelv, dtype=np.single)
    offset = int(2 * gdim)

    buff[: offset * nelv : offset] = np.min(x[:nelv], axis=(1, 2, 3))
    buff[1 : offset * nelv : offset] = np.max(x[:nelv], axis=(1, 2, 3))
    buff[2 : offset * nelv : offset] = np.min(y[:nelv], axis=(1, 2, 3))
    buff[3 : offset * nelv : offset] = np.max(y[:nelv], axis=(1, 2, 3))
    if gdim > 2:
        buff[4 : offset * nelv : offset] = np.min(z[:nelv], axis=(1, 2, 3))
        buff[5 : offset * nelv : offset] = np.max(z[:nelv], axis=(1, 2, 3))
    
    # Check if it is too much data
    check_data_count(buff)

    fh.Write_at_all(byte_offset, buff, status=None)

    return


def fld_file_write_metadata(fh, byte_offset, x, ioh):
    """Function used to write metadata of a scalar field to a fld file"""

    # Associate
    nelv = ioh.nelv

    buff = np.zeros(2 * nelv, dtype=np.single)

    offset = int(2 * 1)

    buff[: offset * nelv : offset] = np.min(x[:nelv], axis=(1, 2, 3))
    buff[1 : offset * nelv : offset] = np.max(x[:nelv], axis=(1, 2, 3))

    # Check if it is too much data
    check_data_count(buff)

    fh.Write_at_all(byte_offset, buff, status=None)

    return

def check_data_count(data: np.ndarray):

    if (np.int64(data.size) >= int32_limit): 
        raise ValueError("The count to write is too large according to MPI standard (max int32 = 2**31 -1 counts), use chunks or more ranks")
    
    return