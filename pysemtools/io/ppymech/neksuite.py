""" Module that wraps the parallel IO calls and put data in the pymech format"""

import sys
from mpi4py import MPI
import numpy as np
from .parallel_io import (
    fld_file_read_vector_field,
    fld_file_read_field,
    fld_file_write_vector_field,
    fld_file_write_field,
    fld_file_write_vector_metadata,
    fld_file_write_metadata,
)
from ...monitoring.logger import Logger

# from memory_profiler import profile
class NekHeader:
    """
    Class containing the header information for a nek file
    This is HEAVILY inspired by pymech subroutines: https://github.com/eX-Mech/pymech
    The only reason it is reproduced here is to avoid importing too many modules that take a lot of memory, specially with mpi4py
    """

    def __init__(self,
                 wdsz = None,
                 orders = None,
                 nb_elems = None,
                 nb_elems_file = None,
                 time = None,
                 istep = None,
                 fid = None,
                 nb_files = None,
                 nb_vars = None,
                 header_string = None):
        
        if header_string is not None:
            self.init_from_string(header_string)
        else:
            self.init_from_attributes(
                wdsz=wdsz,
                orders=orders,
                nb_elems=nb_elems,
                nb_elems_file=nb_elems_file,
                time=time,
                istep=istep,
                fid=fid,
                nb_files=nb_files,
                nb_vars=nb_vars,
            )

    def init_from_attributes(self,
                            wdsz: int,
                            orders: tuple[int, ...],
                            nb_elems: int,
                            nb_elems_file: int,
                            time: float,
                            istep: int,
                            fid: int,
                            nb_files: int, 
                            nb_vars: tuple[int, ...]):
        self.wdsz = wdsz
        self.orders = orders
        self.nb_elems = nb_elems
        self.nb_elems_file = nb_elems_file
        self.time = time
        self.istep = istep
        self.fid = fid
        self.nb_files = nb_files
        self.nb_vars = nb_vars

        self.variables = ''
        if self.nb_vars[0] > 0:
            self.variables += f"X"
        if self.nb_vars[1] > 0:
            self.variables += f"U"
        if self.nb_vars[2] > 0:
            self.variables += f"P"
        if self.nb_vars[3] > 0:
            self.variables += f"T"
        if self.nb_vars[4] > 0:
            self.variables += f"S{self.nb_vars[4]:02d}"

        if self.wdsz == 4:
            self.realtype = "f"
        elif self.wdsz == 8:
            self.realtype = "d"

        if self.orders[2] > 1:
            self.nb_dims = 3
        else:
            self.nb_dims = 2

    def init_from_string(self, header_string):
        """
        """

        self.wdsz = int(header_string[1])
        self.orders = tuple(int(header_string[i]) for i in range(2, 5))
        self.nb_elems = int(header_string[5])
        self.nb_elems_file = int(header_string[6])
        self.time = float(header_string[7])
        self.istep = int(header_string[8])
        self.fid = int(header_string[9])
        self.nb_files = int(header_string[10])
        self.variables = header_string[11].decode("utf-8") 
        
        if self.wdsz == 4:
            self.realtype = "f"
        elif self.wdsz == 8:
            self.realtype = "d"

        if self.orders[2] > 1:
            self.nb_dims = 3
        else:
            self.nb_dims = 2

        if "S" in self.variables:
            num_scal = int(self.variables[self.variables.index("S") + 1 :])

        self.nb_vars = (
            self.nb_dims if "X" in self.variables else 0,
            self.nb_dims if "U" in self.variables else 0,
            1 if "P" in self.variables else 0,
            1 if "T" in self.variables else 0,
            num_scal if "S" in self.variables else 0,
        )

    def as_bytestring(self) -> bytes:
        header = "#std %1i %2i %2i %2i %10i %10i %20.13E %9i %6i %6i %s" % (
            self.wdsz,
            self.orders[0],
            self.orders[1],
            self.orders[2],
            self.nb_elems,
            self.nb_elems_file,
            self.time,
            self.istep,
            self.fid,
            self.nb_files,
            self.variables,
        )
        return header.ljust(132).encode("utf-8")
    
def read_nekheader(path, return_string = False):
    """
    Read the header of a nek type file.
    This is HEAVILY inspired by pymech subroutines: https://github.com/eX-Mech/pymech
    The only reason it is reproduced here is to avoid importing too many modules that take a lot of memory, specially with mpi4py
    """
    if isinstance(path, str):
        with open(path, "rb") as fp:
            header = fp.read(132).split()
    else:
        raise ValueError("Should be a path")

    if len(header) < 12:
        raise IOError("Header of the file was too short.")

    # Relying on attrs converter to type-cast. Mypy will complain
    if return_string:
        return header
    else:
        return NekHeader(header_string=header)

class IoHelper:
    """
    Class to contain general information of the file and some buffers

    This is used primarly to pass data around in writing routines/reading routines.

    :meta private:
    """

    def __init__(self, h, pynek_dtype=np.double):

        self.fld_data_size = np.int64(h.wdsz)
        self.pynek_dtype = pynek_dtype
        self.lx = np.int64(h.orders[0])
        self.ly = np.int64(h.orders[1])
        self.lz = np.int64(h.orders[2])
        self.lxyz = np.int64(self.lx * self.ly * self.lz)
        self.glb_nelv = np.int64(h.nb_elems)
        self.time = h.time
        self.istep = np.int64(h.istep)
        self.variables = h.variables
        self.realtype = h.realtype
        self.gdim = np.int64(h.nb_dims)
        self.pos_variables = np.int64(h.nb_vars[0])
        self.vel_variables = np.int64(h.nb_vars[1])
        self.pres_variables = np.int64(h.nb_vars[2])
        self.temp_variables = np.int64(h.nb_vars[3])
        self.scalar_variables = np.int64(h.nb_vars[4])

        # Allocate optional variables
        self.nelv = np.int64(0)
        self.n = np.int64(0)
        self.offset_el = np.int64(0)

        self.m = np.int64(0)
        self.pe_rank = np.int64(0)
        self.pe_size = np.int64(0)
        self.l = np.int64(0)
        self.r = np.int64(0)
        self.ip = np.int64(0)

        self.tmp_sp_vector = None
        self.tmp_dp_vector = None
        self.tmp_sp_field = None
        self.tmp_dp_field = None

    def element_mapping(self, comm):
        """
        Maps the number of elements each processor has equally.

        Not used anymore.

        Parameters
        ----------
        comm : Comm
            MPI communicator

        Returns
        -------

        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Divide the global number of elements equally
        self.nelv = np.int64(self.glb_nelv / size)
        self.n = self.lxyz * self.nelv
        self.offset_el = rank * self.nelv

    def element_mapping_load_balanced_linear(self, comm):
        """Maps the number of elements each processor has
        in a linearly load balanced manner

        Parameters
        ----------
        comm :


        Returns
        -------

        """
        self.m = self.glb_nelv
        self.pe_rank = np.int64(comm.Get_rank())
        self.pe_size = np.int64(comm.Get_size())
        self.l = np.floor(np.double(self.m) / np.double(self.pe_size))
        self.r = np.mod(self.m, self.pe_size)
        self.ip = np.floor(
            (
                np.double(self.m)
                + np.double(self.pe_size)
                - np.double(self.pe_rank)
                - np.double(1)
            )
            / np.double(self.pe_size)
        )

        self.nelv = np.int64(self.ip)
        self.offset_el = np.int64(self.pe_rank * self.l + min(self.pe_rank, self.r))
        self.n = self.lxyz * self.nelv

    def element_mapping_from_parallel_hexadata(self, comm):
        """Find the element mapping when the input data was already parallel and divided equally

        Parameters
        ----------
        comm :


        Returns
        -------

        """
        rank = comm.Get_rank()
        size = comm.Get_size()

        # io helper assume that the nel in header is the global one
        # So we have to correct if the header is initialized from a parallel hexadata object
        self.nelv = self.glb_nelv
        self.n = self.lxyz * self.nelv

        # Later do a running sum
        self.offset_el = rank * self.nelv

        # Later on, update this to an mpi reduction,
        # now we assume that all elements are divided equally
        self.glb_nelv = self.nelv * size

    def element_mapping_from_parallel_hexadata_mpi(self, comm):
        """Find the element mapping when the input data was already parallel and divided
        in a linearly load balanced manner

        Parameters
        ----------
        comm :


        Returns
        -------

        """

        # io helper assume that the nel in header is the global one
        # So we have to correct if the header is initialized from a parallel hexadata object
        self.nelv = self.glb_nelv
        self.n = self.lxyz * self.nelv

        # do a running sum
        sendbuf = np.ones((1), np.int64) * self.nelv
        recvbuf = np.zeros((1), np.int64)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - self.nelv

        # Later on, update this to an mpi reduction,
        sendbuf = np.ones((1), np.int64) * self.nelv
        recvbuf = np.zeros((1), np.int64)
        comm.Allreduce(sendbuf, recvbuf)
        self.glb_nelv = recvbuf[0]

    def allocate_temporal_arrays(self):
        """'Allocate temporal arrays for reading and writing fields"""
        if self.fld_data_size == 4:
            self.tmp_sp_vector = np.zeros(self.gdim * self.n, dtype=np.single)
            self.tmp_sp_field = np.zeros(self.n, dtype=np.single)
        elif self.fld_data_size == 8:
            self.tmp_dp_vector = np.zeros(self.gdim * self.n, dtype=np.double)
            self.tmp_dp_field = np.zeros(self.n, dtype=np.double)


# @profile
def preadnek(filename, comm, data_dtype=np.double):
    """
    Read and fld file and return a pymech hexadata object (Parallel).

    Main function for readinf nek type fld filed.

    Parameters
    ----------
    filename : str
        The filename of the fld file.

    comm : Comm
        MPI communicator.

    data_dtype : str
        The data type of the data in the file. (Default value = "float64").

    Returns
    -------
    HexaData
        The data read from the file in a pymech hexadata object.

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from pysemtools.io.ppymech.neksuite import preadnek
    >>> comm = MPI.COMM_WORLD
    >>> data = preadnek('field00001.fld', comm)
    """
    from pymech.core import HexaData
    from pymech.neksuite.field import read_header

    log = Logger(comm=comm, module_name="preadnek")
    log.tic()
    log.write("info", "Reading file: {}".format(filename))

    mpi_int_size = MPI.INT.Get_size()
    mpi_real_size = MPI.REAL.Get_size()
    # mpi_double_size = MPI.DOUBLE.Get_size()
    mpi_character_size = MPI.CHARACTER.Get_size()

    # Read the header
    header = read_header(filename)

    # Initialize the io helper
    ioh = IoHelper(header, pynek_dtype=data_dtype)

    # Find the appropiate partitioning of the file
    # ioh.element_mapping(comm)
    ioh.element_mapping_load_balanced_linear(comm)

    # allocate temporal arrays
    log.write("debug", "Allocating temporal arrays")
    ioh.allocate_temporal_arrays()

    # Create the pymech hexadata object
    log.write("debug", "Creating HexaData object")
    data = HexaData(
        header.nb_dims, ioh.nelv, header.orders, header.nb_vars, 0, dtype=data_dtype
    )
    data.time = header.time
    data.istep = header.istep
    data.wdsz = header.wdsz
    data.endian = sys.byteorder

    # Open the file
    fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)

    # Read the test pattern
    mpi_offset = np.int64(132 * mpi_character_size)
    test_pattern = np.zeros(1, dtype=np.single)
    fh.Read_at_all(mpi_offset, test_pattern, status=None)

    # Read the indices?
    mpi_offset += mpi_real_size
    idx = np.zeros(ioh.nelv, dtype=np.int32)
    byte_offset = mpi_offset + ioh.offset_el * mpi_int_size
    fh.Read_at_all(byte_offset, idx, status=None)
    data.elmap = idx
    mpi_offset += ioh.glb_nelv * mpi_int_size

    # Read the coordinates
    if ioh.pos_variables > 0:
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        )
        log.write("debug", "Reading coordinate data")
        x = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
        y = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
        z = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
        fld_file_read_vector_field(fh, byte_offset, ioh, x=x, y=y, z=z)
        for e in range(0, ioh.nelv):
            data.elem[e].pos[0, :, :, :] = x[e, :, :, :].copy()
            data.elem[e].pos[1, :, :, :] = y[e, :, :, :].copy()
            data.elem[e].pos[2, :, :, :] = z[e, :, :, :].copy()
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Read the velocity
    if ioh.vel_variables > 0:
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        )

        log.write("debug", "Reading velocity data")
        if "x" not in locals():
            x = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
        if "y" not in locals():
            y = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
        if "z" not in locals():
            z = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)

        u = x
        v = y
        w = z
        fld_file_read_vector_field(fh, byte_offset, ioh, x=u, y=v, z=w)
        for e in range(0, ioh.nelv):
            data.elem[e].vel[0, :, :, :] = u[e, :, :, :].copy()
            data.elem[e].vel[1, :, :, :] = v[e, :, :, :].copy()
            data.elem[e].vel[2, :, :, :] = w[e, :, :, :].copy()
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Read pressure
    if ioh.pres_variables > 0:
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size

        log.write("debug", "Reading pressure data")
        if "x" not in locals():
            x = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)

        p = x
        fld_file_read_field(fh, byte_offset, ioh, x=p)
        for e in range(0, ioh.nelv):
            data.elem[e].pres[0, :, :, :] = p[e, :, :, :].copy()
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Read temperature
    if ioh.temp_variables > 0:
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size
        log.write("debug", "Reading temperature data")
        if "x" not in locals():
            x = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
        t = x
        fld_file_read_field(fh, byte_offset, ioh, x=t)
        for e in range(0, ioh.nelv):
            data.elem[e].temp[0, :, :, :] = t[e, :, :, :].copy()
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Read scalars
    ii = 0
    for var in range(0, ioh.scalar_variables):
        if ii == 0:  # Only print once
            log.write("debug", "Reading scalar data")
            ii += 1

        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size
        if "x" not in locals():
            x = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
        s = x
        fld_file_read_field(fh, byte_offset, ioh, x=s)
        for e in range(0, ioh.nelv):
            data.elem[e].scal[var, :, :, :] = s[e, :, :, :].copy()
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    fh.Close()

    log.write("debug", "File read")
    log.toc()

    del log

    return data


# @profile
def pynekread(filename, comm, data_dtype=np.double, msh=None, fld=None, overwrite_fld = False):
    """
    Read nek file and returs a pynekobject (Parallel).

    Main function for readinf nek type fld filed.

    Parameters
    ----------
    filename : str
        The filename of the fld file.

    comm : Comm
        MPI communicator.

    data_dtype : str
        The data type of the data in the file. (Default value = "float64").

    msh : Mesh
        The mesh object to put the data in. (Default value = None).

    fld : Field
        The field object to put the data in. (Default value = None).

    overwrite_fld : bool
        Wether or not to overwrite the contents of fld. (Default value = False).

    Returns
    -------
    None
        Nothing is returned, the attributes are set in the object.

    Examples
    --------
    >>> from mpi4py import MPI
    >>> from pysemtools.io.ppymech.neksuite import pynekread
    >>> comm = MPI.COMM_WORLD
    >>> msh = msh_c(comm)
    >>> fld = field_c(comm)
    >>> pynekread(fname, comm, msh = msh, fld=fld)
    """

    log = Logger(comm=comm, module_name="pynekread")
    log.tic()
    log.write("info", "Reading file: {}".format(filename))

    mpi_int_size = MPI.INT.Get_size()
    mpi_real_size = MPI.REAL.Get_size()
    # mpi_double_size = MPI.DOUBLE.Get_size()
    mpi_character_size = MPI.CHARACTER.Get_size()

    # Read the header
    header = read_nekheader(filename)

    # Initialize the io helper
    ioh = IoHelper(header, pynek_dtype=data_dtype)

    # Find the appropiate partitioning of the file
    # ioh.element_mapping(comm)
    ioh.element_mapping_load_balanced_linear(comm)

    # allocate temporal arrays
    log.write("debug", "Allocating temporal arrays")
    ioh.allocate_temporal_arrays()

    ## Create the pymech hexadata object
    # data = HexaData(
    #    header.nb_dims, ioh.nelv, header.orders, header.nb_vars, 0, dtype=data_dtype
    # )
    # data.time = header.time
    # data.istep = header.istep
    # data.wdsz = header.wdsz
    # data.endian = sys.byteorder

    # Open the file
    fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)

    # Read the test pattern
    mpi_offset = np.int64(132 * mpi_character_size)
    test_pattern = np.zeros(1, dtype=np.single)
    fh.Read_at_all(mpi_offset, test_pattern, status=None)

    # Read the indices?
    mpi_offset += mpi_real_size
    idx = np.zeros(ioh.nelv, dtype=np.int32)
    byte_offset = mpi_offset + ioh.offset_el * mpi_int_size
    fh.Read_at_all(byte_offset, idx, status=None)
    mpi_offset += ioh.glb_nelv * mpi_int_size

    # Read the coordinates
    if ioh.pos_variables > 0:

        if not isinstance(msh, type(None)):
            byte_offset = (
                mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
            )
            log.write("debug", "Reading coordinate data")
            x = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            y = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            z = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            fld_file_read_vector_field(fh, byte_offset, ioh, x=x, y=y, z=z)

            msh.init_from_coords(comm, x, y, z, elmap=idx)

            mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    if not isinstance(fld, type(None)):
        log.write("info", "Reading field data")
        if overwrite_fld:
            log.write("info", "Overwriting fld object")
            fld.clear()

    # Read the velocity
    if ioh.vel_variables > 0:
        if not isinstance(fld, type(None)):
            byte_offset = (
                mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
            )

            log.write("debug", "Reading velocity data")
            u = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            v = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            w = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)

            fld_file_read_vector_field(fh, byte_offset, ioh, x=u, y=v, z=w)
            if ioh.gdim == 3:
                fld.fields["vel"].extend([u, v, w])
            elif ioh.gdim == 2:
                fld.fields["vel"].extend([u, v])

            mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Read pressure
    if ioh.pres_variables > 0:
        if not isinstance(fld, type(None)):
            byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size

            log.write("debug", "Reading pressure data")
            p = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)

            fld_file_read_field(fh, byte_offset, ioh, x=p)
            fld.fields["pres"].append(p)

            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Read temperature
    if ioh.temp_variables > 0:
        if not isinstance(fld, type(None)):
            byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size

            log.write("debug", "Reading temperature data")
            t = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)

            fld_file_read_field(fh, byte_offset, ioh, x=t)
            fld.fields["temp"].append(t)

            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Read scalars
    ii = 0
    for var in range(0, ioh.scalar_variables):
        if not isinstance(fld, type(None)):
            log.write("debug", f"Reading scalar {var} data")

            byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size

            s = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            fld_file_read_field(fh, byte_offset, ioh, x=s)
            fld.fields["scal"].append(s.copy())

            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    if not isinstance(fld, type(None)):
        fld.t = header.time
        fld.update_vars()

    fh.Close()

    log.write("info", "File read")
    log.toc()

    del log
    return

def get_file_time(filename):
    '''
    Get the time from the file header. This is useful for calls to the individual field reader.
    '''

    header = read_nekheader(filename)

    return header.time

def pynekread_field(filename, comm, data_dtype=np.double, key=""):
    """
    Read nek file and returs a pynekobject (Parallel).

    Main function for readinf nek type fld filed.

    Parameters
    ----------
    filename : str
        The filename of the fld file.

    comm : Comm
        MPI communicator.

    data_dtype : str
        The data type of the data in the file. (Default value = "float64").

    key : str
        The key of the field to read.
        Typically "vel", "pres", "temp" or "scal_1", "scal_2", etc.

    Returns
    -------
    list
        The data read from the file in a list.
    """

    log = Logger(comm=comm, module_name="pynekread_field")
    log.tic()

    key_prefix = key.split("_")[0]
    try:
        key_suffix = int(key.split("_")[1])
    except IndexError:
        key_suffix = 0

    log.write("info", f"Reading field: {key} from file: {filename}")

    mpi_int_size = MPI.INT.Get_size()
    mpi_real_size = MPI.REAL.Get_size()
    # mpi_double_size = MPI.DOUBLE.Get_size()
    mpi_character_size = MPI.CHARACTER.Get_size()

    # Read the header
    header = read_nekheader(filename)

    # Initialize the io helper
    ioh = IoHelper(header, pynek_dtype=data_dtype)

    # Find the appropiate partitioning of the file
    # ioh.element_mapping(comm)
    ioh.element_mapping_load_balanced_linear(comm)

    # allocate temporal arrays
    log.write("debug", "Allocating temporal arrays")
    ioh.allocate_temporal_arrays()

    ## Create the pymech hexadata object
    # data = HexaData(
    #    header.nb_dims, ioh.nelv, header.orders, header.nb_vars, 0, dtype=data_dtype
    # )
    # data.time = header.time
    # data.istep = header.istep
    # data.wdsz = header.wdsz
    # data.endian = sys.byteorder

    # Open the file
    fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)

    # Read the test pattern
    mpi_offset = np.int64(132 * mpi_character_size)
    test_pattern = np.zeros(1, dtype=np.single)
    fh.Read_at_all(mpi_offset, test_pattern, status=None)

    # Read the indices?
    mpi_offset += mpi_real_size
    idx = np.zeros(ioh.nelv, dtype=np.int32)
    byte_offset = mpi_offset + ioh.offset_el * mpi_int_size
    fh.Read_at_all(byte_offset, idx, status=None)
    mpi_offset += ioh.glb_nelv * mpi_int_size

    # Read the coordinates
    if ioh.pos_variables > 0:

        if key_prefix == "pos":
            byte_offset = (
                mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
            )
            log.write("debug", "Reading coordinate data")
            x = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            y = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            z = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            fld_file_read_vector_field(fh, byte_offset, ioh, x=x, y=y, z=z)

            mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Read the velocity
    if ioh.vel_variables > 0:
        if key_prefix == "vel":
            byte_offset = (
                mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
            )

            log.write("debug", "Reading velocity data")
            u = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            v = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            w = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            fld_file_read_vector_field(fh, byte_offset, ioh, x=u, y=v, z=w)

            mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Read pressure
    if ioh.pres_variables > 0:
        if key_prefix == "pres":
            byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size

            log.write("debug", "Reading pressure data")
            p = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)

            fld_file_read_field(fh, byte_offset, ioh, x=p)

            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Read temperature
    if ioh.temp_variables > 0:
        if key_prefix == "temp":
            byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size

            log.write("debug", "Reading temperature data")
            t = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)

            fld_file_read_field(fh, byte_offset, ioh, x=t)

            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Read scalars
    if ioh.scalar_variables > 0:
        if key_prefix == "scal":
            var = int(key_suffix)
            log.write("debug", f"Reading scalar {var} data")

            if var >= ioh.scalar_variables:
                raise ValueError(f"Scalar {var} does not exist in the file.")

            mpi_offset += (ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size) * var

            byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size

            s = np.zeros(ioh.nelv * ioh.lxyz, dtype=ioh.pynek_dtype)
            fld_file_read_field(fh, byte_offset, ioh, x=s)

            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size
        else:
            mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    fh.Close()

    log.write("info", "File read")
    log.toc()

    del log

    if key_prefix == "pos":
        return [x, y, z]
    elif key_prefix == "vel" and ioh.gdim == 3:
        return [u, v, w]
    elif key_prefix == "vel" and ioh.gdim == 2:
        return [u, v]
    elif key_prefix == "pres":
        return [p]
    elif key_prefix == "temp":
        return [t]
    elif key_prefix == "scal":
        return [s]
    else:
        raise ValueError(f"Key {key} not recognized.")


# @profile
def pwritenek(filename, data, comm):
    """
    Write and fld file and from a pymech hexadata object (Parallel).

    Main function to write fld files.

    Parameters
    ----------
    filename : str
        The filename of the fld file.

    data : HexaData
        The data to write to the file.

    comm : Comm
        MPI communicator.

    Examples
    --------
    Assuming you have a hexadata object already:

    >>> from pysemtools.io.ppymech.neksuite import pwritenek
    >>> pwritenek('field00001.fld', data, comm)
    """
    from pymech.neksuite.field import Header

    mpi_int_size = MPI.INT.Get_size()
    mpi_real_size = MPI.REAL.Get_size()
    # mpi_double_size = MPI.DOUBLE.Get_size()
    mpi_character_size = MPI.CHARACTER.Get_size()

    # instance a dummy header
    dh = Header(
        data.wdsz,
        data.lr1,
        data.nel,
        data.nel,
        data.time,
        data.istep,
        fid=0,
        nb_files=1,
        nb_vars=data.var,
    )

    # instance the parallel io helper with the dummy header
    ioh = IoHelper(dh)

    # Get actual element mapping from the parallel hexadata
    # We need this since what we have in data.nel is the
    # local number of elements, not the global one
    # ioh.element_mapping_from_parallel_hexadata(comm)
    ioh.element_mapping_from_parallel_hexadata_mpi(comm)

    # allocate temporal arrays
    ioh.allocate_temporal_arrays()

    # instance actual header
    h = Header(
        data.wdsz,
        data.lr1,
        ioh.glb_nelv,
        ioh.glb_nelv,
        data.time,
        data.istep,
        fid=0,
        nb_files=1,
        nb_vars=data.var,
    )

    # Open the file
    amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
    fh = MPI.File.Open(comm, filename, amode)

    # Write the header
    mpi_offset = np.int64(0)
    fh.Write_all(h.as_bytestring())
    mpi_offset += 132 * mpi_character_size

    # write test pattern
    test_pattern = np.zeros(1, dtype=np.single)
    test_pattern[0] = 6.54321
    fh.Write_all(test_pattern)
    mpi_offset += mpi_real_size

    # write element mapping
    idx = np.zeros(ioh.nelv, dtype=np.int32)
    for i in range(0, data.nel):
        idx[i] = data.elmap[i]
    byte_offset = mpi_offset + ioh.offset_el * mpi_int_size
    fh.Write_at_all(byte_offset, idx, status=None)
    mpi_offset += ioh.glb_nelv * mpi_int_size

    # Write the coordinates
    if ioh.pos_variables > 0:
        ddtype = data.elem[0].pos.dtype
        x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
        y = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
        z = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
        for e in range(0, ioh.nelv):
            x[e, :, :, :] = data.elem[e].pos[0, :, :, :].copy()
            y[e, :, :, :] = data.elem[e].pos[1, :, :, :].copy()
            z[e, :, :, :] = data.elem[e].pos[2, :, :, :].copy()
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        )
        fld_file_write_vector_field(fh, byte_offset, x, y, z, ioh)
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Write the velocity
    if ioh.vel_variables > 0:
        ddtype = data.elem[0].vel.dtype
        if "x" not in locals():
            x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
        if "y" not in locals():
            y = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
        if "z" not in locals():
            z = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

        u = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
        v = y.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
        w = z.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)

        for e in range(0, ioh.nelv):
            u[e, :, :, :] = data.elem[e].vel[0, :, :, :].copy()
            v[e, :, :, :] = data.elem[e].vel[1, :, :, :].copy()
            w[e, :, :, :] = data.elem[e].vel[2, :, :, :].copy()
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        )
        fld_file_write_vector_field(fh, byte_offset, u, v, w, ioh)
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Write pressure
    if ioh.pres_variables > 0:
        ddtype = data.elem[0].pres.dtype
        if "x" not in locals():
            x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

        p = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
        for e in range(0, ioh.nelv):
            p[e, :, :, :] = data.elem[e].pres[0, :, :, :].copy()
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size
        fld_file_write_field(fh, byte_offset, p, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Write Temperature
    if ioh.temp_variables > 0:
        ddtype = data.elem[0].temp.dtype
        if "x" not in locals():
            x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

        t = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
        for e in range(0, ioh.nelv):
            t[e, :, :, :] = data.elem[e].temp[0, :, :, :].copy()
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size
        fld_file_write_field(fh, byte_offset, t, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Write scalars
    for var in range(0, ioh.scalar_variables):
        ddtype = data.elem[0].scal.dtype
        if "x" not in locals():
            x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

        s = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
        for e in range(0, ioh.nelv):
            s[e, :, :, :] = data.elem[e].scal[var, :, :, :].copy()
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size
        fld_file_write_field(fh, byte_offset, s, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # ================== Metadata
    if ioh.gdim > 2:
        meta_fld_data_size = np.int64(4) # Always in single precision

        # Write the coordinates
        if ioh.pos_variables > 0:
            ddtype = data.elem[0].pos.dtype
            if "x" not in locals():
                x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
            if "y" not in locals():
                y = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
            if "z" not in locals():
                z = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

            x = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
            y = y.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
            z = z.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)

            for e in range(0, ioh.nelv):
                x[e, :, :, :] = data.elem[e].pos[0, :, :, :]
                y[e, :, :, :] = data.elem[e].pos[1, :, :, :]
                z[e, :, :, :] = data.elem[e].pos[2, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * ioh.gdim * 2 * meta_fld_data_size
            fld_file_write_vector_metadata(fh, byte_offset, x, y, z, ioh)
            mpi_offset += ioh.glb_nelv * ioh.gdim * 2 * meta_fld_data_size

        # Write the velocity
        if ioh.vel_variables > 0:
            ddtype = data.elem[0].vel.dtype
            if "x" not in locals():
                x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
            if "y" not in locals():
                y = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)
            if "z" not in locals():
                z = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

            u = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
            v = y.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
            w = z.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
            for e in range(0, ioh.nelv):
                u[e, :, :, :] = data.elem[e].vel[0, :, :, :]
                v[e, :, :, :] = data.elem[e].vel[1, :, :, :]
                w[e, :, :, :] = data.elem[e].vel[2, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * ioh.gdim * 2 * meta_fld_data_size
            fld_file_write_vector_metadata(fh, byte_offset, u, v, w, ioh)
            mpi_offset += ioh.glb_nelv * ioh.gdim * 2 * meta_fld_data_size

        # Write pressure
        if ioh.pres_variables > 0:
            ddtype = data.elem[0].pres.dtype
            if "x" not in locals():
                x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

            p = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
            for e in range(0, ioh.nelv):
                p[e, :, :, :] = data.elem[e].pres[0, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * meta_fld_data_size
            fld_file_write_metadata(fh, byte_offset, p, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * meta_fld_data_size

        # Write Temperature
        if ioh.temp_variables > 0:
            ddtype = data.elem[0].temp.dtype
            if "x" not in locals():
                x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

            t = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
            for e in range(0, ioh.nelv):
                t[e, :, :, :] = data.elem[e].temp[0, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * meta_fld_data_size
            fld_file_write_metadata(fh, byte_offset, t, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * meta_fld_data_size

        # Write scalars
        for var in range(0, ioh.scalar_variables):
            ddtype = data.elem[0].scal.dtype
            if "x" not in locals():
                x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=ddtype)

            s = x.reshape(ioh.nelv, ioh.lz, ioh.ly, ioh.lx)
            for e in range(0, ioh.nelv):
                s[e, :, :, :] = data.elem[e].scal[var, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * meta_fld_data_size
            fld_file_write_metadata(fh, byte_offset, s, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * meta_fld_data_size

    fh.Close()

    return


# @profile
def pynekwrite(filename, comm, msh=None, fld=None, wdsz=4, istep=0, write_mesh=True):
    """
    Write and fld file and from pynekdatatypes (Parallel).

    Main function to write fld files.

    Parameters
    ----------
    filename : str
        The filename of the fld file.

    comm : Comm
        MPI communicator.

    msh : Mesh
        The mesh object to write to the file. (Default value = None).

    fld : Field
        The field object to write to the file. (Default value = None).

    wdsz : int
        The word size of the data in the file. (Default value = 4).

    istep : int
        The time step of the data. (Default value = 0).

    write_mesh : bool
        If True, write the mesh data. (Default value = True).

    Examples
    --------
    Assuming a mesh object and field object are already present in the namespace:

    >>> from pysemtools.io.ppymech.neksuite import pwritenek
    >>> pynekwrite('field00001.fld', comm, msh = msh, fld=fld)
    """

    log = Logger(comm=comm, module_name="pynekwrite")
    log.tic()
    log.write("info", "Writing file: {}".format(filename))

    mpi_int_size = MPI.INT.Get_size()
    mpi_real_size = MPI.REAL.Get_size()
    # mpi_double_size = MPI.DOUBLE.Get_size()
    mpi_character_size = MPI.CHARACTER.Get_size()

    # associate inputs
    if write_mesh:
        msh_fields = msh.gdim
    else:
        msh_fields = 0
    vel_fields = fld.vel_fields
    pres_fields = fld.pres_fields
    temp_fields = fld.temp_fields
    scal_fields = fld.scal_fields
    time = fld.t
    lx = msh.lx
    ly = msh.ly
    lz = msh.lz
    nelv = msh.nelv

    # instance a dummy header
    dh = NekHeader(
        wdsz,
        (lx, ly, lz),
        nelv,
        nelv,
        time,
        istep,
        fid=0,
        nb_files=1,
        nb_vars=(msh_fields, vel_fields, pres_fields, temp_fields, scal_fields),
    )

    # instance the parallel io helper with the dummy header
    ioh = IoHelper(dh)

    # Get actual element mapping from the parallel hexadata
    # We need this since what we have in data.nel is the
    # local number of elements, not the global one
    # ioh.element_mapping_from_parallel_hexadata(comm)
    ioh.element_mapping_from_parallel_hexadata_mpi(comm)

    # allocate temporal arrays
    ioh.allocate_temporal_arrays()

    # instance actual header
    h = NekHeader(
        wdsz,
        (lx, ly, lz),
        ioh.glb_nelv,
        ioh.glb_nelv,
        time,
        istep,
        fid=0,
        nb_files=1,
        nb_vars=(msh_fields, vel_fields, pres_fields, temp_fields, scal_fields),
    )

    # Open the file
    amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
    fh = MPI.File.Open(comm, filename, amode)

    # Write the header
    mpi_offset = np.int64(0)
    fh.Write_all(h.as_bytestring())
    mpi_offset += 132 * mpi_character_size

    # write test pattern
    test_pattern = np.zeros(1, dtype=np.single)
    test_pattern[0] = 6.54321
    fh.Write_all(test_pattern)
    mpi_offset += mpi_real_size

    # write element mapping
    idx = np.zeros(ioh.nelv, dtype=np.int32)
    if (msh is not None) and (msh.elmap is not None):
        idx[:] = msh.elmap[:]
    else:
        for i in range(0, ioh.nelv):
            idx[i] = i + ioh.offset_el
    byte_offset = mpi_offset + ioh.offset_el * mpi_int_size
    fh.Write_at_all(byte_offset, idx, status=None)
    mpi_offset += ioh.glb_nelv * mpi_int_size

    # Array shape
    field_shape = (ioh.nelv, ioh.lz, ioh.ly, ioh.lx)

    # Write the coordinates
    if (ioh.pos_variables and write_mesh) > 0:

        log.write("debug", "Writing coordinate data")

        x = msh.x
        y = msh.y
        z = msh.z
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        )
        fld_file_write_vector_field(fh, byte_offset, x, y, z, ioh)
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Write the velocity
    if ioh.vel_variables > 0:

        log.write("debug", "Writing velocity data")

        u = fld.fields["vel"][0]
        v = fld.fields["vel"][1]
        if len(fld.fields["vel"]) > 2:
            w = fld.fields["vel"][2]
        else:
            w = np.zeros_like(u)

        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.fld_data_size
        )
        fld_file_write_vector_field(fh, byte_offset, u, v, w, ioh)
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.fld_data_size

    # Write pressure
    if ioh.pres_variables > 0:

        log.write("debug", "Writing pressure data")

        p = fld.fields["pres"][0]
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size
        fld_file_write_field(fh, byte_offset, p, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Write Temperature
    if ioh.temp_variables > 0:

        log.write("debug", "Writing temperature data")

        t = fld.fields["temp"][0]
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size
        fld_file_write_field(fh, byte_offset, t, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Write scalars
    ii = 0
    for var in range(0, ioh.scalar_variables):
        if ii == 0:  # Only print once
            log.write("debug", "Writing scalar data")
            ii += 1
        s = fld.fields["scal"][var]
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.fld_data_size
        fld_file_write_field(fh, byte_offset, s, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.fld_data_size

    # Reshape data
    msh.x.shape = field_shape
    msh.y.shape = field_shape
    msh.z.shape = field_shape
    for key in fld.fields.keys():
        for i in range(len(fld.fields[key])):
            fld.fields[key][i].shape = field_shape

    # ================== Metadata
    if ioh.gdim > 2:

        log.write("debug", "Writing metadata")
        meta_fld_data_size = np.int64(4) # Always in single precision

        # Write the coordinates
        if (ioh.pos_variables and write_mesh) > 0:

            x = msh.x
            y = msh.y
            z = msh.z

            byte_offset = mpi_offset + ioh.offset_el * ioh.gdim * 2 * meta_fld_data_size
            fld_file_write_vector_metadata(fh, byte_offset, x, y, z, ioh)
            mpi_offset += ioh.glb_nelv * ioh.gdim * 2 * meta_fld_data_size

        # Write the velocity
        if ioh.vel_variables > 0:

            u = fld.fields["vel"][0]
            v = fld.fields["vel"][1]
            w = fld.fields["vel"][2]
            byte_offset = mpi_offset + ioh.offset_el * ioh.gdim * 2 * meta_fld_data_size
            fld_file_write_vector_metadata(fh, byte_offset, u, v, w, ioh)
            mpi_offset += ioh.glb_nelv * ioh.gdim * 2 * meta_fld_data_size

        # Write pressure
        if ioh.pres_variables > 0:

            p = fld.fields["pres"][0]

            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * meta_fld_data_size
            fld_file_write_metadata(fh, byte_offset, p, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * meta_fld_data_size

        # Write Temperature
        if ioh.temp_variables > 0:

            t = fld.fields["temp"][0]
            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * meta_fld_data_size
            fld_file_write_metadata(fh, byte_offset, t, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * meta_fld_data_size

        # Write scalars
        for var in range(0, ioh.scalar_variables):

            s = fld.fields["scal"][var]
            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * meta_fld_data_size
            fld_file_write_metadata(fh, byte_offset, s, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * meta_fld_data_size

    # Reshape data
    msh.x.shape = field_shape
    msh.y.shape = field_shape
    msh.z.shape = field_shape
    for key in fld.fields.keys():
        for i in range(len(fld.fields[key])):
            fld.fields[key][i].shape = field_shape

    fh.Close()

    log.write("debug", "File written")
    log.toc()

    del log

    return
