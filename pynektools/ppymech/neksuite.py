from mpi4py import MPI
import numpy as np
from pymech.neksuite.field import read_header
from pymech.core import HexaData
from pymech.neksuite.field import Header
from .parallel_io import *


class io_helper:
    def __init__(self, h):

        self.FLD_DATA_SIZE = h.wdsz
        self.lx = h.orders[0]
        self.ly = h.orders[1]
        self.lz = h.orders[2]
        self.lxyz = self.lx * self.ly * self.lz
        self.glb_nelv = h.nb_elems
        self.time = h.time
        self.istep = h.istep
        self.variables = h.variables
        self.realtype = h.realtype
        self.gdim = h.nb_dims
        self.pos_variables = h.nb_vars[0]
        self.vel_variables = h.nb_vars[1]
        self.pres_variables = h.nb_vars[2]
        self.temp_variables = h.nb_vars[3]
        self.scalar_variables = h.nb_vars[4]

    def element_mapping(self, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Divide the global number of elements equally
        self.nelv = int(self.glb_nelv / size)
        self.n = self.lxyz * self.nelv
        self.offset_el = rank * self.nelv

    def element_mapping_load_balanced_linear(self, comm):

        self.M = self.glb_nelv
        self.pe_rank = comm.Get_rank()
        self.pe_size = comm.Get_size()
        self.L = np.floor(np.double(self.M) / np.double(self.pe_size))
        self.R = np.mod(self.M, self.pe_size)
        self.Ip = np.floor(
            (
                np.double(self.M)
                + np.double(self.pe_size)
                - np.double(self.pe_rank)
                - np.double(1)
            )
            / np.double(self.pe_size)
        )

        self.nelv = int(self.Ip)
        self.offset_el = int(self.pe_rank * self.L + min(self.pe_rank, self.R))
        self.n = self.lxyz * self.nelv

        # print("Reading Rank: {}, nelv: {}, offset: {}".format(self.pe_rank, self.nelv, self.offset_el))

    def element_mapping_from_parallel_hexadata(self, comm):
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
        rank = comm.Get_rank()
        size = comm.Get_size()

        # io helper assume that the nel in header is the global one
        # So we have to correct if the header is initialized from a parallel hexadata object
        self.nelv = self.glb_nelv
        self.n = self.lxyz * self.nelv

        # do a running sum
        sendbuf = np.ones((1), np.intc) * self.nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - self.nelv

        # Later on, update this to an mpi reduction,
        sendbuf = np.ones((1), np.intc) * self.nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Allreduce(sendbuf, recvbuf)
        self.glb_nelv = recvbuf[0]

        # print("Writing Rank: {}, nelv: {}, offset: {}, global_nelv: {}".format(rank, self.nelv, self.offset_el, self.glb_nelv))

    def allocate_temporal_arrays(self):
        self.tmp_sp_vector = np.zeros(self.gdim * self.n, dtype=np.single)
        self.tmp_dp_vector = np.zeros(self.gdim * self.n, dtype=np.double)
        self.tmp_sp_field = np.zeros(self.n, dtype=np.single)
        self.tmp_dp_field = np.zeros(self.n, dtype=np.double)


def preadnek(filename, comm, data_dtype="float64"):

    # Get mpi info
    rank = comm.Get_rank()
    size = comm.Get_size()

    MPI_INT_SIZE = MPI.INT.Get_size()
    MPI_REAL_SIZE = MPI.REAL.Get_size()
    MPI_DOUBLE_SIZE = MPI.DOUBLE.Get_size()
    MPI_CHARACTER_SIZE = MPI.CHARACTER.Get_size()

    # Read the header
    header = read_header(filename)

    # Initialize the io helper
    ioh = io_helper(header)

    # Find the appropiate partitioning of the file
    # ioh.element_mapping(comm)
    ioh.element_mapping_load_balanced_linear(comm)

    # allocate temporal arrays
    ioh.allocate_temporal_arrays()

    # Create the pymech hexadata object
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
    mpi_offset = 132 * MPI_CHARACTER_SIZE
    test_pattern = np.zeros(1, dtype=np.single)
    fh.Read_at_all(mpi_offset, test_pattern, status=None)

    # Read the indices?
    mpi_offset += MPI_REAL_SIZE
    idx = np.zeros(ioh.nelv, dtype=np.intc)
    byte_offset = mpi_offset + ioh.offset_el * MPI_INT_SIZE
    fh.Read_at_all(byte_offset, idx, status=None)
    data.elmap = idx
    mpi_offset += ioh.glb_nelv * MPI_INT_SIZE

    # Read the coordinates
    if ioh.pos_variables > 0:
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.FLD_DATA_SIZE
        )
        x, y, z = fld_file_read_vector_field(fh, byte_offset, ioh)
        for e in range(0, ioh.nelv):
            data.elem[e].pos[0, :, :, :] = x[e, :, :, :]
            data.elem[e].pos[1, :, :, :] = y[e, :, :, :]
            data.elem[e].pos[2, :, :, :] = z[e, :, :, :]
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.FLD_DATA_SIZE

    # Read the velocity
    if ioh.vel_variables > 0:
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.FLD_DATA_SIZE
        )
        u, v, w = fld_file_read_vector_field(fh, byte_offset, ioh)
        for e in range(0, ioh.nelv):
            data.elem[e].vel[0, :, :, :] = u[e, :, :, :]
            data.elem[e].vel[1, :, :, :] = v[e, :, :, :]
            data.elem[e].vel[2, :, :, :] = w[e, :, :, :]
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.FLD_DATA_SIZE

    # Read pressure
    if ioh.pres_variables > 0:
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE
        p = fld_file_read_field(fh, byte_offset, ioh)
        for e in range(0, ioh.nelv):
            data.elem[e].pres[0, :, :, :] = p[e, :, :, :]
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE

    # Read temperature
    if ioh.temp_variables > 0:
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE
        t = fld_file_read_field(fh, byte_offset, ioh)
        for e in range(0, ioh.nelv):
            data.elem[e].temp[0, :, :, :] = t[e, :, :, :]
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE

    # Read scalars
    for var in range(0, ioh.scalar_variables):
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE
        s = fld_file_read_field(fh, byte_offset, ioh)
        for e in range(0, ioh.nelv):
            data.elem[e].scal[var, :, :, :] = s[e, :, :, :]
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE

    fh.Close()

    return data


def pwritenek(filename, data, comm):

    # Get mpi info
    rank = comm.Get_rank()
    size = comm.Get_size()

    MPI_INT_SIZE = MPI.INT.Get_size()
    MPI_REAL_SIZE = MPI.REAL.Get_size()
    MPI_DOUBLE_SIZE = MPI.DOUBLE.Get_size()
    MPI_CHARACTER_SIZE = MPI.CHARACTER.Get_size()

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
    ioh = io_helper(dh)

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
    mpi_offset = 0
    fh.Write_all(h.as_bytestring())
    mpi_offset += 132 * MPI_CHARACTER_SIZE

    # write test pattern
    test_pattern = np.zeros(1, dtype=np.single)
    test_pattern[0] = 6.54321
    fh.Write_all(test_pattern)
    mpi_offset += MPI_REAL_SIZE

    # write element mapping
    idx = np.zeros(ioh.nelv, dtype=np.intc)
    for i in range(0, data.nel):
        idx[i] = data.elmap[i]
    byte_offset = mpi_offset + ioh.offset_el * MPI_INT_SIZE
    fh.Write_at_all(byte_offset, idx, status=None)
    mpi_offset += ioh.glb_nelv * MPI_INT_SIZE

    # Write the coordinates
    if ioh.pos_variables > 0:
        x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        y = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        z = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        for e in range(0, ioh.nelv):
            x[e, :, :, :] = data.elem[e].pos[0, :, :, :]
            y[e, :, :, :] = data.elem[e].pos[1, :, :, :]
            z[e, :, :, :] = data.elem[e].pos[2, :, :, :]
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.FLD_DATA_SIZE
        )
        fld_file_write_vector_field(fh, byte_offset, x, y, z, ioh)
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.FLD_DATA_SIZE

    # Write the velocity
    if ioh.vel_variables > 0:
        u = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        v = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        w = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        for e in range(0, ioh.nelv):
            u[e, :, :, :] = data.elem[e].vel[0, :, :, :]
            v[e, :, :, :] = data.elem[e].vel[1, :, :, :]
            w[e, :, :, :] = data.elem[e].vel[2, :, :, :]
        byte_offset = (
            mpi_offset + ioh.offset_el * ioh.gdim * ioh.lxyz * ioh.FLD_DATA_SIZE
        )
        fld_file_write_vector_field(fh, byte_offset, u, v, w, ioh)
        mpi_offset += ioh.glb_nelv * ioh.gdim * ioh.lxyz * ioh.FLD_DATA_SIZE

    # Write pressure
    if ioh.pres_variables > 0:
        p = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        for e in range(0, ioh.nelv):
            p[e, :, :, :] = data.elem[e].pres[0, :, :, :]
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE
        fld_file_write_field(fh, byte_offset, p, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE

    # Write Temperature
    if ioh.temp_variables > 0:
        t = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        for e in range(0, ioh.nelv):
            t[e, :, :, :] = data.elem[e].temp[0, :, :, :]
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE
        fld_file_write_field(fh, byte_offset, t, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE

    # Write scalars
    for var in range(0, ioh.scalar_variables):
        s = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
        for e in range(0, ioh.nelv):
            s[e, :, :, :] = data.elem[e].scal[var, :, :, :]
        byte_offset = mpi_offset + ioh.offset_el * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE
        fld_file_write_field(fh, byte_offset, s, ioh)
        mpi_offset += ioh.glb_nelv * 1 * ioh.lxyz * ioh.FLD_DATA_SIZE

    # ================== Metadata
    if ioh.gdim > 2:

        # Write the coordinates
        if ioh.pos_variables > 0:
            x = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            y = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            z = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            for e in range(0, ioh.nelv):
                x[e, :, :, :] = data.elem[e].pos[0, :, :, :]
                y[e, :, :, :] = data.elem[e].pos[1, :, :, :]
                z[e, :, :, :] = data.elem[e].pos[2, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * ioh.gdim * 2 * ioh.FLD_DATA_SIZE
            fld_file_write_vector_metadata(fh, byte_offset, x, y, z, ioh)
            mpi_offset += ioh.glb_nelv * ioh.gdim * 2 * ioh.FLD_DATA_SIZE

        # Write the velocity
        if ioh.vel_variables > 0:
            u = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            v = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            w = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            for e in range(0, ioh.nelv):
                u[e, :, :, :] = data.elem[e].vel[0, :, :, :]
                v[e, :, :, :] = data.elem[e].vel[1, :, :, :]
                w[e, :, :, :] = data.elem[e].vel[2, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * ioh.gdim * 2 * ioh.FLD_DATA_SIZE
            fld_file_write_vector_metadata(fh, byte_offset, u, v, w, ioh)
            mpi_offset += ioh.glb_nelv * ioh.gdim * 2 * ioh.FLD_DATA_SIZE

        # Write pressure
        if ioh.pres_variables > 0:
            p = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            for e in range(0, ioh.nelv):
                p[e, :, :, :] = data.elem[e].pres[0, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * ioh.FLD_DATA_SIZE
            fld_file_write_metadata(fh, byte_offset, p, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * ioh.FLD_DATA_SIZE

        # Write Temperature
        if ioh.temp_variables > 0:
            t = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            for e in range(0, ioh.nelv):
                t[e, :, :, :] = data.elem[e].temp[0, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * ioh.FLD_DATA_SIZE
            fld_file_write_metadata(fh, byte_offset, t, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * ioh.FLD_DATA_SIZE

        # Write scalars
        for var in range(0, ioh.scalar_variables):
            s = np.zeros((ioh.nelv, ioh.lz, ioh.ly, ioh.lx), dtype=np.double)
            for e in range(0, ioh.nelv):
                s[e, :, :, :] = data.elem[e].scal[var, :, :, :]
            byte_offset = mpi_offset + ioh.offset_el * 1 * 2 * ioh.FLD_DATA_SIZE
            fld_file_write_metadata(fh, byte_offset, s, ioh)
            mpi_offset += ioh.glb_nelv * 1 * 2 * ioh.FLD_DATA_SIZE

    fh.Close()

    return
