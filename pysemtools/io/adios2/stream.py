"""Module that defines classes to use adios2 to connect python code to 
other codes that have matching pair adios streams."""

import time
import numpy as np

# Adios2 is assumed to be available
try:
    # This import for ver >= adios2.10, otherwise just import adios2
    import adios2.bindings as adios2
except ImportError:
    print("Error: adios2 is not available. This is needed to stream data. Exiting.")
    raise

# Type definition
NoneType = type(None)


def _dbg(comm, msg):
    """Print a rank-tagged debug message."""
    try:
        rank = comm.Get_rank()
        size = comm.Get_size()
        prefix = f"[PY-ADIOS][rank={rank}/{size}]"
    except Exception:
        prefix = "[PY-ADIOS]"
    print(f"{prefix} {msg}", flush=True)


class DataStreamer:
    """
    Class used to communicate data between codes using adios2.

    Use this to send and recieve data.

    The data is always transported as a 1d array, so it is necessary to reshape.

    Parameters
    ----------
    comm : Comm
        MPI communicator.
    from_nek : bool
        Define if the data that is being stream comes from a nek-like code.
        (Default value = True).

    Attributes
    ----------
    glb_nelv : int
        Total number of elements in the global domain.
    lxyz : int
        Number of points per element.
    gdim : int
        Problem dimension.
    nelv : int
        Number of elements that this rank has.

    Examples
    --------
    This type must be paired with another streaming in the other executable/code.
    The codes will not start if the streams are not paired.

    A full of example of recieving data is shown below. It is possible to pair
    with other classes to, for example, write data to disk.

    >>> ds = data_streamer_c(comm)
    >>> x = get_fld_from_ndarray(ds.recieve(), ds.lx, ds.ly, ds.lz, ds.nelv) # Recieve and reshape x
    >>> y = get_fld_from_ndarray(ds.recieve(), ds.lx, ds.ly, ds.lz, ds.nelv) # Recieve and reshape y
    >>> z = get_fld_from_ndarray(ds.recieve(), ds.lx, ds.ly, ds.lz, ds.nelv) # Recieve and reshape z
    >>> ds.finalize()
    >>> msh = msh_c(comm, x = x, y = y, z = z)
    >>> write_fld_file_from_list("field0.f00001", comm, msh, [x, y, z])

    To send data to the other code, use the stream method.

    >>> ds.stream(x.reshape(x.size))
    """

    def __init__(
        self,
        comm,
        from_nek=True,
        timeout_seconds=300,
        sync_comm=None,
    ):

        # Adios status
        self.okstep = adios2.StepStatus.OK
        self.endstream = adios2.StepStatus.EndOfStream
        self.comm = comm

        # ADIOS2 instance
        self.adios = adios2.ADIOS(comm)
        # ADIOS IO
        self.io_stream = self.adios.DeclareIO("streamIO")
        self.io_stream.SetEngine("SST")
        self.io_stream.SetParameters(
            {"OpenTimeoutSecs": str(timeout_seconds)}
        )

        _dbg(
            comm,
            "init start "
            f"timeout={timeout_seconds} from_nek={from_nek}",
        )
        _dbg(comm, "declared shared streamIO")

        if sync_comm is None:
            sync_comm = comm

        _dbg(comm, "barrier before reader open")
        sync_comm.Barrier()
        _dbg(comm, "barrier before reader open complete")
        time.sleep(2)
        _dbg(comm, "opening reader globalArray_f2py")
        self.reader_st = self.io_stream.Open(
            "globalArray_f2py", adios2.Mode.Read, comm
        )
        _dbg(comm, "reader globalArray_f2py open complete")
        time.sleep(2)
        _dbg(comm, "barrier before writer open")
        sync_comm.Barrier()
        _dbg(comm, "barrier before writer open complete")
        time.sleep(2)
        _dbg(comm, "opening writer globalArray_py2f")
        self.writer_st = self.io_stream.Open(
            "globalArray_py2f", adios2.Mode.Write, comm
        )
        _dbg(comm, "writer globalArray_py2f open complete")
        time.sleep(2)
        _dbg(comm, "barrier after stream opens")
        sync_comm.Barrier()
        _dbg(comm, "barrier after stream opens complete")
        time.sleep(2)

        # Access header stream to calculate my element counts
        _dbg(comm, "barrier before header read")
        sync_comm.Barrier()
        _dbg(comm, "barrier before header read complete")
        time.sleep(2)
        _dbg(comm, "reader BeginStep for header")
        self.step_status = self.reader_st.BeginStep()
        _dbg(comm, f"reader header BeginStep status={self.step_status}")

        hdr_elems = self.io_stream.InquireVariable("global_elements")
        hdr_lxyz = self.io_stream.InquireVariable("points_per_element")
        hdr_gdim = self.io_stream.InquireVariable("problem_dimension")
        _dbg(
            comm,
            "header variables "
            f"global_elements={hdr_elems is not None} "
            f"points_per_element={hdr_lxyz is not None} "
            f"problem_dimension={hdr_gdim is not None}",
        )

        elems = np.zeros((1), dtype=np.intc)
        lxyz = np.zeros((1), dtype=np.intc)
        gdim = np.zeros((1), dtype=np.intc)

        self.reader_st.Get(hdr_elems, elems)
        self.reader_st.Get(hdr_lxyz, lxyz)
        self.reader_st.Get(hdr_gdim, gdim)

        self.reader_st.EndStep()  # Data is read here
        _dbg(
            comm,
            "header read complete "
            f"glb_nelv={int(elems.item())} "
            f"lxyz={int(lxyz.item())} "
            f"gdim={int(gdim.item())}",
        )
        time.sleep(2)
        _dbg(comm, "barrier after header read")
        sync_comm.Barrier()
        _dbg(comm, "barrier after header read complete")

        # Assign values
        self.glb_nelv = int(elems.item())
        self.lxyz = int(lxyz.item())
        self.gdim = int(gdim.item())

        # Determine how many elements each reader rank should have
        ## Preallocate the variable names that are populated in function
        self.nelv = None
        self.offset_el = None
        self.n = None
        element_mapping_load_balanced_linear(self, comm)
        _dbg(
            comm,
            "element mapping "
            f"nelv={self.nelv} offset_el={self.offset_el} n={self.n}",
        )

        # Determine the orders if the stream comes from nek
        if from_nek:
            if self.gdim == 3:
                self.lx = int(np.cbrt(self.lxyz))
            else:
                self.lx = int(np.sqrt(self.lxyz))
            self.ly = self.lx
            if self.gdim == 3:
                self.lz = self.lx
            else:
                self.lz = 1

        # Declare writing variable
        tmp = np.zeros((1), dtype=np.double)
        self.py2f_field_totalcount = int(self.glb_nelv * self.lxyz)
        self.py2f_field_my_start = int(self.offset_el * self.lxyz)
        self.py2f_field_my_count = int(self.nelv * self.lxyz)
        self.py2f_field = self.io_stream.DefineVariable(
            "py2f_field",
            tmp,
            [self.py2f_field_totalcount],
            [self.py2f_field_my_start],
            [self.py2f_field_my_count],
        )
        _dbg(
            comm,
            "defined py2f_field "
            f"total={self.py2f_field_totalcount} "
            f"start={self.py2f_field_my_start} "
            f"count={self.py2f_field_my_count}",
        )

    def finalize(self):
        """
        Finalize the execution of the module.

        Used to close reader and writer. The stream will end and the code will not be coupled.
        """
        _dbg(self.comm, "closing reader and writer")
        self.reader_st.Close()
        self.writer_st.Close()
        _dbg(self.comm, "reader and writer closed")

    def stream(self, fld):
        """
        Send data to another code using adios2.

        Couple 2 code or executables.

        Parameters
        ----------
        fld : ndarray
            Field to be sent. Must be a 1d array.
        """
        # Begin a step
        _dbg(
            self.comm,
            "writer BeginStep for py2f_field "
            f"count={self.py2f_field_my_count}",
        )
        step_status = self.writer_st.BeginStep()
        _dbg(self.comm, f"writer BeginStep status={step_status}")
        self.writer_st.Put(
            self.py2f_field,
            fld,
        )
        self.writer_st.EndStep()  # Data is sent here
        _dbg(self.comm, "writer EndStep complete for py2f_field")

    def recieve(self, fld=None, variable="f2py_field"):
        """
        Recieve data from another code using adios2.

        The data is always transported as a 1d array, so it is necessary to reshape to deinred shape.

        Parameters
        ----------
        fld : ndarray
            Buffer to contain the data. If None, it will be created.
            (Default value = None).
        variable : str
            Name of the adios2 variable to be read. Neko used default name. Change as needed.
            This name can remain the same during the execution even if different quantities are being transported.
            (Default value = "f2py_field").

        Returns
        -------
        ndarray
            Returns the field that was recieved.
        """

        if isinstance(fld, NoneType):
            fld = np.zeros((self.py2f_field_my_count), dtype=np.double)

        # Begin a step
        _dbg(self.comm, f"reader BeginStep for variable={variable}")
        self.step_status = self.reader_st.BeginStep()
        _dbg(self.comm, f"reader BeginStep status={self.step_status}")

        if self.step_status == adios2.StepStatus.OK:
            # Set up the offsets
            f2py_field = self.io_stream.InquireVariable(variable)
            _dbg(
                self.comm,
                f"InquireVariable({variable}) found={f2py_field is not None}",
            )
            f2py_field.SetSelection(
                [[self.py2f_field_my_start], [self.py2f_field_my_count]]
            )
            self.reader_st.Get(f2py_field, fld)

            self.reader_st.EndStep()  # Data is read here
            _dbg(
                self.comm,
                "reader EndStep complete "
                f"start={self.py2f_field_my_start} "
                f"count={self.py2f_field_my_count}",
            )

        return fld


def element_mapping_load_balanced_linear(self, comm):
    """
    Assing the number of elements that each ranks has.

    The distribution is done in a lonear load balanced manner.

    Parameters
    ----------
    comm : Comm
        MPI communicator.

    :meta private:
    """

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
