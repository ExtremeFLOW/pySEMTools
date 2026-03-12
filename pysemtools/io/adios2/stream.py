"""Module that defines classes to use adios2 to connect python code to 
other codes that have matching pair adios streams."""

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

    def __init__(self, comm, id, name="globalArray", timeout_seconds = 300):

        # Adios status
        self.okstep = adios2.StepStatus.OK
        self.endstream = adios2.StepStatus.EndOfStream
        self.id = id

        # ADIOS2 instance
        self.adios = adios2.ADIOS(comm)
        # ADIOS IO - Engine
        self.io_asynchronous = self.adios.DeclareIO(f"{name}_io")
        self.io_asynchronous.SetEngine("SST")
        self.io_asynchronous.SetParameters({"OpenTimeoutSecs": str(timeout_seconds)})

        # Open the streams
        print(f"Opening adios2 streams with name {name} and timeout {timeout_seconds} seconds.")
        self.reader_st = self.io_asynchronous.Open(
            f"{name}_f2py", adios2.Mode.Read, comm
        )
        self.writer_st = self.io_asynchronous.Open(
            f"{name}_py2f", adios2.Mode.Write, comm
        )

    def finalize(self):
        """
        Finalize the execution of the module.

        Used to close reader and writer. The stream will end and the code will not be coupled.
        """
        self.reader_st.Close()
        self.writer_st.Close()

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
        step_status = self.writer_st.BeginStep()
        self.writer_st.Put(
            self.py2f_field,
            fld,
        )
        self.writer_st.EndStep()  # Data is sent here

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
        self.step_status = self.reader_st.BeginStep()

        if self.step_status == adios2.StepStatus.OK:
            # Set up the offsets
            f2py_field = self.io_asynchronous.InquireVariable(variable)
            f2py_field.SetSelection(
                [[self.py2f_field_my_start], [self.py2f_field_my_count]]
            )
            self.reader_st.Get(f2py_field, fld)

            self.reader_st.EndStep()  # Data is read here

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
