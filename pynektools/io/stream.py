# General repositories
import numpy as np

# Adios2 is assumed to be available
import adios2

# Type definition
NoneType = type(None)


class data_streamer_c:

    def __init__(self, comm, from_nek=True):

        # Adios status
        self.okstep = adios2.StepStatus.OK
        self.endstream = adios2.StepStatus.EndOfStream

        # ADIOS2 instance
        self.adios = adios2.ADIOS(comm)
        # ADIOS IO - Engine
        self.io_asynchronous = self.adios.DeclareIO("streamIO")
        self.io_asynchronous.SetEngine("SST")

        # Open the streams
        self.reader_st = self.io_asynchronous.Open(
            "globalArray_f2py", adios2.Mode.Read, comm
        )
        self.writer_st = self.io_asynchronous.Open(
            "globalArray_py2f", adios2.Mode.Write, comm
        )

        # Access header stream to calculate my element counts
        self.stepStatus = self.reader_st.BeginStep()

        hdr_elems = self.io_asynchronous.InquireVariable("global_elements")
        hdr_lxyz = self.io_asynchronous.InquireVariable("points_per_element")
        hdr_gdim = self.io_asynchronous.InquireVariable("problem_dimension")

        elems = np.zeros((1), dtype=np.intc)
        lxyz = np.zeros((1), dtype=np.intc)
        gdim = np.zeros((1), dtype=np.intc)

        self.reader_st.Get(hdr_elems, elems)
        self.reader_st.Get(hdr_lxyz, lxyz)
        self.reader_st.Get(hdr_gdim, gdim)

        self.reader_st.EndStep()  # Data is read here

        # Assign values
        self.glb_nelv = elems
        self.lxyz = lxyz
        self.gdim = gdim

        # Determine how many elements each reader rank should have
        element_mapping_load_balanced_linear(self, comm)

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
        self.py2f_field = self.io_asynchronous.DefineVariable(
            "py2f_field",
            tmp,
            [self.py2f_field_totalcount],
            [self.py2f_field_my_start],
            [self.py2f_field_my_count],
        )

        return

    def finalize(self):
        self.reader_st.Close()
        self.writer_st.Close()

    def stream(self, fld):

        # Begin a step
        stepStatus = self.writer_st.BeginStep()
        self.writer_st.Put(
            self.py2f_field,
            fld,
        )
        self.writer_st.EndStep()  # Data is sent here

        return

    def recieve(self, fld=None, variable="f2py_field"):

        if type(fld) == NoneType:
            fld = np.zeros((self.py2f_field_my_count), dtype=np.double)

        # Begin a step
        self.stepStatus = self.reader_st.BeginStep()

        if self.stepStatus == adios2.StepStatus.OK:
            # Set up the offsets
            f2py_field = self.io_asynchronous.InquireVariable(variable)
            f2py_field.SetSelection(
                [[self.py2f_field_my_start], [self.py2f_field_my_count]]
            )
            self.reader_st.Get(f2py_field, fld)

            self.reader_st.EndStep()  # Data is read here

        return fld


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

    return
