"""Module that defines classes to use adios2 to 
compress data in parallel"""

import json
import numpy as np
from ...datatypes.msh import Mesh
from ...datatypes.field import Field

# Adios2 is assumed to be available
try:
    # This import for ver >= adios2.10, otherwise just import adios2
    import adios2.bindings as adios2
except ImportError:
    print("Error: adios2 is not available. This is needed to compress data")
    raise

# Type definition
NoneType = type(None)


class DataCompressor:
    """
    Class used to write compressed data to disk.

    This Assumes that the input data has a msh object available.

    Parameters
    ----------
    comm : Comm
        MPI communicator.
    mesh_info : dict
        Dictionary with mesh information.
    wrd_size : int
        Word size to write data. (Default value = 4). Single precsiion is 4, double is 8.

    Returns
    -------

    Examples
    --------
    This class is used to write data to disk. The data is compressed using bzip2.

    >>> mesh_info = {"glb_nelv": msh.glb_nelv, "lxyz": msh.lxyz, "gdim": msh.gdim}
    >>> dc = DataCompressor(comm, mesh_info = mesh_info, wrd_size = 4)
    """

    def __init__(self, comm, mesh_info=None, wrd_size=4):

        # ADIOS2 instance
        self.adios = adios2.ADIOS(comm)
        # ADIOS IO - Engine
        self.io = self.adios.DeclareIO("compressIO")
        self.io.SetEngine("BP5")

        # Declare useful data
        self.glb_nelv = np.ones((1), dtype=np.ulong)
        self.lx = 0
        self.ly = 0
        self.lz = 0
        self.lxyz = np.ones((1), dtype=np.ulong)
        self.gdim = np.ones((1), dtype=np.ulong)
        self.wrd_size = np.ones((1), dtype=np.ulong) * wrd_size
        if wrd_size == 4:
            self.dtype = np.single
        elif wrd_size == 8:
            self.dtype = np.double
        self.known_mesh_info = False
        self.nelv = None
        self.offset_el = None
        self.n = None
        # If mesh info is available, determine the counts
        if not isinstance(mesh_info, NoneType):
            self.glb_nelv[0] = mesh_info["glb_nelv"]
            self.lxyz[0] = mesh_info["lxyz"]
            self.gdim[0] = mesh_info["gdim"]

            # Check element distribution
            element_mapping_load_balanced_linear(self, comm)

            self.totalcount = int(self.glb_nelv * self.lxyz)
            self.my_start = int(self.offset_el * self.lxyz)
            self.my_count = int(self.nelv * self.lxyz)

            self.known_mesh_info = True

        # Define dummies
        self.write_bp = None
        self.read_bp = None

    def write(self, comm, fname="compress.bp", variable_names=None, data=None):
        """
        Write data to disk using adios2.

        Lossless compression with bzip2.

        Parameters
        ----------
        comm : Comm
            MPI communicator.
        fname : str
            File name to write. (Default value = "compress.bp").
        variable_names : list
            List of string with the names of the variables to write.
            This is very important, as adios2 will use these names to process the data.
        data : list
            List of numpy arrays with the data to write. Corresponding in index to variable_names.
            The arrays must be 1d.

        Examples
        --------
        This function is used to write data to disk. The data is compressed using bzip2.

        >>> variable_names = ["x", "y", "z"]
        >>> data = [msh.x, msh.y, msh.z]
        >>> dc.write(comm, fname = "compress.bp", variable_names = variable_names, data = data)
        """

        # Check if the saved dtype is the same as input file, if not, perform a cast
        if self.dtype != data[0].dtype:
            if comm.Get_rank() == 0:
                string = f"Data type mismatch. Expected {self.dtype} but got {data[0].dtype}. Casting data."
                print(string)
            data = [dat.astype(self.dtype) for dat in data]

        # Open a file
        self.write_bp = self.io.Open(fname, adios2.Mode.Write, comm)
        # Begin the step
        step_status = self.write_bp.BeginStep()

        # Declare header adios2 variables
        tmp = np.zeros((1), dtype=np.ulong)
        hdr_elems = self.io.DefineVariable("global_elements", tmp)
        hdr_lxyz = self.io.DefineVariable("points_per_element", tmp)
        hdr_gdim = self.io.DefineVariable("problem_dimension", tmp)
        hdr_wrd_size = self.io.DefineVariable("word_size", tmp)

        # Declare adios2 variables
        tmp = np.zeros((1), dtype=self.dtype)
        variables = []
        for var_name in variable_names:
            variables.append(
                self.io.DefineVariable(
                    var_name, tmp, [self.totalcount], [self.my_start], [self.my_count]
                )
            )

        # Add operations
        for var in variables:
            var.AddOperation("bzip2", {"blockSize100k": "9"})

        # Write the header
        self.write_bp.Put(hdr_elems, self.glb_nelv)
        self.write_bp.Put(hdr_lxyz, self.lxyz)
        self.write_bp.Put(hdr_gdim, self.gdim)
        self.write_bp.Put(hdr_wrd_size, self.wrd_size)

        # Write the data
        for var, dat in zip(variables, data):
            self.write_bp.Put(var, dat)

        # End step and close file
        self.write_bp.EndStep()  # Data is sent here
        self.write_bp.Close()

        # Clean up
        self.io.RemoveAllVariables()
        self.io.RemoveAllAttributes()

    def read(self, comm, fname="compress.bp", variable_names=None):
        """
        Read data from disk using adios2.

        Read compressed data and internally decompress it.

        Parameters
        ----------
        comm : Comm
            MPI communicator.
        fname : str
            File name to read. (Default value = "compress.bp").
        variable_names : list
            List of string with the names of the variables to read.
            These names NEED to match the names adios2 used to write the data.

        Returns
        -------
        list
            List of numpy arrays with the data read.
            the ndarrays in the list are 1d.

        Examples
        --------
        This function is used to read data from disk. The data is compressed using bzip2.

        >>> variable_names = ["x", "y", "z"]
        >>> data = dc.read(comm, fname = "compress.bp", variable_names = variable_names)
        """

        # Opean the file and read the header
        self.read_bp = self.io.Open(fname, adios2.Mode.Read, comm)
        step_status = self.read_bp.BeginStep()

        if not self.known_mesh_info:
            hdr_elems = self.io.InquireVariable("global_elements")
            hdr_lxyz = self.io.InquireVariable("points_per_element")
            hdr_gdim = self.io.InquireVariable("problem_dimension")
            hdr_wrd_size = self.io.InquireVariable("word_size")

            # Read the header
            self.read_bp.Get(hdr_elems, self.glb_nelv)
            self.read_bp.Get(hdr_lxyz, self.lxyz)
            self.read_bp.Get(hdr_gdim, self.gdim)
            self.read_bp.Get(hdr_wrd_size, self.wrd_size)

            if self.wrd_size[0] == 4:
                self.dtype = np.single
            elif self.wrd_size[0] == 8:
                self.dtype = np.double

            if comm.Get_rank() == 0:
                string = f"Updated header from file to:: glb_nlv: {self.glb_nelv[0]}, lxyz: {self.lxyz[0]},  gdim: {self.gdim[0]}, dtype: {self.dtype}"
                print(string)

            # Check element distribution
            element_mapping_load_balanced_linear(self, comm)

            self.totalcount = int(self.glb_nelv * self.lxyz)
            self.my_start = int(self.offset_el * self.lxyz)
            self.my_count = int(self.nelv * self.lxyz)

            self.known_mesh_info = True

        # Allcate the data
        data = []
        for var_name in variable_names:
            data.append(np.zeros((self.my_count), dtype=self.dtype))

        # Inquire the variables form the file
        variables = []
        for var_name in variable_names:
            variables.append(self.io.InquireVariable(var_name))

        # Choose the extend of data that my rank needs to read
        for var in variables:
            var.SetSelection([[self.my_start], [self.my_count]])

        # Read the data
        for var, dat in zip(variables, data):
            self.read_bp.Get(var, dat)

        # Close the reader
        self.read_bp.EndStep()
        self.read_bp.Close()

        # Clean up
        self.io.RemoveAllVariables()
        self.io.RemoveAllAttributes()

        return data


def write_field(
    comm,
    msh=None,
    fld=None,
    fname="compressed_field0.f00001",
    wrd_size=4,
    write_mesh=True,
):
    """
    Wrapper to data compressor writer.

    Writes nek like data compressed.

    Parameters
    ----------
    comm : Comm
        MPI communicator.

    msh : Mesh
        Mesh object to write. (Default value = None).
    fld : Field
        Field object to write. (Default value = None).
    fname : str
        File name to write. (Default value = "compressed_field0.f00001").
    wrd_size : int
        Word size to write data. (Default value = 4). Single precsiion is 4, double is 8.
    write_mesh : bool
        Flag to write the mesh. (Default value = True).

    Examples
    --------
    This function is used to write data to disk. The data is compressed using bzip2.

    >>> write_field(comm, msh = msh, fld = fld, fname = "compressed_field0.f00001", wrd_size = 4, write_mesh = True)
    """

    # Inputs to write operation
    mesh_info = {"glb_nelv": msh.glb_nelv, "lxyz": msh.lxyz, "gdim": msh.gdim}

    # Set the variable names and reshape data to 0d (as in memory)
    if not write_mesh:
        variable_names = []
        variable_data = []
    else:
        variable_names = ["x", "y", "z"]
        variable_data = [msh.x, msh.y, msh.z]

    for key in fld.fields:
        lst = fld.fields[key]
        if len(lst) > 0:
            for index, data in enumerate(lst):
                variable_names.append(f"{key}_{index}")
                variable_data.append(data.reshape((data.size)))

    # Instance compressor and write
    dc = DataCompressor(comm, mesh_info=mesh_info, wrd_size=wrd_size)
    dc.write(comm, fname=fname, variable_names=variable_names, data=variable_data)

    if comm.Get_rank() == 0:
        with open(fname + "/variable_names.json", "w") as f:
            json.dump(variable_names, f)


def read_field(comm, fname="compressed_field0.f00001"):
    """
    Wrapper to data compressor reader.

    Reads nek like data compressed.

    Parameters
    ----------
    comm : Comm
        MPI communicator.
    fname : str
        File name to read. (Default value = "compressed_field0.f00001").

    Returns
    -------
    msh: Mesh
        Mesh object read.
    fld : Field
        Field object read.

    Examples
    --------

    This function is used to read data from disk. The data is compressed using bzip2.

    >>> msh, fld = read_field(comm, fname = "compressed_field0.f00001")
    """

    # Read variable names
    with open(fname + "/variable_names.json", "r") as f:
        variable_names = json.load(f)

    dc = DataCompressor(comm)
    variable_data = dc.read(comm, fname=fname, variable_names=variable_names)

    if dc.gdim == 3:
        dc.lx = int(np.cbrt(dc.lxyz[0]))
    else:
        dc.lx = int(np.sqrt(dc.lxyz[0]))
    dc.ly = dc.lx
    if dc.gdim == 3:
        dc.lz = dc.lx
    else:
        dc.lz = 1

    data_to_process = len(variable_data)
    
    process_counter = 0
    if variable_names[0] == "x":
        # Initialize a new mesh object
        x = variable_data[process_counter].reshape((dc.nelv, dc.lz, dc.ly, dc.lx))
        process_counter += 1
        y = variable_data[process_counter].reshape((dc.nelv, dc.lz, dc.ly, dc.lx))
        process_counter += 1
        if dc.gdim == 3:
            z = variable_data[process_counter].reshape((dc.nelv, dc.lz, dc.ly, dc.lx))
            process_counter += 1
        else:
            z = np.zeros_like(x)
        msh = Mesh(comm, x=x, y=y, z=z)
    else:
        msh = None

    fld = Field(comm)
    for i in range(process_counter, data_to_process):
        key = variable_names[i].split("_")[0]
        data = variable_data[i].reshape((dc.nelv, dc.lz, dc.ly, dc.lx))
        fld.fields[key].append(data)

    # Update the variable quantities
    fld.update_vars()

    return msh, fld


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
